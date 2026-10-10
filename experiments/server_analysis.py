#!/usr/bin/env python3
"""Share E11–E13 validation analyses across three GPUs; never run stage-7 tests."""
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('RESULTS_ROOT', str(REPO.parent / 'nnUNet_data/nnUNet_results_v2')))
HERE = ROOT / '_orchestration'
OUT = ROOT / '_analysis'
PY = REPO.parent / 'envs/mamba/bin/python'
CASES_FILE = REPO / 'experiments/eval/val_cases.txt'
CASES = CASES_FILE.read_text().split()
FOLD = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
E11_RUNS = [f'A6_s{s}' for s in range(3)]
E12_RUNS = [f'{g}_s{s}' for g in ('A0', 'A1', 'A6') for s in range(3)] + ['A7_s0']
CONDITIONS = ['none'] + [f'pc_zero@{l}' for l in range(5)] + ['pc_zero@all', 'route_uniform', 'prior_zero', 'gate_zero']
ALPHAS = ['a1', 'a0.75', 'a0.5', 'a0.25']
LOCK = threading.RLock()
STOP = threading.Event()
STATE = {}


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def save():
    with LOCK:
        STATE['updated_at'] = stamp()
        temp = HERE / 'analysis_state.tmp'
        temp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2) + '\n')
        temp.replace(HERE / 'analysis_state.json')


def log(message):
    print(f'[{stamp()}] {message}', flush=True)


def env_for(gpu=None):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('PHASER_') or key in ('EXTERNAL_IMAGES', 'EXTERNAL_LABELS', 'EPOCHS'):
            del env[key]
    env.update(PYTHONPATH=str(REPO / 'nnUNet'), PATH=str(PY.parent) + ':' + env.get('PATH', ''),
               RESULTS_ROOT=str(ROOT), DATA=str(REPO.parent / 'nnUNet_data'), TZ='Asia/Shanghai',
               nnUNet_raw=str(REPO.parent / 'nnUNet_data/nnUNet_raw'),
               nnUNet_preprocessed=str(REPO.parent / 'nnUNet_data/nnUNet_preprocessed'), nnUNet_results=str(ROOT),
               nnUNet_def_n_proc='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='2', PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '')
    return env


def expected_files(experiment, cases=None):
    cases = CASES if cases is None else cases
    runs, conditions = (E11_RUNS, CONDITIONS) if experiment == 'E11' else (E12_RUNS, ALPHAS)
    return [OUT / experiment / 'pred' / run / cond / f'{case}.nii.gz'
            for run in runs for cond in conditions for case in cases]


def update_progress():
    with LOCK:
        for experiment in ('E11', 'E12'):
            paths = expected_files(experiment)
            STATE['items'][experiment]['completed_predictions'] = sum(p.is_file() for p in paths)
        e13_log = HERE / 'analysis_E13.log'
        if e13_log.exists():
            lines = e13_log.read_text(errors='replace')
            matches = re.findall(r'\[(\d+)/21\] (LiTS_\d+):', lines)
            STATE['items']['E13']['completed_cases'] = len({case for _, case in matches})
        save()


def monitor():
    while not STOP.is_set():
        update_progress()
        STOP.wait(15)


def execute(task, gpu):
    experiment, shard = task
    name = experiment if shard is None else f'{experiment}_shard{shard}'
    script = {'E11': 'e11_interventions.py', 'E12': 'e12_degradation.py', 'E13': 'e13_mechanism.py'}[experiment]
    command = [str(PY), '-u', str(REPO / 'experiments/analysis' / script)]
    if shard is not None:
        command += ['run', '--shard', f'{shard}/3']
    command += ['--device', 'cuda', '--out', str(OUT / experiment), '--cases-file', str(CASES_FILE)]
    with LOCK:
        if STATE['items'][experiment]['status'] != 'failed':
            STATE['items'][experiment]['status'] = 'running'
        STATE['jobs'][name] = dict(status='running', gpu=gpu, started_at=stamp(), command=command)
        save()
    log(f'START {name} GPU{gpu}')
    try:
        with (HERE / f'analysis_{name}.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=REPO, env=env_for(gpu), stdout=stream, stderr=subprocess.STDOUT)
            with LOCK:
                STATE['jobs'][name]['pid'] = process.pid
                save()
            assert process.wait() == 0, name + ' command failed'
        if shard is not None:
            assert all(p.is_file() for p in expected_files(experiment, CASES[shard::3])), 'Missing predictions'
            detail = 'utility' if experiment == 'E11' else 'lesions'
            for case in CASES[shard::3]:
                records = [OUT / experiment / detail / r / f'{case}.json' for r in E11_RUNS] if experiment == 'E11' else [OUT / experiment / detail / f'{case}.json']
                for record in records:
                    data = json.loads(record.read_text())
                    if experiment == 'E11':
                        assert set(data['loss']) == set(CONDITIONS)
        else:
            summary = json.loads((OUT / 'E13/summary.json').read_text())
            assert set(summary['runs']) == set(E11_RUNS + ['A7_s0'])
            for filename in ['filters.csv', 'lesion_level_stats.csv', 'pc_by_level_stratum.csv', 'route_by_level.csv', 'summary.md']:
                assert (OUT / 'E13' / filename).is_file(), filename
        good, error = True, None
    except Exception as exc:
        good, error = False, repr(exc)
    with LOCK:
        STATE['jobs'][name].update(status='complete' if good else 'failed', ended_at=stamp(), error=error)
        if not good:
            STATE['items'][experiment].update(status='failed', error=error)
        elif experiment == 'E13':
            STATE['items'][experiment]['status'] = 'complete'
        elif all(STATE['jobs'].get(f'{experiment}_shard{i}', {}).get('status') == 'complete' for i in range(3)):
            STATE['items'][experiment]['status'] = 'predictions_complete'
        save()
    log(f'END {name} success={good} {error or ""}')
    return good


def summarize(experiment):
    script = 'e11_interventions.py' if experiment == 'E11' else 'e12_degradation.py'
    with LOCK:
        STATE['items'][experiment]['status'] = 'summarizing'
        save()
    log(f'SUMMARIZE {experiment}')
    try:
        with (HERE / f'analysis_{experiment}_summary.log').open('a') as stream:
            subprocess.run([str(PY), '-u', str(REPO / 'experiments/analysis' / script), 'summarize',
                            '--workers', '2', '--out', str(OUT / experiment), '--cases-file', str(CASES_FILE)],
                           cwd=REPO, env=env_for(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        checks = []
        for run in E11_RUNS if experiment == 'E11' else E12_RUNS:
            cond = 'none' if experiment == 'E11' else 'a1'
            current = json.loads((OUT / experiment / 'eval' / run / cond / 'summary.json').read_text())
            ref = json.loads((ROOT / run / FOLD / 'eval_val/summary.json').read_text())
            delta = abs(current['Tumor']['tumor_dice_mean'] - ref['Tumor']['tumor_dice_mean'])
            checks.append({'run': run, 'abs_tumor_dice_diff': delta})
            assert current['All']['n_cases'] == 21 and delta <= 1e-6, f'{run} baseline did not reproduce eval_val: {delta}'
        if experiment == 'E12':
            check = json.loads((OUT / experiment / 'e12_cnr_check.json').read_text())
            for alpha in (1, .75, .5, .25):
                assert abs(check[f'a{alpha:g}'] - alpha) < .02, 'Realised CNR check failed'
        (OUT / experiment / 'baseline_integrity.json').write_text(json.dumps(checks, indent=2) + '\n')
        good, error = True, None
    except Exception as exc:
        good, error = False, repr(exc)
    with LOCK:
        STATE['items'][experiment].update(status='complete' if good else 'failed', error=error)
        save()
    log(f'SUMMARY END {experiment} success={good} {error or ""}')
    return good


def archive():
    paths = []
    for experiment in ('E11', 'E12', 'E13'):
        if STATE['items'][experiment]['status'] != 'complete':
            continue
        destination = REPO / 'results' / experiment
        for source in (OUT / experiment).rglob('*'):
            relative = source.relative_to(OUT / experiment)
            if source.is_file() and 'pred' not in relative.parts and source.suffix in ('.csv', '.md', '.json', '.png'):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        paths.append(str(destination.relative_to(REPO)))
    target = REPO / 'results/server/analysis_state.json'
    target.write_text(json.dumps(STATE, ensure_ascii=False, indent=2) + '\n')
    paths.append(str(target.relative_to(REPO)))
    subprocess.run(['git', 'add', '--', *paths], cwd=REPO, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=REPO).returncode:
        subprocess.run(['git', 'commit', '-m', 'Results: validation E11-E13 analyses and integrity checks'], cwd=REPO, check=True)


def main():
    HERE.mkdir(exist_ok=True)
    with (HERE / 'analysis_controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = HERE / 'analysis_state.json'
        if prior.exists():
            for job in json.loads(prior.read_text()).get('jobs', {}).values():
                pid = job.get('pid')
                cmd = Path(f'/proc/{pid}/cmdline')
                if job['status'] == 'running' and cmd.exists() and b'experiments/analysis/' in cmd.read_bytes():
                    raise RuntimeError('An earlier analysis process is still running; refusing to duplicate it')
        assert len(CASES) == len(set(CASES)) == 21
        STATE.update(pid=os.getpid(), phase='selftest', source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                     started_at=stamp(), jobs={}, items={'E11': {'status': 'pending', 'total_predictions': 630, 'completed_predictions': 0},
                     'E12': {'status': 'pending', 'total_predictions': 840, 'completed_predictions': 0},
                     'E13': {'status': 'pending', 'total_cases': 21, 'completed_cases': 0}},
                     script_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO / 'experiments/analysis').glob('*.py')},
                     test_set='locked; no test inference scheduled', baseline_tumor_dice_tolerance=1e-6,
                     cnr_median_ratio_tolerance=.02,
                     stage7_prerequisites=['E10 postprocessing', 'fixed primary endpoint', 'E14 external cohort', 'E17 pretraining overlap', 'PROTOCOL_FROZEN'])
        save()
        deadline = time.time() + 1800
        while True:
            selftest = (HERE / 'analysis_selftest.log').read_text(errors='replace')
            if 'SELFTEST_EXIT=0' in selftest and 'test_analysis: all checks passed' in selftest:
                break
            if 'SELFTEST_EXIT=' in selftest:
                raise RuntimeError('Analysis self-test failed; GPU queue not started')
            if time.time() > deadline:
                raise RuntimeError('Analysis self-test did not finish within 30 minutes')
            time.sleep(5)
        STATE.update(phase='running', selftest='passed', selftest_log=str(HERE / 'analysis_selftest.log'))
        save()
        log('Self-test passed; starting shared E11-E13 GPU queue on validation cases only')
        threading.Thread(target=monitor, daemon=True).start()
        ready = queue.Queue()
        for task in [('E13', None)] + [('E11', i) for i in range(3)] + [('E12', i) for i in range(3)]:
            ready.put(task)
        summaries = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as cpu:
            def worker(gpu):
                while True:
                    try:
                        task = ready.get_nowait()
                    except queue.Empty:
                        return
                    execute(task, gpu)
                    experiment = task[0]
                    with LOCK:
                        if experiment in ('E11', 'E12') and STATE['items'][experiment]['status'] == 'predictions_complete' and experiment not in summaries:
                            summaries[experiment] = cpu.submit(summarize, experiment)
                    ready.task_done()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as gpu:
                list(gpu.map(worker, range(3)))
            for future in summaries.values():
                future.result()
        STOP.set()
        update_progress()
        STATE.update(phase='complete_stage7_prerequisites_pending' if all(r['status'] == 'complete' for r in STATE['items'].values()) else 'analysis_failed', ended_at=stamp())
        save()
        archive()
        log('Validation analysis queue finished; stage-7 prerequisites still apply; test set remains locked')


if __name__ == '__main__':
    os.environ['TZ'] = 'Asia/Shanghai'
    time.tzset()
    try:
        main()
    except Exception as exc:
        if STATE:
            STATE.update(phase='analysis_failed', attention=repr(exc))
            save()
        log('FAILED ' + repr(exc))
        raise
    finally:
        STOP.set()
