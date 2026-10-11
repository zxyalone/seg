#!/usr/bin/env python3
"""Run the authorized A9/A11/A12 pilots, retain state, archive and publish.

Use an immutable worktree; this controller does not resume the E11/E12 queues.
Any newer global stop marker, or this controller's STOP file, cancels its jobs.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
PY = Path('/root/autodl-tmp/envs/mamba/bin/python')
ROOT = Path('/root/autodl-tmp/nnUNet_data/nnUNet_results_v2')
OUT = ROOT / '_pilot_lowcnr'
TRAINER = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
JOBS = [('A9', 0), ('A11', 1), ('A12', 2)]
STATE = {}
PROCESSES = []
START = None
CONTROL = None


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def save():
    STATE['updated_at'] = stamp()
    temp = OUT / 'state.tmp'
    temp.write_text(json.dumps(STATE, indent=2, ensure_ascii=False)+'\n')
    temp.replace(OUT / 'state.json')


def log(message):
    print(f'[{stamp()}] {message}', flush=True)


def stop_requested():
    markers = [OUT/'STOP', CONTROL/'experiments/analysis/STOP_ALL_WORK.json',
               ROOT/'_orchestration/all_work_stopped.json']
    return any(p.exists() and (p.name == 'STOP' or p.stat().st_mtime > START) for p in markers)


def check_stop():
    if stop_requested():
        raise KeyboardInterrupt('Pilot stopped by a new stop marker')


def terminate_jobs():
    for p in PROCESSES:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for p in PROCESSES:
        if p.poll() is None:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)


def command(args, stdout=None):
    check_stop()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=stdout,
                         stderr=subprocess.STDOUT, start_new_session=True)
    PROCESSES.append(p)
    while p.poll() is None:
        check_stop()
        time.sleep(5)
    if p.returncode:
        raise RuntimeError(f'Command failed ({p.returncode}): {args}')


def progress(group):
    text = (OUT/f'{group}.log').read_text(errors='replace')
    epochs = re.findall(r'\bEpoch (\d+)\b', text)
    times = [float(s) for s in re.findall(r'Epoch time: ([0-9.]+) s', text)]
    job = STATE['jobs'][group]
    job['latest_epoch'] = int(epochs[-1]) if epochs else None
    job['completed_epoch_times'] = len(times)
    if times:
        if 'first_epoch_seconds' not in job:
            job['first_epoch_seconds'] = times[0]
            log(f"{group}: first epoch {times[0]:.2f}s (A0 reference 80–88s)")
        job['last_epoch_seconds'] = times[-1]
        job['mean_recent_epoch_seconds'] = sum(times[-10:])/len(times[-10:])
        job['estimated_remaining_training_hours'] = max(200-len(times), 0)*job['mean_recent_epoch_seconds']/3600
        if group == 'A11':
            job['first_epoch_slow_relative_to_A0'] = times[0] > 1.5*88
    job['validation_cases_written'] = len(list((ROOT/f'{group}_s0'/TRAINER/'validation').glob('*.nii.gz')))


def archive(group):
    folder = ROOT/f'{group}_s0'/TRAINER
    manifest = json.loads((folder/'run_manifest.json').read_text())
    assert manifest['epochs_completed'] == manifest['num_epochs'] == 200
    assert manifest['fold'] == manifest['seed'] == 0
    assert manifest['num_iterations_per_epoch'] == 250
    assert manifest['env']['VOCOMNI_NNUNET_CKPT'] == STATE['pretrained_path']
    cases = (REPO/'experiments/eval/val_cases.txt').read_text().split()
    assert len(cases) == 21
    assert all((folder/'validation'/f'{c}.nii.gz').is_file() for c in cases)
    assert (folder/'checkpoint_final.pth').is_file()
    for name in ('cases.csv', 'lesions.csv', 'pred_lesions.csv', 'summary.json'):
        assert (folder/'eval_val'/name).is_file()
    target = REPO/f'results/{group}_s0'
    shutil.copytree(folder/'eval_val', target/'eval_val', dirs_exist_ok=True)
    for name in ('run_manifest.json', 'progress.png'):
        if (folder/name).is_file():
            shutil.copyfile(folder/name, target/name)
    shutil.copyfile(OUT/f'{group}.log', target/'pilot_run.log')


def main():
    global START, CONTROL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-root', default='/root/autodl-tmp/Downstream_2')
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    CONTROL = Path(args.control_root); OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    START = time.time()
    if (OUT/'STOP').exists():
        raise RuntimeError('Pilot STOP marker present; explicit removal is required')
    pretrained = CONTROL/'VoComni_nnunet.pt'
    assert pretrained.is_file(), pretrained
    statefile = OUT/'state.json'
    if statefile.exists():
        raise RuntimeError('Pilot state already exists; refuse an implicit second launch')
    STATE.update(phase='starting', pid=os.getpid(), started_at=stamp(), jobs={},
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
                 worktree=str(REPO), control_root=str(CONTROL),
                 pretrained_path=str(pretrained), pretrained_sha256=hashlib.sha256(pretrained.read_bytes()).hexdigest(),
                 scope='A9/A11/A12 seed 0, 200 epochs, fold0; A10 preflight only; no test-set evaluation',
                 stop_markers_before_start_retained=True, publish_status='pending' if args.publish else 'disabled')
    save()
    try:
        for group, gpu in JOBS:
            env = os.environ.copy()
            for k in list(env):
                if k.startswith('PHASER_') or k == 'EPOCHS':
                    del env[k]
            env.update(GPU=str(gpu), RESULTS_ROOT=str(ROOT), ENV_BIN=str(PY.parent),
                       VOCOMNI_NNUNET_CKPT=str(pretrained), nnUNet_n_proc_DA='6', nnUNet_def_n_proc='2',
                       EVAL_WORKERS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1', TZ='Asia/Shanghai')
            stream = (OUT/f'{group}.log').open('w')
            proc = subprocess.Popen(['bash', str(REPO/'experiments/run.sh'), group, '0'],
                                    cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            stream.close(); PROCESSES.append(proc)
            STATE['jobs'][group] = dict(pid=proc.pid, gpu=gpu, status='training_and_evaluating', logfile=str(OUT/f'{group}.log'))
            log(f'Started {group}_s0 on GPU {gpu}; PID {proc.pid}')
        STATE['phase'] = 'running'; save()
        while any(p.poll() is None for p in PROCESSES):
            check_stop()
            for (group, _), proc in zip(JOBS, PROCESSES):
                progress(group)
                if proc.poll() is not None and STATE['jobs'][group]['status'] == 'training_and_evaluating':
                    STATE['jobs'][group].update(status='complete' if proc.returncode == 0 else 'failed',
                                                exit_code=proc.returncode, ended_at=stamp())
                    log(f'{group}_s0 exited: {proc.returncode}')
            save(); time.sleep(15)
        for (group, _), proc in zip(JOBS, PROCESSES):
            progress(group)
            STATE['jobs'][group].update(status='complete' if proc.returncode == 0 else 'failed', exit_code=proc.returncode)
        save()
        if any(p.returncode for p in PROCESSES):
            raise RuntimeError('At least one pilot failed; retain all outputs and do not publish partial results as complete')
        STATE['phase'] = 'archiving'; save(); check_stop()
        for group, _ in JOBS:
            archive(group)
        target = REPO/'results/lowcnr_pilots'; target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OUT/'preflight', target/'preflight', dirs_exist_ok=True)
        STATE['phase'] = 'fixed_4mm_metrics'; save()
        with (OUT/'post4mm.log').open('w') as stream:
            command([str(PY), str(REPO/'experiments/analysis/lowcnr_pilot_metrics.py'), '--out', str(target)], stream)
        STATE['phase'] = 'complete'; save()
        shutil.copyfile(statefile, target/'state.json')
        if args.publish:
            paths = [f'results/{g}_s0' for g, _ in JOBS]+['results/lowcnr_pilots']
            command(['git','add','--',*paths])
            command(['git','commit','--only','-m','Results: archive all low-CNR pilots and fixed-4-mm gates','--',*paths])
            # Integrate unrelated remote updates in this isolated, clean worktree;
            # conflicts remain visible and no forced push is ever attempted.
            command(['git','fetch','origin'])
            command(['git','merge','--no-edit','origin/main'])
            command(['git','push','origin','HEAD:main'])
            STATE['publish_status'] = 'published'
            STATE['published_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
            save(); log(f"Published pilot results to main: {STATE['published_commit']}")
    except KeyboardInterrupt as e:
        terminate_jobs(); STATE.update(phase='cancelled', error=str(e)); save(); log(str(e))
    except Exception as e:
        terminate_jobs(); STATE.update(phase='failed', error=str(e)); save(); log(f'Failed: {e}'); raise


if __name__ == '__main__':
    def terminated(*_):
        raise KeyboardInterrupt('Pilot controller terminated')
    signal.signal(signal.SIGTERM, terminated)
    main()
