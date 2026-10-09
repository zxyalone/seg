#!/usr/bin/env python3
"""Checklist stages 4–6: adopt running stage 4, then share work across three GPUs.

Run in tmux. Training/configuration files are used as supplied. Test prediction
is intentionally gated on the checklist's later analysis and protocol freeze.
"""
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('DATA', str(REPO.parent / 'nnUNet_data')))
BIN = Path(os.environ.get('ENV_BIN', str(REPO.parent / 'envs/mamba/bin')))
ROOT = Path(os.environ.get('RESULTS_ROOT', str(DATA / 'nnUNet_results_v2')))
HERE = ROOT / '_orchestration'
FOLD = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
RESFOLD = 'Dataset102_LiTS/nnUNetTrainer_200epochs__nnUNetResEncUNetPlans__3d_fullres/fold_0'
LOCK = threading.RLock()
STOP_MONITOR = threading.Event()
STATE = {}
PREPROCESS_CONFIG_KEYS = ('spacing', 'preprocessor_name', 'normalization_schemes', 'use_mask_for_norm',
                         'resampling_fn_data', 'resampling_fn_data_kwargs', 'resampling_fn_seg',
                         'resampling_fn_seg_kwargs')
PREPROCESS_PLAN_KEYS = ('transpose_forward', 'transpose_backward', 'image_reader_writer',
                       'foreground_intensity_properties_per_channel', 'label_manager', 'dataset_name')


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def save():
    with LOCK:
        STATE['updated_at'] = stamp()
        tmp = HERE / 'state.tmp'
        tmp.write_text(json.dumps(STATE, indent=2, ensure_ascii=False))
        tmp.replace(HERE / 'state.json')


def log(message):
    print(f'[{stamp()}] {message}', flush=True)


def env_for(gpu):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('PHASER_') or key in ('EPOCHS', 'EXTERNAL_IMAGES', 'EXTERNAL_LABELS'):
            del env[key]
    env.update(DATA=str(DATA), ENV_BIN=str(BIN), RESULTS_ROOT=str(ROOT), GPU=str(gpu),
               VOCOMNI_NNUNET_CKPT=str(REPO / 'VoComni_nnunet.pt'),
               nnUNet_raw=str(DATA / 'nnUNet_raw'), nnUNet_preprocessed=str(DATA / 'nnUNet_preprocessed'),
               nnUNet_results=str(ROOT), nnUNet_n_proc_DA='6', nnUNet_def_n_proc='4',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONPATH=str(REPO / 'nnUNet'), CUDA_VISIBLE_DEVICES=str(gpu),
               TZ='Asia/Shanghai', PATH=str(BIN) + ':' + env.get('PATH', ''))
    return env


def pid_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
        if stat[0] == 'Z':
            return None
        return stat[19]  # /proc field 22: process start time
    except FileNotFoundError:
        return None


def finished(group, seed=0):
    name = f'{group}_s{seed}'
    fold = ROOT / name / (RESFOLD if group == 'ResEnc' else FOLD)
    checkpoint = fold / 'checkpoint_final.pth'
    summary_path = fold / 'eval_val/summary.json'
    if not checkpoint.is_file() or not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    expected = set((REPO / 'experiments/eval/val_cases.txt').read_text().split())
    with (fold / 'eval_val/cases.csv').open() as stream:
        actual = [row['case'] for row in csv.DictReader(stream)]
    if set(actual) != expected or len(actual) != len(expected) or summary['All']['n_cases'] != 21:
        return False
    if group != 'ResEnc':
        manifest = json.loads((fold / 'run_manifest.json').read_text())
        if manifest['epochs_completed'] != 200 or manifest['num_epochs'] != 200:
            return False
        if not manifest.get('checkpoint_final_sha256'):
            return False
    lines = (ROOT / name / 'launcher.log').read_text(errors='replace').lower()
    # Skipped non-finite batches are reported in the audit, not a completion veto.
    return not any(word in lines for word in ('out of memory', 'traceback', 'cuda error'))


def monitor():
    while not STOP_MONITOR.is_set():
        try:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True, timeout=10)
            rows = [{'gpu': int(a), 'memory_mib': int(b), 'util_percent': int(c)}
                    for a, b, c in csv.reader(gpu.splitlines(), skipinitialspace=True)]
            with LOCK:
                STATE['gpu_snapshot'] = rows
                for name, job in STATE.get('jobs', {}).items():
                    path = ROOT / name / 'launcher.log'
                    if not path.exists():
                        continue
                    lines = path.read_text(errors='replace').splitlines()
                    epochs = [line.split('Epoch ', 1)[1] for line in lines if 'Epoch ' in line and 'Epoch time' not in line]
                    if epochs:
                        job['latest_epoch'] = epochs[-1]
                    bad = [line for line in lines if any(word in line.lower() for word in
                           ('non-finite', 'out of memory', 'traceback', 'cuda error'))]
                    if bad:
                        job['warnings'] = bad[-3:]
                save()
            with (HERE / 'gpu_usage.jsonl').open('a') as stream:
                stream.write(json.dumps({'time': stamp(), 'stage': STATE['stage'], 'gpus': rows}) + '\n')
        except Exception as error:
            log('MONITOR ' + repr(error))
        STOP_MONITOR.wait(30)


def adopt_stage4(old_controller, controller_lock):
    old = json.loads((HERE / 'state.json').read_text())
    assert old['stage'] == 'stage4', 'Only an in-progress stage 4 can be adopted'
    identities = {}
    for group in ('A6', 'A0', 'A1'):
        job = old['jobs'][group + '_s0']
        identities[group] = pid_identity(job['pid'])
        if identities[group] is None:
            assert finished(group), f'{group} already exited without complete validation'
        else:
            command = Path(f"/proc/{job['pid']}/cmdline").read_bytes().replace(b'\0', b' ')
            assert f'experiments/run.sh {group} 0'.encode() in command, f'Unexpected PID for {group}'
    assert old['pid'] == old_controller
    command = Path(f'/proc/{old_controller}/cmdline').read_bytes()
    assert any(name in command for name in (b'schedule_three_gpu.py', b'server_three_gpu.py')), \
        'Refusing to stop an unrelated process'
    if old.get('controller') == 'tmux shared GPU queue':
        snapshot = HERE / 'schedule_before_report_only_audit.json'
    else:
        snapshot = HERE / 'schedule_before_shared_queue.json'
    snapshot.write_text(json.dumps(old, indent=2, ensure_ascii=False))
    os.kill(old_controller, signal.SIGTERM)  # controller only; existing GPU jobs are preserved
    for _ in range(50):
        if pid_identity(old_controller) is None:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError('Old controller did not stop')
    # A running shared-queue controller owns this lock until it exits.
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    STATE.update(old)
    STATE.update(pid=os.getpid(), controller='tmux shared GPU queue', adopted_at=stamp(),
                 git_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                 stage6_policy='FIFO checklist order; next available GPU takes the next ready job',
                 gate_document_sha256=hashlib.sha256((REPO / 'experiments/DECISION_GATE.md').read_bytes()).hexdigest(),
                 nonfinite_policy='report counts only; continue automatically when outputs are complete')
    STATE.pop('stage6_lanes', None)
    for key in ('stage6_common_lanes', 'branch_a_extra'):
        STATE.pop(key, None)
    (HERE / 'scheduler.pid').write_text(str(os.getpid()) + '\n')
    save()
    log('Adopted stage-4 jobs without stopping or restarting GPU training')

    def observe(group):
        job = STATE['jobs'][group + '_s0']
        identity = identities[group]
        while identity is not None and pid_identity(job['pid']) == identity:
            time.sleep(5)
        good = finished(group)
        with LOCK:
            job.update(status='complete' if good else 'failed', ended_at=stamp(),
                       completion_evidence='final checkpoint + 200 epoch manifest + exact 21-case evaluation + no hard errors; non-finite warnings reported separately')
            save()
        log(f'Stage4 END {group}_s0 complete={good}')
        return good

    threading.Thread(target=monitor, daemon=True).start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        return all(list(pool.map(observe, ('A6', 'A0', 'A1'))))


def commit_results(message, paths):
    subprocess.run(['git', 'add', '--', *paths], cwd=REPO, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=REPO).returncode:
        subprocess.run(['git', 'commit', '-m', message], cwd=REPO, check=True)


def copy_evaluation(name):
    fold = ROOT / name / (RESFOLD if name.startswith('ResEnc') else FOLD)
    dest = REPO / 'results' / name
    shutil.copytree(fold / 'eval_val', dest / 'eval_val', dirs_exist_ok=True)
    if (fold / 'run_manifest.json').exists():
        shutil.copy2(fold / 'run_manifest.json', dest / 'run_manifest.json')
    return str(dest.relative_to(REPO))


def audit_stage4():
    """Print the requested warning counts before automatically evaluating the gate."""
    STATE['stage'] = 'stage4_audit'
    save()
    report = HERE / 'stage4_nonfinite_audit.log'
    env = env_for(0)
    env.update(R=str(ROOT), F=FOLD)
    with report.open('w') as stream:
        result = subprocess.run(['bash', str(REPO / 'experiments/audit_stage4_nonfinite.sh')],
                                cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    print(report.read_text(), end='', flush=True)
    status = {'status': 'complete' if result.returncode == 0 else 'failed',
              'exit_code': result.returncode, 'report': str(report), 'finished_at': stamp(),
              'policy': 'non-finite counts are informational; complete outputs proceed automatically'}
    (HERE / 'stage4_nonfinite_audit_status.json').write_text(json.dumps(status, indent=2))
    STATE['nonfinite_audit'] = status
    save()
    if result.returncode:
        raise RuntimeError('Stage-4 diagnostic command failed; inspect the audit log')
    destination = REPO / 'results/server/stage4_nonfinite_audit.log'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report, destination)


def gate():
    STATE['stage'] = 'stage5'
    save()
    paths = ['results/server/stage4_nonfinite_audit.log']
    paths += [copy_evaluation(group + '_s0') for group in ('A0', 'A1', 'A6')]
    for cohort, filename in [('Tumor', 'gate_val.csv'), ('All', 'gate_val_all.csv')]:
        cmd = [str(BIN / 'python'), str(REPO / 'experiments/eval/stats.py'), '--cohort', cohort]
        for group in ('A0', 'A1', 'A6'):
            cmd += ['--method', f'{group}={ROOT / (group + "_s0") / FOLD / "eval_val"}']
        cmd += ['--compare', 'A6:A0', 'A6:A1', '--out', str(REPO / 'results' / filename)]
        with (HERE / ('gate_' + cohort + '.log')).open('w') as stream:
            subprocess.run(cmd, cwd=REPO, env=env_for(0), check=True, stdout=stream, stderr=subprocess.STDOUT)
        paths.append('results/' + filename)
    with (REPO / 'results/gate_val.csv').open() as stream:
        rows = {row['metric']: row for row in csv.DictReader(stream) if row['comparison'] == 'A6 - A1'}
    dice, f1, fp = [float(rows[key]['delta']) for key in ('tumor_dice', 'lesion_f1@0.1', 'fp_per_scan@0.1')]
    assert all(math.isfinite(x) for x in (dice, f1, fp)), 'Non-finite gate; no stage-6 launch'
    branch = 'A' if dice > 0 and (f1 > 0 or fp < 0) else 'B'
    STATE['gate'] = {'branch': branch, 'delta_dice': dice, 'delta_f1': f1, 'delta_fp_per_scan': fp, 'decided_at': stamp()}
    save()
    (REPO / 'results/gate_decision.json').write_text(json.dumps(STATE['gate'], indent=2))
    paths.append('results/gate_decision.json')
    commit_results('Results: stage-4 validation and frozen stage-5 decision', paths)
    log(f'GATE branch={branch}; Dice={dice} F1={f1} FP/scan={fp}')
    return branch


def preprocessing_differences(old, new):
    a, b = old['configurations']['3d_fullres'], new['configurations']['3d_fullres']
    return {**{key: [a.get(key), b.get(key)] for key in PREPROCESS_CONFIG_KEYS if a.get(key) != b.get(key)},
            **{key: [old.get(key), new.get(key)] for key in PREPROCESS_PLAN_KEYS if old.get(key) != new.get(key)}}


def prepare_resenc(env, stream):
    pre = DATA / 'nnUNet_preprocessed/Dataset102_LiTS'
    subprocess.run([str(BIN / 'nnUNetv2_plan_experiment'), '-d', '102', '-pl', 'ResEncUNetPlanner'],
                   cwd=REPO, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
    old = json.loads((pre / 'nnUNetPlans.json').read_text())
    new = json.loads((pre / 'nnUNetResEncUNetPlans.json').read_text())
    diff = preprocessing_differences(old, new)
    config = new['configurations']['3d_fullres']
    target = pre / config['data_identifier']
    source = pre / old['configurations']['3d_fullres']['data_identifier']
    proof = {'differences': diff, 'source': str(source), 'target': str(target),
             'patch_size': config['patch_size'], 'batch_size': config['batch_size'], 'time': stamp()}
    if not diff:
        cases = json.loads((pre / 'dataset.json').read_text())['training']
        assert all((source / (case + suffix)).is_file() for case in cases for suffix in ('.npz', '.pkl', '.npy', '_seg.npy'))
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
        elif target.is_symlink():
            assert target.resolve() == source.resolve(), 'Unexpected existing ResEnc cache alias'
        else:
            assert all((target / (case + suffix)).is_file() for case in cases for suffix in ('.npz', '.pkl'))
        proof['cache'] = 'same preprocessing; reuse existing cases; no source overwrite'
        log('E9: exact preprocessing match, reusing existing cache without a second 60+ GiB array copy')
    else:
        need = sum(path.stat().st_size for path in source.iterdir() if path.is_file()) + 5 * 2**30
        if shutil.disk_usage(DATA).free < need:
            raise RuntimeError('ResEnc preprocessing differs and available disk cannot fit a new copy')
        subprocess.run([str(BIN / 'nnUNetv2_preprocess'), '-d', '102', '-plans_name', 'nnUNetResEncUNetPlans', '-c', '3d_fullres'],
                       cwd=REPO, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
        proof['cache'] = 'separate native preprocessing'
    (HERE / 'resenc_preprocessing_proof.json').write_text(json.dumps(proof, indent=2))
    return config


def resenc(gpu, stream):
    env = env_for(gpu)
    env['nnUNet_results'] = str(ROOT / 'ResEnc_s0')
    config = prepare_resenc(env, stream)
    cmd = [str(BIN / 'nnUNetv2_train'), '102', '3d_fullres', '0', '-p', 'nnUNetResEncUNetPlans', '-tr', 'nnUNetTrainer_200epochs']
    fold = ROOT / 'ResEnc_s0' / RESFOLD
    if not (fold / 'checkpoint_final.pth').exists():
        if (fold / 'checkpoint_latest.pth').exists():
            cmd += ['--c']
        subprocess.run(cmd, cwd=REPO, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
    images = HERE / 'resenc_val_images'
    images.mkdir(exist_ok=True)
    for case in (REPO / 'experiments/eval/val_cases.txt').read_text().split():
        link = images / f'{case}_0000.nii.gz'
        if not link.exists():
            link.symlink_to(DATA / 'nnUNet_raw/Dataset102_LiTS/imagesTr' / link.name)
    subprocess.run([str(BIN / 'nnUNetv2_predict'), '-i', str(images), '-o', str(fold / 'validation_no_tta'),
                    '-d', '102', '-c', '3d_fullres', '-tr', 'nnUNetTrainer_200epochs', '-p', 'nnUNetResEncUNetPlans',
                    '-f', '0', '-chk', 'checkpoint_final.pth', '--disable_tta', '-device', 'cuda', '-npp', '1', '-nps', '1'],
                   cwd=REPO, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
    subprocess.run([str(BIN / 'python'), str(REPO / 'experiments/eval/evaluate.py'),
                    '--pred-dir', str(fold / 'validation_no_tta'), '--gt-dir', str(DATA / 'nnUNet_raw/Dataset102_LiTS/labelsTr'),
                    '--image-dir', str(DATA / 'nnUNet_raw/Dataset102_LiTS/imagesTr'),
                    '--cases-file', str(REPO / 'experiments/eval/val_cases.txt'), '--out-dir', str(fold / 'eval_val')],
                   cwd=REPO, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
    (fold / 'execution_record.json').write_text(json.dumps({'epochs': 200, 'patch_size': config['patch_size'],
         'batch_size': config['batch_size'], 'validation_tta': False}, indent=2))


def execute(job, gpu):
    group, seed = job
    name = f'{group}_s{seed}' if group != 'Cost' else 'E16_cost'
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    with LOCK:
        STATE['jobs'][name] = {'gpu': gpu, 'status': 'running', 'started_at': stamp()}
        save()
    log(f'START {name} on GPU{gpu}')
    try:
        with (folder / 'launcher.log').open('a') as stream:
            if group == 'ResEnc':
                resenc(gpu, stream)
            else:
                cmd = (['bash', str(REPO / 'experiments/run.sh'), group, str(seed)] if group != 'Cost' else
                       [str(BIN / 'python'), str(REPO / 'experiments/cost.py'), '--groups', 'A0', 'A1', 'A6', '--device', 'cuda'])
                process = subprocess.Popen(cmd, cwd=REPO, env=env_for(gpu), stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT)
                with LOCK:
                    STATE['jobs'][name]['pid'] = process.pid
                    save()
                assert process.wait() == 0, f'{name} command failed'
        assert group == 'Cost' or finished(group, seed), f'{name} output integrity check failed'
        good, error = True, None
    except Exception as exc:
        good, error = False, repr(exc)
        log(f'FAILED {name}: {error}')
    with LOCK:
        STATE['jobs'][name].update(status='complete' if good else 'failed', ended_at=stamp())
        if error:
            STATE['jobs'][name]['error'] = error
        save()
    log(f'END {name} success={good}')
    return good


def stage6_plan(branch, include_a8=True):
    if branch == 'B':
        jobs = [('A6', 1), ('A0', 1), ('A1', 1), ('A6', 2), ('A0', 2), ('A1', 2), ('A7', 0)]
    else:
        jobs = [('A7', 0), ('A6', 1), ('A0', 1), ('A6', 2), ('A0', 2), ('A5', 0)]
    if include_a8:
        jobs.append(('A8', 0))
    if branch == 'A':
        jobs.append(('ResEnc', 0))
    return jobs + [('Cost', 0)]


def job_name(job):
    group, seed = job
    return f'{group}_s{seed}' if group != 'Cost' else 'E16_cost'


def adopt_stage6(old_controller, jobs):
    """Suspend only the old supervisor so its terminal cannot hang up live jobs."""
    with (HERE / 'handoff.lock').open('w') as handoff:
        fcntl.flock(handoff, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads((HERE / 'state.json').read_text())
        assert old['stage'] == 'stage6' and old['pid'] == old_controller
        command = Path(f'/proc/{old_controller}/cmdline').read_bytes()
        assert b'server_three_gpu.py' in command, 'Unexpected controller process'
        retired_lock = HERE / f'controller_suspended_{old_controller}.lock'
        assert not retired_lock.exists(), 'Handoff was already started'
        handle, renamed, committed = None, False, False
        os.kill(old_controller, signal.SIGSTOP)
        try:
            for _ in range(50):
                status = Path(f'/proc/{old_controller}/stat').read_text().rpartition(')')[2].split()[0]
                if status in ('T', 't'):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError('Old controller did not suspend')
            old = json.loads((HERE / 'state.json').read_text())
            assert old['stage'] == 'stage6' and old['pid'] == old_controller
            plan_names = {job_name(job) for job in jobs}
            adopted = {}
            for name, record in old['jobs'].items():
                if record['status'] != 'running':
                    continue
                assert name in plan_names, f'Active job {name} is outside the new plan'
                pid = record.get('pid')
                assert pid, f'Missing PID for {name}'
                identity = pid_identity(pid)
                if identity is not None:
                    group, seed = name.rsplit('_s', 1)
                    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
                    assert f'experiments/run.sh {group} {seed}'.encode() in cmd
                else:
                    group, seed = name.rsplit('_s', 1)
                    assert finished(group, int(seed)), f'{name} exited without complete outputs'
                assert record['gpu'] not in adopted, 'Multiple jobs on one GPU'
                group, seed = name.rsplit('_s', 1)
                adopted[record['gpu']] = {'job': (group, int(seed)), 'pid': pid, 'identity': identity}
            (HERE / 'schedule_before_multiseed_update.json').write_text(json.dumps(old, indent=2, ensure_ascii=False))
            (HERE / 'controller.lock').rename(retired_lock)
            renamed = True
            handle = (HERE / 'controller.lock').open('w')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            STATE.update(old)
            STATE.update(pid=os.getpid(), git_commit=subprocess.check_output(
                ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                stage6_queue=[f'{group}:{seed}' for group, seed in jobs],
                stage6_optional=['A8:0'] if ('A8', 0) in jobs else [],
                stage6_priority_reason='Match three seeds for A0/A1/A6 before pending controls; preserve already-running A7',
                stage6_updated_at=stamp(), stage6_adopted=[job_name(item['job']) for item in adopted.values()],
                suspended_controller={'pid': old_controller, 'status': 'suspended_until_adopted_jobs_exit'})
            (HERE / 'scheduler.pid').write_text(str(os.getpid()) + '\n')
            save()
            committed = True
            log('Stage6 handoff: preserved all running GPU jobs; prioritized matched A1 seeds; gate unchanged')
            threading.Thread(target=monitor, daemon=True).start()
            return handle, adopted
        except Exception:
            if not committed:
                if handle:
                    handle.close()
                if renamed:
                    retired_lock.replace(HERE / 'controller.lock')
                (HERE / 'state.json').write_text(json.dumps(old, indent=2, ensure_ascii=False))
                (HERE / 'scheduler.pid').write_text(str(old_controller) + '\n')
                STATE.clear()
                os.kill(old_controller, signal.SIGCONT)
            raise


def retire_suspended_controller():
    record = STATE.get('suspended_controller')
    if not record or record['status'] == 'retired':
        return
    pid = record['pid']
    if pid_identity(pid) is not None:
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        assert b'server_three_gpu.py' in command
        os.kill(pid, signal.SIGKILL)  # Only after its original GPU jobs have exited.
    record.update(status='retired', retired_at=stamp())
    save()


def shared_queue(jobs, run=execute, adopted=None):
    adopted = adopted or {}
    ready = queue.Queue()
    for job in jobs:
        ready.put(job)
    outcomes = []
    remaining_adopted = set(adopted)
    def worker(gpu):
        if gpu in adopted:
            item = adopted[gpu]
            while item['identity'] is not None and pid_identity(item['pid']) == item['identity']:
                time.sleep(5)
            group, seed = item['job']
            good = finished(group, seed)
            with LOCK:
                STATE['jobs'][job_name(item['job'])].update(status='complete' if good else 'failed', ended_at=stamp())
                outcomes.append((item['job'], good))
                remaining_adopted.remove(gpu)
                if not remaining_adopted:
                    retire_suspended_controller()
                save()
            log(f'Adopted END {job_name(item["job"])} success={good}; GPU{gpu} takes the next priority job')
        while True:
            try:
                job = ready.get_nowait()
            except queue.Empty:
                return
            with LOCK:
                label = f'{job[0]}:{job[1]}'
                if label in STATE.get('stage6_pending_queue', []):
                    STATE['stage6_pending_queue'].remove(label)
                    save()
            result = run(job, gpu)
            with LOCK:
                outcomes.append((job, result))
            ready.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker, range(3)))
    return outcomes


def finish_stage6(outcomes):
    paths = [copy_evaluation(f'{group}_s{seed}') for (group, seed), good in outcomes if good and group != 'Cost']
    if paths:
        commit_results('Results: stage-6 training and validation evidence', paths)
    STATE['stage'] = 'training_complete_analysis_pending' if all(good for _, good in outcomes) else 'stage6_has_failed_jobs'
    STATE['pending_checklist'] = ['E10 postprocessing selection', 'E11 interventions', 'E12 contrast degradation',
                                  'E13 trained mechanism analysis', 'E14 external cohort', 'stage7 protocol freeze and one-time test']
    save()
    log('Training queue finished. Analysis and protocol prerequisites precede stage7; no test set prediction was launched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-controller', type=int, required=True)
    parser.add_argument('--resume-stage6', action='store_true', help='Adopt existing stage-6 jobs and update pending priorities')
    parser.add_argument('--skip-a8', action='store_true', help='Omit the optional final A8 run')
    args = parser.parse_args()
    HERE.mkdir(exist_ok=True)
    if args.resume_stage6:
        old = json.loads((HERE / 'state.json').read_text())
        jobs = stage6_plan(old['gate']['branch'], include_a8=not args.skip_a8)
        handle, adopted = adopt_stage6(args.old_controller, jobs)
        occupied = {job_name(item['job']) for item in adopted.values()}
        pending = [job for job in jobs if job_name(job) not in occupied and
                   STATE['jobs'].get(job_name(job), {}).get('status') != 'complete']
        prior_results = [(job, True) for job in jobs if job_name(job) not in occupied and
                         STATE['jobs'].get(job_name(job), {}).get('status') == 'complete']
        STATE['stage6_pending_queue'] = [f'{group}:{seed}' for group, seed in pending]
        save()
        with handle:
            finish_stage6(prior_results + shared_queue(pending, adopted=adopted))
        return
    with (HERE / 'controller.lock').open('w') as handle:
        complete = adopt_stage4(args.old_controller, handle)
        audit_stage4()
        if not complete:
            STATE['stage'] = 'stage4_failed'
            save()
            return
        branch = gate()
        jobs = stage6_plan(branch, include_a8=not args.skip_a8)
        STATE.update(stage='stage6', stage6_queue=[f'{group}:{seed}' for group, seed in jobs])
        save()
        outcomes = shared_queue(jobs)
        finish_stage6(outcomes)


if __name__ == '__main__':
    os.environ['TZ'] = 'Asia/Shanghai'
    time.tzset()
    try:
        main()
    except Exception as error:
        if STATE:
            STATE.update(stage='scheduler_failed', attention=repr(error))
            save()
        log('ERROR ' + repr(error))
        raise
    finally:
        STOP_MONITOR.set()
