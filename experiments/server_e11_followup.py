#!/usr/bin/env python3
"""Wait for GPU 1 or 2, run the E11 joint-zero follow-up, archive and publish."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
import os

from server_analysis import REPO, ROOT, HERE, PY, CASES, CASES_FILE, env_for, stamp

OUT = ROOT / '_analysis/E11_followup'
STATE = {}


def save():
    STATE['updated_at'] = stamp()
    STATE['items']['E11_joint']['completed_predictions'] = sum(
        (OUT / 'pred' / f'A6_s{s}' / 'pc_prior_zero' / f'{c}.nii.gz').is_file()
        for s in range(3) for c in CASES)
    temp = HERE / 'analysis_followup_state.tmp'
    temp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2) + '\n')
    temp.replace(HERE / 'analysis_followup_state.json')


def log(message):
    print(f'[{stamp()}] {message}', flush=True)


def free_gpu():
    state = json.loads((HERE / 'analysis_state.json').read_text())
    allowed = set(state.get('active_gpus', [1, 2])) - set(state.get('reserved_gpus', [0]))
    allowed &= {1, 2}
    mapping = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    identities = {int(line.split(',')[0]): line.split(',')[1].strip() for line in mapping.splitlines()}
    busy = {s.strip() for s in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines()}
    return next((g for g in sorted(allowed) if identities[g] not in busy), None)


def run(command, name, gpu=None):
    with (HERE / f'analysis_{name}.log').open('a') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env_for(gpu), stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        if gpu is not None:
            STATE['jobs'][name] = dict(status='running', gpu=gpu, pid=process.pid,
                                       started_at=stamp(), command=command)
        save()
        while process.poll() is None:
            save()
            time.sleep(10)
        if gpu is not None:
            STATE['jobs'][name].update(status='complete' if process.returncode == 0 else 'failed',
                                       ended_at=stamp())
        assert process.returncode == 0, f'{name} failed with exit {process.returncode}'


def archive_and_publish():
    target = REPO / 'results/E11_followup'
    for source in OUT.rglob('*'):
        relative = source.relative_to(OUT)
        if source.is_file() and 'pred' not in relative.parts and source.suffix in ('.csv', '.md', '.json', '.png'):
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    checklist = REPO / 'experiments/服务器操作清单.md'
    content = checklist.read_text()
    content = content.replace('**已排队**：GPU 1 或 2 空闲后执行 63 次补充推理；完成后自动汇总并推送 main',
                              '**已完成**：63 次补充推理及固定路由核查通过，结果见 `results/E11_followup`')
    checklist.write_text(content)
    STATE.update(phase='complete', publish_status='pending')
    save()
    dest = REPO / 'results/server/e11_followup_state.json'
    shutil.copyfile(HERE / 'analysis_followup_state.json', dest)
    paths = ['results/E11_followup', 'results/server/e11_followup_state.json', 'experiments/服务器操作清单.md']
    subprocess.run(['git', 'add', '--', *paths], cwd=REPO, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'Results: E11 joint PC/prior-zero inference and detection metrics', '--', *paths], cwd=REPO, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    # A normal push rejects concurrent incompatible remote changes; never force.
    subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=REPO, check=True)
    STATE.update(publish_status='published', published_commit=commit, published_branch='main')
    save()
    log(f'Published {commit} to main; test cohort remains locked')


def main():
    with (HERE / 'analysis_followup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        STATE.update(pid=os.getpid(), phase='waiting_for_gpu', started_at=stamp(), jobs={},
                     allowed_gpus=[1, 2], reserved_gpus=[0], condition='pc_prior_zero',
                     source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                     items={'E11_joint': {'status': 'pending', 'total_predictions': 63, 'completed_predictions': 0}},
                     test_set='locked; validation cases only')
        save()
        log('Queued joint PC/prior-zero check; waiting for GPU 1 or 2, GPU 0 reserved')
        while (gpu := free_gpu()) is None:
            save()
            time.sleep(15)
        STATE['items']['E11_joint']['status'] = 'running'
        STATE['phase'] = 'running'
        log(f'START E11_joint GPU{gpu}')
        script = str(REPO / 'experiments/analysis/e11_joint_zero.py')
        run([str(PY), '-u', script, 'run', '--device', 'cuda', '--out', str(OUT),
             '--cases-file', str(CASES_FILE)], 'E11_joint', gpu)
        for seed in range(3):
            for case in CASES:
                assert (OUT / 'pred' / f'A6_s{seed}' / 'pc_prior_zero' / f'{case}.nii.gz').is_file()
                loss = json.loads((OUT / 'utility' / f'A6_s{seed}' / f'{case}.json').read_text())['loss']
                assert {'none', 'pc_prior_zero'} <= set(loss)
        audit = json.loads((OUT / 'joint_zero_router_audit.json').read_text())
        assert len(audit['checks']) == 15 and all(c['max_spatial_range'] < 1e-6 for c in audit['checks'].values())
        STATE['items']['E11_joint']['status'] = 'summarizing'
        STATE['phase'] = 'summarizing'
        save()
        run([str(PY), '-u', script, 'summarize', '--workers', '2', '--out', str(OUT),
             '--cases-file', str(CASES_FILE)], 'E11_joint_summary')
        run([str(PY), str(REPO / 'experiments/analysis/e11_detection_metrics.py'),
             '--source', str(OUT), '--out', str(OUT)], 'E11_joint_metrics')
        STATE['items']['E11_joint']['status'] = 'complete'
        archive_and_publish()


if __name__ == '__main__':
    os.environ['TZ'] = 'Asia/Shanghai'
    time.tzset()
    try:
        main()
    except Exception as exc:
        if STATE:
            if STATE['items']['E11_joint']['status'] == 'complete':
                STATE.update(phase='publish_failed', publish_status='failed', attention=repr(exc))
            else:
                STATE.update(phase='failed', attention=repr(exc))
                STATE['items']['E11_joint']['status'] = 'failed'
            save()
        log('FAILED ' + repr(exc))
        raise
