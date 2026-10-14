#!/usr/bin/env python3
"""Read-only experiment dashboard. No training imports or scheduling actions."""
import argparse
import csv
from collections import deque
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('RESULTS_ROOT', str(REPO.parent / 'nnUNet_data/nnUNet_results_v2')))
HERE = ROOT / '_orchestration'
PILOT = ROOT / '_pilot_lowcnr'
PILOT_RUNS = ('A9_s0', 'A11_s0', 'A12_s0')
REPLICATION = ROOT / '_lowcnr_replication'
REPLICATION_RUNS = tuple(f'{g}_s{s}' for g, seeds in [('A13b', range(3)), ('A13a', range(3)), ('A11', (1, 2))] for s in seeds)
NATIVE_CURVES = ROOT / '_analysis/curves_native'
TEST_OUT = ROOT / '_analysis/test_protocol'
TEST_WORKTREE = REPO.parent / 'Downstream_2_test_protocol'
TEST_RUNS = tuple(f'{g}_s{s}' for g in ('A0', 'A1', 'A11', 'A13a', 'A13b') for s in range(3)) + ('A12_s0',)
TRANSFER = ROOT / '_segresnet_transfer'
REFERENCE = ROOT / '_analysis/reference_baselines'
TRANSFER_RUNS = ('S0_s0', 'S0_s1', 'S13b_s0', 'S13b_s1', 'S13b_bf16_s0', 'S13b_bf16_s1')
# FP16 S13b runs stopped by the user's decision (decoder activations overflowed FP16; rerun in bf16, plan §11)
STOPPED_RUNS = ('S13b_s0', 'S13b_s1')
EXT_WORKTREE = REPO.parent / 'Downstream_2_external'
TRANSFER_RESULTS = EXT_WORKTREE / 'results/segresnet_transfer'
MSD08_OUT = EXT_WORKTREE / 'results/external_msd08'
MSD08_EXT = REPO.parent / 'ext_data/msd08'
MSD08_DATA = MSD08_EXT / 'Dataset104_MSD08'
MSD08_RUNS = tuple(f'{g}_s{s}' for g in ('A0', 'A1', 'A13b') for s in range(3))   # GPU = index % 3
NATIVE_GROUPS = ('A0', 'A1', 'A11', 'A13a', 'A13b')
# A14: Pixel2Cancer comparator (plan §11, 2026-10-14), run from its own worktree
A14_WORKTREE = REPO.parent / 'Downstream_2_p2c'
A14_OUT = A14_WORKTREE / 'results/a14_pixel2cancer'
P2C_SYNTH = REPO.parent / 'nnUNet_data/p2c_synth/Dataset102_fold0'
A14_RUNS = ('A14_s0', 'A14_s1', 'A14_s2')
# TriALS-setting experiment T0 / T13b (plan §11, 2026-10-14), run from its own worktree
TS_OUT = REPO.parent / 'Downstream_2_trials/results/trials_setting'
TS_GROUPS = ('T13b', 'T0')
TS_RUNS = tuple(f'{g}_s{s}' for g in TS_GROUPS for s in range(3))
FOLD = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
RESFOLD = 'Dataset102_LiTS/nnUNetTrainer_200epochs__nnUNetResEncUNetPlans__3d_fullres/fold_0'
CST = timezone(timedelta(hours=8))
HTML = Path(__file__).with_name('index.html')
RUN_NAME = re.compile(r'(?:A(?:[0-9]|1[0-2]|13[ab]|14)|ResEnc)_s[0-9]+|E16_cost')
ANALYSIS_LOGS = {'analysis_scheduler': 'analysis_scheduler.log', 'analysis_selftest': 'analysis_selftest.log',
                 'E11_joint': 'analysis_E11_joint.log', 'E11_joint_summary': 'analysis_E11_joint_summary.log',
                 'analysis_followup': 'analysis_followup.log',
                 'E13': 'analysis_E13.log', 'E11_summary': 'analysis_E11_summary.log', 'E12_summary': 'analysis_E12_summary.log',
                 **{f'{e}_shard{i}': f'analysis_{e}_shard{i}.log' for e in ('E11', 'E12') for i in range(3)}}
LOCK = threading.Lock()
CACHE = {'time': 0, 'data': None, 'gpu_time': 0, 'gpu': [], 'gpu_error': None}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def tail(path, limit=512000):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            data = stream.read().decode('utf-8', errors='replace')
        return data.split('\n', 1)[-1] if size > limit else data
    except OSError:
        return ''


def process_alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rpartition(')')[2].split()[0] not in ('Z', 'T', 't')
    except (ValueError, TypeError, OSError, IndexError):
        return False


def name_for(entry):
    group, seed = entry.split(':')
    return 'E16_cost' if group == 'Cost' else f'{group}_s{seed}'


def transfer_training_log(name):
    """nnU-Net's own training_log of the current attempt. It is written line by
    line, whereas the run.sh stdout log is block-buffered and lags by tens of
    minutes. Logs left by an earlier failed attempt (S0_s1 OOM) are skipped."""
    if not (TRANSFER/f'{name}.log').is_file():
        return None
    logs = sorted((ROOT/name/FOLD).glob('training_log_*.txt'))
    old = TRANSFER/f'{name}_oom_attempt.log'
    if old.is_file():
        logs = [p for p in logs if p.stat().st_mtime > old.stat().st_mtime]
    return logs[-1] if logs else None


def run_log(name):
    if name in A14_RUNS:
        logs = sorted((ROOT/name/FOLD).glob('training_log_*.txt'))
        return logs[-1] if logs else A14_OUT/f'train_{name}.log'
    if name in TS_RUNS:
        logs = sorted((ROOT/name/FOLD).glob('training_log_*.txt'))
        return logs[-1] if logs else TS_OUT/f'train_{name}.log'
    if name in TRANSFER_RUNS:
        return transfer_training_log(name) or TRANSFER/f'{name}.log'
    if name in REPLICATION_RUNS:
        return REPLICATION / f'{name}.log'
    return PILOT / f'{name.split("_s")[0]}.log' if name in PILOT_RUNS else ROOT / name / 'launcher.log'


def replication_info(now):
    state = read_json(REPLICATION / 'state.json', {})
    if not state:
        return None, {}
    state['controller_alive'] = process_alive(state.get('pid'))
    state['age_seconds'] = round(now - (REPLICATION / 'state.json').stat().st_mtime)
    jobs = {}
    for run in REPLICATION_RUNS:
        if run not in state.get('jobs', {}):
            continue
        jobs[run] = dict(state['jobs'][run])
        if state.get('phase') in ('failed', 'cancelled') and jobs[run]['status'] == 'running':
            jobs[run]['status'] = state['phase']
    result = Path(state.get('worktree', str(REPO))) / 'results/lowcnr_replication'
    state['metrics'] = read_json(result / 'summary_pp4mm.json', {})
    state['decisions'] = read_json(result / 'replication_decisions.json', {})
    if state.get('phase') == 'fixed_4mm_metrics':
        state['metric_cases_completed'] = tail(REPLICATION/'post4mm.log').count('fixed 4-mm metrics complete')
        state['metric_cases_total'] = 12 * 21
    for job in state.get('analysis_jobs', {}).values():
        job['process_alive'] = process_alive(job.get('pid'))
    return state, jobs


def native_curves_info(now):
    if not NATIVE_CURVES.exists():
        return None
    cases = set((REPO/'experiments/eval/val_cases.txt').read_text().split())
    counts, runs, dtypes, updated = {}, [], set(), 0
    for group in NATIVE_GROUPS:
        counts[group] = 0
        for seed in range(3):
            run = f'{group}_s{seed}'; done = 0
            for case in cases:
                path = NATIVE_CURVES/'cases'/run/f'{case}.json'
                data = read_json(path)
                if data and data.get('case') == case and data.get('run') == run:
                    done += 1; updated = max(updated, path.stat().st_mtime)
                    dtypes.add(data.get('export_logits_dtype', 'unrecorded'))
            counts[group] += done
            runs.append({'run': run, 'group': group, 'seed': seed, 'completed_cases': done, 'total_cases': len(cases)})
    jobs = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory/'cmdline').read_bytes().decode().split('\0')
            if '--out' not in argv or argv[argv.index('--out')+1] != str(NATIVE_CURVES) or 'run' not in argv:
                continue
            if not any(Path(a).name == 'missed_lesion_probability.py' for a in argv):
                continue
            shard = argv[argv.index('--shard')+1]
            gpu = int(shard.split('/')[0]); pid = int(directory.name)
            if process_alive(pid):
                jobs[f'native_curves_shard{gpu}'] = {'pid': pid, 'gpu': gpu, 'shard': shard, 'status': 'running', 'process_alive': True}
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    completed = sum(counts.values()); total = len(cases)*len(runs)
    parity_path = next(iter(NATIVE_CURVES.rglob('curve_parity.json')), None)
    parity = read_json(parity_path, {}) if parity_path else {}
    expected_runs = {r['run'] for r in runs}
    parity_current = bool(parity_path and parity_path.stat().st_mtime >= updated and
                          set(parity.get('runs', [])) == expected_runs and parity.get('n_cases') == len(cases))
    if not parity_current:
        parity = {}
    summary_file = NATIVE_CURVES/'threshold_sweep_per_run.csv'
    summary_ready = summary_file.is_file() and summary_file.stat().st_mtime >= updated
    phase = ('running' if jobs else 'inference_complete' if completed == total else 'incomplete')
    if completed == total and summary_ready and parity:
        phase = 'complete' if parity.get('passed') else 'parity_failed'
    errors = []
    if not jobs and completed < total:
        for gpu in range(3):
            text = tail(NATIVE_CURVES/f'shard{gpu}.log')
            if 'Traceback (most recent call last)' in text:
                errors.append(f'分片 {gpu} 出现异常，请查看日志')
    return {'phase': phase, 'directory': str(NATIVE_CURVES), 'completed': completed, 'total': total,
            'groups': counts, 'runs': runs, 'jobs': jobs, 'dtypes': sorted(dtypes),
            'last_case_updated_at': datetime.fromtimestamp(updated, CST).isoformat() if updated else None,
            'summary_ready': summary_ready, 'parity': parity, 'errors': errors,
            'scope': '15 runs × 21 validation cases; no test-set inference',
            'parity_rule': 'raw detections identical; after 4 mm detections and FP identical; raw FP difference <=1 per case, explicitly reported'}


def test_protocol_info(now):
    if not (TEST_OUT/'runner.log').is_file():
        return None
    cases = (TEST_WORKTREE/'experiments/eval/test_cases.txt').read_text().split()
    runner = tail(TEST_OUT/'runner.log')
    frozen = (TEST_WORKTREE/'experiments/PROTOCOL_FROZEN').read_text().strip()
    rows = []
    for run in TEST_RUNS:
        fold = ROOT/run/FOLD
        predicted = sum((fold/'test_pred'/f'{case}.nii.gz').is_file() for case in cases)
        summary = read_json(fold/'eval_test/summary.json', {})
        evaluated = summary.get('All', {}).get('n_cases') == len(cases)
        curves = sum(bool(read_json(TEST_OUT/'curves/cases'/run/f'{case}.json')) for case in cases) if run != 'A12_s0' else None
        log = tail(TEST_OUT/f'predict_{run}.log')
        current = re.findall(r'Predicting (LiTS_\d+)', log)
        rows.append(dict(run=run, predicted=predicted, total=len(cases), evaluated=evaluated,
                         curves=curves, current_case=current[-1] if current else None))
    jobs = {}; alive = False
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory/'cmdline').read_bytes().decode().split('\0')
            cwd = (directory/'cwd').resolve()
            if cwd != TEST_WORKTREE or not process_alive(directory.name):
                continue
            if 'experiments/server_test_protocol.sh' in argv:
                alive = True
            if any(Path(a).name in ('nnUNetv2_predict', 'evaluate.py', 'missed_lesion_probability.py') for a in argv):
                env = dict(x.split('=', 1) for x in (directory/'environ').read_bytes().decode().split('\0') if '=' in x)
                gpu = env.get('CUDA_VISIBLE_DEVICES', '')
                if not gpu.isdigit():
                    continue
                result_root = env.get('nnUNet_results', '')
                run = Path(result_root).name
                is_curve = any(Path(a).name == 'missed_lesion_probability.py' for a in argv)
                jobs[f'test_gpu{gpu}'] = dict(gpu=int(gpu), pid=int(directory.name), process_alive=True,
                    name=run if run in TEST_RUNS else f'测试曲线分片 {gpu}',
                    phase='测试阈值曲线' if is_curve else '测试评测' if any(Path(a).name=='evaluate.py' for a in argv) else '测试滑窗推理')
        except (OSError, ValueError, UnicodeError):
            continue
    predicted = sum(r['predicted'] for r in rows); evaluated = sum(r['evaluated'] for r in rows)
    curves = sum(r['curves'] or 0 for r in rows)
    metrics = read_json(TEST_OUT/'tables/metrics_manifest.json', {})
    parity = read_json(TEST_OUT/'curves/parity/curve_parity.json', {})
    stats = read_json(TEST_OUT/'stats/protocol_results.json', {})
    summary_ready = (TEST_OUT/'curves/threshold_sweep_per_run.csv').is_file()
    complete = 'test protocol complete' in runner
    phase = ('已完成' if complete else '冻结统计' if (TEST_OUT/'stats.log').exists() else
             '一致性核验' if (TEST_OUT/'parity.log').exists() else
             '测试阈值曲线' if (TEST_OUT/'curves_summary.log').exists() or (TEST_OUT/'curves_shard0.log').exists() else
             '固定 4 mm 指标' if (TEST_OUT/'metrics.log').exists() else '测试推理与评测')
    errors = []
    for log in TEST_OUT.glob('*.log'):
        text = tail(log, 16000)
        if 'Traceback (most recent call last)' in text:
            errors.append(log.name+' 出现异常，请查看日志')
    if not alive and not complete:
        errors.append('测试执行脚本未运行，尚无完整结束记录')
    return dict(phase=phase, alive=alive, complete=complete, frozen=frozen, runs=rows, jobs=jobs,
                predicted=predicted, total=len(cases)*len(rows), evaluated=evaluated,
                curves=curves, curves_total=len(cases)*15, metrics_ready=bool(metrics),
                summary_ready=summary_ready, parity=parity, stats_ready=bool(stats), errors=errors,
                started_at=re.search(r'\[([^]]+)\] test protocol start', runner).group(1) if 'test protocol start' in runner else None)


def transfer_info(now):
    if not TRANSFER.exists():
        return None
    active = {}; waiting = set(); reference_alive = False; smoke_alive = False
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory/'cmdline').read_bytes().decode().split('\0')
            if not process_alive(directory.name):
                continue
            if 'experiments/run.sh' in argv:
                idx = argv.index('experiments/run.sh'); name = f'{argv[idx+1]}_s{argv[idx+2]}'
                env = dict(x.split('=',1) for x in (directory/'environ').read_bytes().decode().split('\0') if '=' in x)
                if name == 'S0_s99': smoke_alive = True
                if name in TRANSFER_RUNS:
                    active[name] = dict(pid=int(directory.name), gpu=int(env['GPU']), status='running')
            if any(Path(a).name=='reference_baselines_eval.py' for a in argv): reference_alive = True
            if '-c' in argv and 'until grep' in argv[argv.index('-c')+1]:
                for name in TRANSFER_RUNS:
                    if f'/{name}.log' in argv[argv.index('-c')+1]: waiting.add(name)
        except (OSError, ValueError, IndexError, KeyError, UnicodeError):
            continue
    rows=[]
    for name in TRANSFER_RUNS:
        text=tail(TRANSFER/f'{name}.log')
        exits=re.findall(r'RUN_EXIT (\d+)',text)
        status = active.get(name,{}).get('status', 'pending' if name in waiting or not text else 'complete' if exits and exits[-1]=='0' else 'failed' if exits or 'Traceback (most recent call last)' in text else 'unknown')
        row=run_info(name, dict(active.get(name,{}),status=status))
        row['waiting']=name in waiting and name not in active
        row['gpu']=active.get(name,{}).get('gpu', {'S0_s0':2,'S0_s1':2,'S13b_s0':0,'S13b_s1':1,'S13b_bf16_s0':0,'S13b_bf16_s1':1}[name])
        row['phase']='排队等待前序任务' if row['waiting'] else row['phase']
        if name in STOPPED_RUNS and name not in active:
            row.update(status='cancelled', phase='已停止：FP16 激活溢出，改用 bf16 重跑', stopped=True)
        rows.append(row)
    smoke=tail(ROOT/'_smoke_S0.log');smoke_exit=re.findall(r'SMOKE_EXIT (\d+)',smoke)
    smoke_status='running' if smoke_alive else 'complete' if smoke_exit and smoke_exit[-1]=='0' else 'failed' if smoke_exit else 'unknown'
    refs=[];text=tail(REFERENCE/'run.log',2000000)
    for name in ('nnUNet3D','SegResNet','UMambaBot','MedNeXt-L','SwinUNETR','nnUNet2D','SAMed-B','SAMed-H'):
        cases=set(re.findall(re.escape(name)+r' (LiTS_\d+): fixed 4-mm metrics complete',text))
        refs.append(dict(name=name, completed=len(cases), total=27))
    table=[]
    try:
        with (REFERENCE/'reference_table.csv').open() as f: table=list(csv.DictReader(f))
    except OSError: pass
    results={}
    for split in ('val','test'):
        try:
            with (TRANSFER_RESULTS/split/'stats/per_run.csv').open() as f: per_run=list(csv.DictReader(f))
        except OSError: continue
        results[split]=dict(per_run=per_run, stats=read_json(TRANSFER_RESULTS/split/'stats/transfer_stats.json',{}))
    eval_log=tail(TRANSFER_RESULTS/'runner.log')
    return dict(runs=rows,jobs=active,smoke_status=smoke_status,results=results,
                eval_complete='transfer evaluation complete' in eval_log,
                reference=dict(alive=reference_alive, completed=sum(r['completed'] for r in refs),total=216,runs=refs,
                    complete=bool(table) and (REFERENCE/'reference_strata.csv').is_file(), table=table,
                    error='Traceback (most recent call last)' in text),
                historical_oom=(TRANSFER/'S0_s1_oom_attempt.log').is_file())


def pilot_info(now):
    state = read_json(PILOT / 'state.json', {})
    if not state:
        return None, {}
    state['controller_alive'] = process_alive(state.get('pid'))
    state['age_seconds'] = round(now - (PILOT / 'state.json').stat().st_mtime)
    jobs = {}
    for name in PILOT_RUNS:
        original = state.get('jobs', {}).get(name.split('_s')[0])
        if original is None:
            continue
        job = dict(original)
        job['status'] = {'training_and_evaluating': 'running'}.get(job.get('status'), job.get('status', 'pending'))
        if state.get('phase') in ('failed', 'cancelled') and job['status'] == 'running':
            job['status'] = state['phase']
        job.setdefault('started_at', state.get('started_at'))
        jobs[name] = job
    # The controller writes final metrics in its immutable worktree, including
    # the interval before its authorized publication updates the main branch.
    results = Path(state.get('worktree', str(REPO))) / 'results/lowcnr_pilots'
    state['metrics'] = read_json(results / 'summary_pp4mm.json', {})
    state['decisions'] = read_json(results / 'decisions_pp4mm.json', {})
    state['baseline'] = read_json(PILOT / 'preflight/baseline_pp4mm/summary_pp4mm.json', {}).get('A1_s0')
    return state, jobs


def run_info(name, job):
    text = tail(run_log(name))
    current = re.findall(r'\bEpoch (\d+)\b', text)
    durations = [float(v) for v in re.findall(r'Epoch time:\s*([\d.]+)\s*s', text)]
    history = []
    epoch = None
    train_loss = None
    for line in text.splitlines():
        match = re.search(r'\bEpoch (\d+)\b', line)
        if match:
            epoch = int(match[1])
        match = re.search(r'\btrain_loss (-?[\d.]+)', line)
        if match:
            train_loss = float(match[1])
        match = re.search(r'\bval_loss (-?[\d.]+)', line)
        if match and epoch is not None:
            history.append({'epoch': epoch + 1, 'train': train_loss, 'val': float(match[1])})
    completed = max((row['epoch'] for row in history), default=0)
    latest = int(current[-1]) + 1 if current else None
    if latest is None and str(job.get('latest_epoch', '')).isdigit():
        latest = int(job['latest_epoch']) + 1
    is_cost = name == 'E16_cost'
    fold = ROOT / name / (RESFOLD if name.startswith('ResEnc') else FOLD)
    manifest = read_json(fold / 'run_manifest.json', {})
    total = manifest.get('num_epochs', 100 if name in TS_RUNS else 200) if not is_cost else None
    status = job.get('status', 'pending')
    if status == 'complete' and not is_cost:
        completed = total
    phase = {'complete': '已完成', 'failed': '失败', 'pending': '待启动', 'cancelled': '用户取消'}.get(status, '运行中')
    if status == 'running' and not is_cost:
        phase = '验证 / 评测中' if completed >= total else '训练中'
    summary = read_json(fold / 'eval_val/summary.json', {})
    cohort = summary.get('Tumor', {})
    return {
        'name': name, 'gpu': job.get('gpu'), 'status': status, 'phase': phase,
        'current_epoch': latest, 'completed_epochs': completed, 'total_epochs': total,
        'progress': 100 if status == 'complete' else round(min(99, completed / total * 100), 1) if total else 0,
        'epoch_seconds': statistics.median(durations[-10:]) + 1.5 if durations else None,
        'first_epoch_seconds': durations[0] if durations else None,
        'validation_cases_written': job.get('validation_cases_written'),
        'pilot': name in PILOT_RUNS,
        'replication': name in REPLICATION_RUNS,
        'started_at': job.get('started_at'), 'ended_at': job.get('ended_at'),
        'gradient_skips': text.count('non-finite gradient'), 'nonfinite_losses': text.count('non-finite loss'),
        'error': job.get('error'), 'cancellation_reason': job.get('cancellation_reason'), 'history': history[-200:],
        'tumor_dice': cohort.get('tumor_dice_mean'), 'n_cases': summary.get('All', {}).get('n_cases'),
        'lesion_f1': cohort.get('lesion@0.1', {}).get('f1'),
        'process_alive': process_alive(job.get('pid')) if status == 'running' else None,
    }


def stage_cards(state, runs, analysis=None):
    stage = state.get('stage', '')
    # Historical stages are backed by imported records, never inferred from epoch progress.
    evidence = [
        (REPO / 'results/server/stage0.md').is_file(),
        all((REPO / p).is_file() for p in ['experiments/check_grads_gpu.log', 'results/server/copy2_prediction.log']),
        (REPO / 'results/server/smoke_integrity.log').is_file() and (REPO / 'results/server/smoke_summary.csv').is_file(),
        all((REPO / f'step2_mechanism/out/{p}').is_file() for p in ['s44b_pc_cap.json', 's44_routes.json', 's44b.log', 's44.log']),
    ]
    titles = ['环境准备', 'GPU 检查与 Copy2 回归', '六组冒烟测试', 'Step 2 脚本重跑']
    notes = ['83 / 21 / 27 病例划分；权重与依赖核验', '96³ / 50 步梯度检查；Copy2 回归', '2 epoch × 6 组，21 例验证；历史显存峰值未采集', '秩相关与选择性勘误；旧 checkpoint 分析']
    cards = [{'id': i, 'title': titles[i], 'status': 'complete' if evidence[i] else 'unknown',
              'detail': notes[i], 'source': '另一台服务器的已保存证据', 'progress': 100 if evidence[i] else None} for i in range(4)]
    initial = [runs[n] for n in ('A6_s0', 'A0_s0', 'A1_s0') if n in runs]
    ready = len(initial) == 3 and all(r['status'] == 'complete' for r in initial)
    cards.append({'id': 4, 'title': '第一批正式训练', 'status': 'complete' if ready else 'failed' if 'stage4_failed' == stage else 'running' if initial else 'pending',
                  'detail': f"A6 / A0 / A1，seed 0 · {sum(r['status'] == 'complete' for r in initial)} / 3 完成 · 核查{'已完成' if state.get('nonfinite_audit', {}).get('status') == 'complete' else '待完成'}",
                  'source': '当前服务器', 'progress': round(sum(r['progress'] for r in initial) / 3, 1)})
    gate = state.get('gate')
    cards.append({'id': 5, 'title': '决策门', 'status': 'complete' if gate else 'pending', 'progress': 100 if gate else 0,
                  'detail': f"分支 {gate['branch']} · 验证集含肿瘤病例 · 判定规则已冻结" if gate else '等待第一批验证与比较', 'source': '当前服务器'})
    queue = [name_for(e) for e in state.get('stage6_queue', [])]
    done = sum(runs.get(n, {}).get('status') == 'complete' for n in queue)
    train_finished = stage == 'training_complete_analysis_pending'
    cards.append({'id': 6, 'title': '第二批训练与分析', 'status': 'analysis_pending' if train_finished else 'failed' if stage == 'stage6_has_failed_jobs' else 'running' if queue else 'pending',
                  'progress': round(sum(runs.get(n, {}).get('progress', 0) for n in queue) / len(queue), 1) if queue else 0,
                  'detail': f'训练 / 成本队列 {done} / {len(queue)} 完成；E10–E14 分析与外部数据准备待完成', 'source': '当前服务器'})
    if analysis:
        items = analysis.get('items', {})
        count = sum(item.get('status') == 'complete' for item in items.values())
        phase = analysis.get('phase')
        cards[-1]['status'] = 'running' if phase in ('selftest', 'running') else 'failed' if phase == 'analysis_failed' else 'analysis_pending'
        cards[-1]['detail'] = f'训练 / 成本队列 {done} / {len(queue)} 完成；验证分析及补充 {count}/{len(items)} 完成；E10 / E14 / E17 待完成'
    frozen = (REPO / 'experiments/PROTOCOL_FROZEN').is_file()
    test_runs = [name for name, row in runs.items() if name != 'E16_cost' and row['status'] != 'cancelled']
    tested = external = predictions = 0
    if frozen:
        for name in test_runs:
            fold = ROOT / name / (RESFOLD if name.startswith('ResEnc') else FOLD)
            tested += read_json(fold / 'eval_test/summary.json', {}).get('All', {}).get('n_cases') == 27
            external += read_json(fold / 'eval_external/summary.json', {}).get('All', {}).get('n_cases', 0) > 0
            predictions += len(list((fold / 'test_pred').glob('*.nii.gz')))
    detail = (f'协议已冻结；内部测试 {tested} / {len(test_runs)} run 已评测，外部测试 {external} 个 run；协议及完整性核验另计'
              if frozen else '协议尚未冻结；本调度器未启动测试集推理')
    cards.append({'id': 7, 'title': '冻结协议与一次性测试',
                  'status': 'running' if frozen and (predictions or tested or external) else 'pending',
                  'progress': round(tested / len(test_runs) * 100, 1) if test_runs else 0,
                  'detail': detail, 'source': '冻结标记与独立 test_pred / eval_test / eval_external 记录'})
    return cards


def estimates(state, runs):
    # Project FIFO dispatch across currently occupied GPUs; future assignments are estimates only.
    defaults = {}
    for name, row in runs.items():
        if row['epoch_seconds']:
            defaults[name.split('_s')[0]] = row['epoch_seconds']
    lanes = {i: 0 for i in range(3)}
    finishes = {}
    def remaining(row, rate):
        if row['status'] == 'complete':
            return 0
        return max(0, 200 - row['completed_epochs']) * rate + 900
    for name, row in runs.items():
        if row['status'] == 'running' and row['gpu'] in lanes:
            rate = row['epoch_seconds'] or defaults.get(name.split('_s')[0], 135)
            duration = 300 if name == 'E16_cost' else remaining(row, rate)
            lanes[row['gpu']] = duration
            finishes[name] = duration
        elif row['status'] == 'complete':
            finishes[name] = 0
    projected = []
    for entry in state.get('stage6_pending_queue', []):
        name = name_for(entry)
        gpu = min(lanes, key=lanes.get)
        group = entry.split(':')[0]
        rate = defaults.get(group, defaults.get('A6', 135))
        lanes[gpu] += 300 if group == 'Cost' else rate * 200 + 900
        finishes[name] = lanes[gpu]
        projected.append({'name': name, 'estimated_gpu': gpu, 'seconds': round(lanes[gpu])})
    core = [f'{g}_s{s}' for g in ('A0', 'A1', 'A6') for s in range(3)]
    failed = any(r['status'] == 'failed' for r in runs.values())
    return {'core_seconds': round(max(finishes[n] for n in core)) if all(n in finishes for n in core) and not failed else None,
            'queue_seconds': round(max(lanes.values())) if state.get('stage6_queue') and not failed else None,
            'projected': projected, 'basis': '最近 10 轮耗时中位数；每 run 预留 15 分钟验证。仅估算有效训练 / 成本队列，已取消任务不计入。'}


def a14_info(now):
    if not (A14_OUT.exists() or P2C_SYNTH.exists()):
        return None
    runner = tail(A14_OUT/'runner.log', 200000)
    n_variants = len(list(P2C_SYNTH.glob('*.npz'))) if P2C_SYNTH.exists() else 0
    procs, runner_alive, eval_alive = {}, False, False
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory/'cmdline').read_bytes().decode().split('\0')
            if not process_alive(directory.name):
                continue
            joined = ' '.join(argv)
            runner_alive |= 'server_a14.sh' in joined
            eval_alive |= 'a14_eval.py' in joined
            for script, kind in (('experiments/run.sh', 'train'), ('experiments/predict_test.sh', 'predict')):
                idx = next((i for i, a in enumerate(argv) if a.endswith(script)), None)
                if idx is not None and len(argv) > idx + 2 and argv[idx+1] == 'A14':
                    env = dict(x.split('=', 1) for x in (directory/'environ').read_bytes().decode().split('\0') if '=' in x)
                    procs[f'A14_s{argv[idx+2]}'] = (int(env.get('GPU', argv[idx+2])), kind)
        except (OSError, ValueError, UnicodeError):
            continue
    rows, left = [], []
    for name in A14_RUNS:
        fold = ROOT/name/FOLD
        trained = (fold/'checkpoint_final.pth').is_file()
        val_done = (fold/'eval_val/summary.json').is_file()
        test_n = len(list((fold/'test_pred').glob('*.nii.gz'))) if (fold/'test_pred').exists() else 0
        test_done = (fold/'eval_test/summary.json').is_file()
        gpu, kind = procs.get(name, (int(name[-1]), None))
        status = 'complete' if test_done else 'running' if name in procs else 'failed' if re.search(re.escape(name) + r'.*fail', runner) else 'pending'
        row = run_info(name, dict(status=status, gpu=gpu))
        if kind == 'predict' or (trained and val_done and not test_done):
            row['phase'] = f'测试集预测 {test_n} / 27'
        elif trained and not val_done and status == 'running':
            row['phase'] = '验证集推理与评估'
        elif status == 'complete':
            row['phase'] = '训练、验证与测试预测完成'
        row.update(val_done=val_done, test_predicted=test_n, test_done=test_done, trained=trained)
        rows.append(row)
        if status in ('running', 'pending'):
            rate = row['epoch_seconds'] or 70.0
            left.append(max(0, row['total_epochs'] - row['completed_epochs']) * rate
                        + (0 if val_done else 600) + (0 if test_done else 900))
    complete = 'A14_EXIT 0' in runner
    remaining = None if complete else round(max(left, default=0) + 900)
    per_group, comparisons, strata = {}, {}, {}
    for split in ('val', 'test'):
        try:
            with (A14_OUT/f'{split}_per_group.csv').open() as f: per_group[split] = list(csv.DictReader(f))
        except OSError:
            pass
    stats = read_json(A14_OUT/'a14_stats.json', {})
    index = read_json(P2C_SYNTH/'index.json', {})
    steps = [
        dict(name=f'合成变体 {n_variants} / 830', done=n_variants >= 830),
        dict(name='变体索引', done=bool(index)),
        dict(name=f'训练 {sum(r["trained"] for r in rows)} / 3', done=all(r['trained'] for r in rows)),
        dict(name=f'验证集评估 {sum(r["val_done"] for r in rows)} / 3', done=all(r['val_done'] for r in rows)),
        dict(name=f'测试集预测 {sum(r["test_done"] for r in rows)} / 3', done=all(r['test_done'] for r in rows)),
        dict(name='描述性统计', done=bool(stats)),
    ]
    return dict(runs=rows, steps=steps, alive=runner_alive, eval_alive=eval_alive, complete=complete,
                error=bool(re.search(r'failed|Traceback', runner)), n_variants=n_variants,
                remaining_seconds=remaining,
                eta=(datetime.now(CST) + timedelta(seconds=remaining)).strftime('%H:%M') if remaining else None,
                per_group=per_group, stats=stats)


def ts_info(now):
    if not TS_OUT.exists():
        return None
    runner = tail(TS_OUT/'runner.log', 200000)
    procs, runner_alive, eval_alive = {}, False, False
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory/'cmdline').read_bytes().decode().split('\0')
            if not process_alive(directory.name):
                continue
            joined = ' '.join(argv)
            runner_alive |= 'server_trials_setting.sh' in joined
            eval_alive |= 'trials_setting_eval.py' in joined
            for script, kind in (('experiments/run.sh', 'train'), ('experiments/predict_test.sh', 'predict')):
                idx = next((i for i, a in enumerate(argv) if a.endswith(script)), None)
                if idx is not None and len(argv) > idx + 2 and argv[idx+1] in TS_GROUPS:
                    env = dict(x.split('=', 1) for x in (directory/'environ').read_bytes().decode().split('\0') if '=' in x)
                    procs[f'{argv[idx+1]}_s{argv[idx+2]}'] = (int(env.get('GPU', argv[idx+2])), kind)
        except (OSError, ValueError, UnicodeError):
            continue
    rows, left = [], {}
    for name in TS_RUNS:
        fold = ROOT/name/FOLD
        trained = (fold/'checkpoint_final.pth').is_file()
        val_done = (fold/'eval_val/summary.json').is_file()
        best_done = (fold/'eval_test_best/summary.json').is_file()
        test_done = (fold/'eval_test/summary.json').is_file()
        sub = 'test_pred' if best_done else 'test_pred_best'
        test_n = len(list((fold/sub).glob('*.nii.gz'))) if (fold/sub).exists() else 0
        gpu, kind = procs.get(name, (int(name[-1]), None))
        status = ('complete' if test_done else 'running' if name in procs
                  else 'failed' if re.search(re.escape(name) + r' FAILED', runner) else 'pending')
        row = run_info(name, dict(status=status, gpu=gpu))
        if kind == 'predict' or (trained and val_done and not test_done):
            row['phase'] = f"测试集预测（{'final' if best_done else 'best'}）{test_n} / 27"
        elif trained and not val_done and status == 'running':
            row['phase'] = '验证集推理与评估'
        elif status == 'complete':
            row['phase'] = '训练、验证与测试预测完成'
        row.update(val_done=val_done, test_predicted=test_n, best_done=best_done, test_done=test_done, trained=trained)
        rows.append(row)
        if status in ('running', 'pending'):
            rate = row['epoch_seconds'] or 80.0
            left[gpu] = left.get(gpu, 0) + (max(0, row['total_epochs'] - row['completed_epochs']) * rate
                                            + (0 if val_done else 600) + (0 if best_done else 600) + (0 if test_done else 600))
    complete = 'TRIALS_SETTING_EXIT 0' in runner
    remaining = None if complete else round(max(left.values(), default=0) + 900)
    per_group = {}
    for split in ('val', 'test_best', 'test_final'):
        try:
            with (TS_OUT/f'{split}_per_group.csv').open() as f: per_group[split] = list(csv.DictReader(f))
        except OSError:
            pass
    stats = read_json(TS_OUT/'trials_setting_stats.json', {})
    n = len(TS_RUNS)
    steps = [
        dict(name=f'训练 {sum(r["trained"] for r in rows)} / {n}', done=all(r['trained'] for r in rows)),
        dict(name=f'验证集评估 {sum(r["val_done"] for r in rows)} / {n}', done=all(r['val_done'] for r in rows)),
        dict(name=f'测试预测 best {sum(r["best_done"] for r in rows)} / {n}', done=all(r['best_done'] for r in rows)),
        dict(name=f'测试预测 final {sum(r["test_done"] for r in rows)} / {n}', done=all(r['test_done'] for r in rows)),
        dict(name='描述性统计', done=bool(stats)),
    ]
    return dict(runs=rows, steps=steps, alive=runner_alive, eval_alive=eval_alive, complete=complete,
                error=bool(re.search(r'FAILED|failed|Traceback', runner)), remaining_seconds=remaining,
                eta=(datetime.now(CST) + timedelta(seconds=remaining)).strftime('%m-%d %H:%M') if remaining else None,
                per_group=per_group, stats=stats)


def msd08_info(now):
    if not MSD08_OUT.exists():
        return None
    runner = tail(MSD08_OUT/'runner.log', 400000)
    try:
        n_cases = len((MSD08_DATA/'cases.txt').read_text().split())
    except OSError:
        n_cases = 303
    procs, evals, runner_alive = {}, set(), False
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory/'cmdline').read_bytes().decode().split('\0')
            if not process_alive(directory.name):
                continue
            joined = ' '.join(argv)
            if 'server_external_msd08.sh' in joined:
                runner_alive = True
            if any(a.endswith('nnUNetv2_predict') for a in argv[:2]) and 'external_msd08_pred' in joined:
                m = re.search(r'nnUNet_results_v2/([A-Za-z0-9]+_s\d+)/', joined)
                if m and m.group(1) not in procs:
                    env = dict(x.split('=', 1) for x in (directory/'environ').read_bytes().decode().split('\0') if '=' in x)
                    procs[m.group(1)] = int(env.get('CUDA_VISIBLE_DEVICES', '-1').split(',')[0])
            if any(Path(a).name == 'evaluate.py' for a in argv) and 'eval_external_msd08' in joined:
                m = re.search(r'nnUNet_results_v2/([A-Za-z0-9]+_s\d+)/', joined)
                if m: evals.add(m.group(1))
        except (OSError, ValueError, UnicodeError):
            continue
    rows, rates = [], []
    for i, run in enumerate(MSD08_RUNS):
        pred = ROOT/run/FOLD/'external_msd08_pred'
        try:
            times = sorted(p.stat().st_mtime for p in pred.glob('*.nii.gz'))
        except OSError:
            times = []
        n = len(times)
        rate = statistics.median([b - a for a, b in zip(times[-25:-1], times[-24:])]) if n >= 6 else None
        if run in procs and rate:
            rates.append(rate)
        start = re.findall(re.escape(run) + r' prediction start \(GPU (\d)\)', runner)
        evaluated = (ROOT/run/FOLD/'eval_external_msd08/summary.json').is_file()
        status = ('complete' if evaluated else 'evaluating' if run in evals else 'predicting' if run in procs
                  else 'predicted' if n >= n_cases else 'pending')
        rows.append(dict(name=run, group=run.rsplit('_s', 1)[0], gpu=procs.get(run, int(start[-1]) if start else i % 3),
                         predicted=n, total=n_cases, seconds_per_case=round(rate, 1) if rate else None,
                         status=status, evaluated=evaluated, last_case_age=round(now - times[-1]) if times else None))
    default_rate = statistics.median(rates) if rates else 11.0
    lanes = {}
    for i, r in enumerate(rows):
        g = i % 3
        lanes[g] = lanes.get(g, 0) + max(0, r['total'] - r['predicted']) * (r['seconds_per_case'] or default_rate)
    predict_left = max(lanes.values()) if lanes else 0
    ref_n = len(list((MSD08_DATA/'labelsTs').glob('*.nii.gz'))) if (MSD08_DATA/'labelsTs').exists() else 0
    n_eval = sum(r['evaluated'] for r in rows)
    stats = read_json(MSD08_OUT/'stats/protocol_results.json', {})
    secondary = []
    try:
        with (MSD08_OUT/'stats/secondary_per_group.csv').open() as f: secondary = list(csv.DictReader(f))
    except OSError:
        pass
    steps = [
        dict(name='整理 303 例影像', done=f'{n_cases} cases staged' in runner),
        dict(name='9 个模型推理', done=all(r['predicted'] >= n_cases for r in rows)),
        dict(name='肝脏参照标签（基线多数投票）', done='reference labels built' in runner),
        dict(name=f'评估 {n_eval} / 9', done=n_eval == 9),
        dict(name='冻结协议统计', done=bool(stats)),
    ]
    remaining = None
    if not stats:
        left_eval = (9 - n_eval) / 3 * 20 * 60
        remaining = round(predict_left + (0 if steps[2]['done'] else 300) + left_eval + 20 * 60)
    overlap = read_json(MSD08_EXT/'overlap_check/overlap.json', {})
    inspect = read_json(MSD08_EXT/'inspect/summary.json', {})
    return dict(runs=rows, steps=steps, alive=runner_alive, complete='EXTERNAL_EXIT 0' in runner,
                error=bool(re.search(r'incomplete|missing|Traceback', runner)), n_cases=n_cases, reference_labels=ref_n,
                remaining_seconds=remaining, eta=(datetime.now(CST) + timedelta(seconds=remaining)).strftime('%H:%M') if remaining else None,
                stats=stats, secondary=secondary,
                overlap=dict(independent=overlap.get('independent'), min_fingerprint=overlap.get('min_fingerprint_mae_hu'),
                             n_lits=overlap.get('n_lits')),
                annotation=dict(components=inspect.get('n_tumour_components'),
                                single=(inspect.get('tumour_components_per_case') or {}).get('1')))


def snapshot():
    with LOCK:
        now = time.time()
        if CACHE['data'] is not None and now - CACHE['time'] < 5:
            return CACHE['data']
        state = read_json(HERE / 'state.json', {})
        names = list(state.get('jobs', {}))
        names += [name_for(e) for e in state.get('stage6_queue', [])]
        pilots, pilot_jobs = pilot_info(now)
        replication, replication_jobs = replication_info(now)
        native_curves = native_curves_info(now)
        test_protocol = test_protocol_info(now)
        transfer = transfer_info(now)
        msd08 = msd08_info(now)
        a14 = a14_info(now)
        ts = ts_info(now)
        a13_curves = read_json(REPLICATION/'probability_curves_A13/state.json', {})
        if a13_curves:
            a13_curves['controller_alive'] = process_alive(a13_curves.get('pid'))
            a13_curves['age_seconds'] = round(now-(REPLICATION/'probability_curves_A13/state.json').stat().st_mtime)
            for job in a13_curves.get('jobs', {}).values():
                job['process_alive'] = process_alive(job.get('pid'))
        jobs = dict(state.get('jobs', {}))
        jobs.update(pilot_jobs)
        jobs.update(replication_jobs)
        names = list(replication_jobs) + list(pilot_jobs) + names
        runs = {name: run_info(name, jobs.get(name, {})) for name in dict.fromkeys(names) if RUN_NAME.fullmatch(name)}
        if now - CACHE['gpu_time'] >= 15:
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], text=True, timeout=4)
                CACHE['gpu'] = []
                for line in output.splitlines():
                    i, model, util, used, total, temp = [v.strip() for v in line.split(',')]
                    CACHE['gpu'].append({'gpu': int(i), 'model': model, 'util': int(util), 'used_mib': int(used), 'total_mib': int(total), 'temperature': int(temp)})
                CACHE['gpu_error'] = None
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                CACHE['gpu_error'] = type(exc).__name__
            CACHE['gpu_time'] = now
        try:
            mtime = (HERE / 'state.json').stat().st_mtime
        except OSError:
            mtime = 0
        analysis = read_json(HERE / 'analysis_state.json', {})
        if analysis:
            analysis['controller_alive'] = process_alive(analysis.get('pid'))
            analysis['age_seconds'] = round(now - (HERE / 'analysis_state.json').stat().st_mtime)
            followup = read_json(HERE / 'analysis_followup_state.json', {})
            if followup:
                analysis.setdefault('items', {}).update(followup.get('items', {}))
                analysis.setdefault('jobs', {}).update(followup.get('jobs', {}))
                analysis['followup'] = followup
                if followup.get('phase') in ('waiting_for_gpu', 'running', 'summarizing'):
                    analysis['phase'] = 'running'
                    alive = process_alive(followup.get('pid'))
                    analysis['controller_alive'] |= alive
                    analysis['age_seconds'] = min(analysis['age_seconds'], round(now - (HERE / 'analysis_followup_state.json').stat().st_mtime))
                    if not alive:
                        analysis['attention'] = 'E11 补充核验调度进程未运行'
                elif followup.get('phase') in ('failed', 'publish_failed'):
                    analysis['attention'] = followup.get('attention', 'E11 补充核验失败')
            for job in analysis.get('jobs', {}).values():
                job['process_alive'] = process_alive(job.get('pid')) if job.get('status') == 'running' else None
        stages = stage_cards(state, runs, analysis)
        eta = estimates(state, runs)
        if pilots:
            pilot_rows = [runs[n] for n in PILOT_RUNS if n in runs]
            phase = pilots.get('phase')
            stages.append({'id': 'P', 'title': '低 CNR 三组 pilot',
                           'status': 'complete' if phase == 'complete' else phase if phase in ('failed', 'cancelled') else 'running',
                           'detail': f"A9 / A11 / A12 · {sum(r['status'] == 'complete' for r in pilot_rows)} / 3 完成；固定 4 mm 判据；全部报告",
                           'source': '独立 pilot 控制器、训练日志与全卷验证',
                           'progress': round(sum(r['progress'] for r in pilot_rows) / 3, 1)})
            eta['pilot_seconds'] = round(max((max(0, r['total_epochs'] - r['completed_epochs']) * r['epoch_seconds'] + 900
                                              for r in pilot_rows if r['status'] == 'running' and r['epoch_seconds']), default=0))
        if replication:
            phase = replication.get('phase')
            rr = [runs[n] for n in REPLICATION_RUNS if n in runs]
            stages.append({'id': 'R', 'title': 'A13 / A11 多 seed 复现',
                           'status': 'complete' if phase == 'complete' else phase if phase in ('failed', 'cancelled') else 'running',
                           'detail': f"{sum(r['status'] == 'complete' for r in rr)} / 8 训练与验证完成；固定 4 mm 核验及阈值曲线另计",
                           'source': '独立三卡队列；§6c 固定顺序与复现条件',
                           'progress': round(sum(r['progress'] for r in rr)/8,1)})
            lanes = []
            active_rates = [r['epoch_seconds'] for r in rr if r['status'] == 'running' and r['epoch_seconds']]
            for queue in replication.get('queues', {}).values():
                duration = 0
                for n in queue:
                    row = runs.get(n)
                    if row and row['status'] in ('pending', 'running'):
                        default = 95.29*1.4 if n.startswith('A13b_') else 95.29
                        rate = row['epoch_seconds'] or (statistics.median(active_rates) if n.startswith('A13b_') and active_rates else default)
                        duration += max(0,200-row['completed_epochs'])*rate+900
                lanes.append(duration)
            eta['replication_seconds'] = round(max(lanes, default=0))
        if native_curves:
            stages.append({'id': 'N', 'title': '原 dtype 验证集阈值曲线',
                           'status': 'complete' if native_curves['phase']=='complete' else 'failed' if native_curves['phase']=='parity_failed' else 'running' if native_curves['jobs'] else 'analysis_pending',
                           'detail': f"{native_curves['completed']} / {native_curves['total']} 病例-运行结果；原 dtype 导出，固定 4 mm 后核验",
                           'source': '当前三卡进程、逐例 JSON、curve_parity.json',
                           'progress': round(native_curves['completed']/native_curves['total']*100,1) if native_curves['total'] else 0})
            for stage in stages:
                if stage['id']=='R' and replication.get('phase')=='failed':
                    stage.update(status='complete' if native_curves['phase']=='complete' else 'analysis_pending',
                                 detail='8 / 8 训练与验证已完成，4 mm 判定已生成；旧曲线核验失败记录保留；'+('新曲线核验已通过' if native_curves['phase']=='complete' else '新曲线重新导出 / 待核验'))
        if test_protocol:
            for stage in stages:
                if stage['id'] == 7:
                    stage.update(status='complete' if test_protocol['complete'] else 'running' if test_protocol['alive'] else 'failed',
                        detail=f"{test_protocol['phase']}；{test_protocol['predicted']} / {test_protocol['total']} 分割，{test_protocol['evaluated']} / 16 评测；{test_protocol['curves']} / {test_protocol['curves_total']} 曲线",
                        source='冻结工作树、test_protocol 日志与逐例产物',
                        progress=100 if test_protocol['complete'] else round(test_protocol['predicted']/test_protocol['total']*100,1))
        if transfer:
            live=[r for r in transfer['runs'] if not r.get('stopped')]
            stages.append(dict(id='S',title='SegResNet 架构迁移',
                status='complete' if transfer['eval_complete'] else 'running' if transfer['jobs'] else 'analysis_pending',
                detail=(f"{sum(r['status']=='complete' for r in live)} / {len(live)} 训练完成；"
                        + ('验证集与测试集评估完成：方法在 SegResNet 上无提升（负结果）' if transfer['eval_complete'] else '测试集评估进行中')
                        + '；FP16 的 S13b 两次运行已停止'),
                source='训练日志、results/segresnet_transfer',progress=100 if transfer['eval_complete'] else round(sum(r['progress'] for r in live)/len(live),1)))
            ref=transfer['reference']
            stages.append(dict(id='B',title='八个参考基线统一评测',status='complete' if ref['complete'] else 'running' if ref['alive'] else 'analysis_pending',
                detail=f"{ref['completed']} / {ref['total']} 固定 4 mm 病例评测完成；汇总另计",source='当前 run.log 与汇总 CSV',progress=round(ref['completed']/ref['total']*100,1)))
        if msd08:
            pred = sum(r['predicted'] for r in msd08['runs']); tot = sum(r['total'] for r in msd08['runs'])
            stages.append(dict(id='X', title='MSD Task08 外部验证',
                status='complete' if msd08['complete'] else 'failed' if msd08['error'] else 'running' if msd08['alive'] else 'analysis_pending',
                detail=f"{pred} / {tot} 例推理；{sum(r['evaluated'] for r in msd08['runs'])} / 9 评估；独立性核查通过；A0 / A1 / A13b × 3 seeds",
                source='server_external_msd08.sh、预测目录与评估结果', progress=100 if msd08['complete'] else round(pred / tot * 95, 1)))
        if a14:
            tr = [r for r in a14['runs']]
            stages.append(dict(id='C', title='Pixel2Cancer 对比（A14）',
                status='complete' if a14['complete'] else 'failed' if a14['error'] else 'running' if a14['alive'] else 'analysis_pending',
                detail=(f"{sum(r['trained'] for r in tr)} / 3 训练完成，{sum(r['test_done'] for r in tr)} / 3 测试预测完成；"
                        f"830 个离线合成变体，p = 0.9；其余设置同 A0；描述性对比"),
                source='server_a14.sh、训练日志与 results/a14_pixel2cancer',
                progress=100 if a14['complete'] else round(sum(r['progress'] for r in tr) / 3 * 0.9 + 10 * (a14['n_variants'] >= 830), 1)))
        if ts:
            tr = ts['runs']
            stages.append(dict(id='T', title='TriALS 训练设置下的比较（T0 / T13b）',
                status='complete' if ts['complete'] else 'failed' if ts['error'] else 'running' if ts['alive'] else 'analysis_pending',
                detail=(f"{sum(r['trained'] for r in tr)} / 6 训练完成，{sum(r['test_done'] for r in tr)} / 6 测试预测完成；"
                        f"从头训练、SGD lr 1e-2、100 epoch（同 TriALS）；描述性对比"),
                source='server_trials_setting.sh、训练日志与 results/trials_setting',
                progress=100 if ts['complete'] else round(sum(r['progress'] for r in tr) / len(tr) * 0.9
                                                          + sum(r['test_done'] for r in tr) / len(tr) * 10, 1)))
        CACHE['data'] = {
            'now': datetime.now(CST).isoformat(), 'updated_at': state.get('updated_at'),
            'state_age_seconds': round(now - mtime) if mtime else None,
            'controller_alive': process_alive(state.get('pid')), 'stage': state.get('stage', 'unknown'),
            'stages': stages, 'runs': list(runs.values()) + (transfer['runs'] if transfer else []) + (a14['runs'] if a14 else []) + (ts['runs'] if ts else []), 'analysis': analysis, 'pilots': pilots, 'replication': replication,
            'a13_curves': a13_curves,
            'native_curves': native_curves, 'test_protocol': test_protocol, 'transfer': transfer, 'msd08': msd08, 'a14': a14, 'ts': ts,
            'gpus': CACHE['gpu'], 'gpu_error': CACHE['gpu_error'],
            'gpu_age_seconds': round(now - CACHE['gpu_time']),
            'pending': [name_for(e) for e in state.get('stage6_pending_queue', [])],
            'exclusions': state.get('stage6_exclusions', {}),
            'gate': state.get('gate'), 'attention': state.get('attention'),
            'audit_complete': state.get('nonfinite_audit', {}).get('status') == 'complete',
            'eta': eta,
        }
        CACHE['time'] = now
        return CACHE['data']


class Handler(BaseHTTPRequestHandler):
    def respond(self, body, content_type, code=200):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'same-origin')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        try:
            if url.path == '/':
                self.respond(HTML.read_bytes(), 'text/html; charset=utf-8')
            elif url.path == '/api/status':
                self.respond(json.dumps(snapshot(), ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')
            elif url.path == '/api/log':
                name = parse_qs(url.query).get('run', ['scheduler'])[0]
                if name == 'scheduler':
                    path = HERE / 'scheduler.log'
                elif name == 'pilot_controller':
                    path = PILOT / 'controller.log'
                elif name == 'replication_controller':
                    path = REPLICATION / 'controller.log'
                elif name == 'replication_metrics':
                    path = REPLICATION / 'post4mm.log'
                elif name in ('curves_shard0', 'curves_shard1', 'curves_shard2'):
                    path = REPLICATION / f'{name}.log'
                elif name == 'a13_curve_controller':
                    path = REPLICATION/'probability_curves_A13/controller.log'
                elif name in ('curves_A13_shard0', 'curves_A13_shard1', 'curves_A13_shard2'):
                    path = REPLICATION/'probability_curves_A13'/f'{name}.log'
                elif name in ('native_curves_shard0', 'native_curves_shard1', 'native_curves_shard2'):
                    path = NATIVE_CURVES/f'shard{name[-1]}.log'
                elif name.startswith('transfer_') and name[9:] in TRANSFER_RUNS:
                    # Live training_log while running; the full stdout (validation and
                    # evaluation included) once run.sh has exited and flushed it.
                    out=TRANSFER/f'{name[9:]}.log'
                    path=out if 'RUN_EXIT' in tail(out, 4096) else run_log(name[9:])
                elif name == 'transfer_eval':
                    path=TRANSFER_RESULTS/'runner.log'
                elif name == 'ts_runner':
                    path=TS_OUT/'runner.log'
                elif name in ('ts_metrics_val', 'ts_metrics_test_best', 'ts_metrics_test_final', 'ts_stats'):
                    path=TS_OUT/(f'metrics_{name[11:]}.log' if 'metrics' in name else 'stats.log')
                elif name.startswith('ts_predict_') and name[11:].rsplit('_', 1)[0] in TS_RUNS:
                    path=TS_OUT/f'predict_{name[11:]}.log'
                elif name.startswith('ts_train_') and name[9:] in TS_RUNS:
                    path=TS_OUT/f'train_{name[9:]}.log'
                elif name == 'a14_runner':
                    path=A14_OUT/'runner.log'
                elif name in ('a14_metrics_val', 'a14_metrics_test', 'a14_stats'):
                    path=A14_OUT/(f'metrics_{name[12:]}.log' if 'metrics' in name else 'stats.log')
                elif name.startswith('a14_predict_') and name[12:] in A14_RUNS:
                    path=A14_OUT/f'predict_{name[12:]}.log'
                elif name.startswith('a14_train_') and name[10:] in A14_RUNS:
                    path=A14_OUT/f'train_{name[10:]}.log'
                elif name == 'msd08_runner':
                    path=MSD08_OUT/'runner.log'
                elif name in ('msd08_reference', 'msd08_metrics', 'msd08_stats'):
                    path=MSD08_OUT/f'{name[6:]}.log'
                elif name.startswith('msd08_predict_') and name[14:] in MSD08_RUNS:
                    path=MSD08_OUT/f'predict_{name[14:]}.log'
                elif name.startswith('msd08_eval_') and name[11:] in MSD08_RUNS:
                    path=MSD08_OUT/f'eval_{name[11:]}.log'
                elif name == 'transfer_smoke':
                    path=ROOT/'_smoke_S0.log'
                elif name == 'reference_baselines':
                    path=REFERENCE/'run.log'
                elif name == 'test_runner':
                    path = TEST_OUT/'runner.log'
                elif name in ('test_metrics', 'test_parity', 'test_stats', 'test_curves_summary'):
                    path = TEST_OUT/f'{name[5:]}.log'
                elif name in ('test_curves_shard0', 'test_curves_shard1', 'test_curves_shard2'):
                    path = TEST_OUT/f'curves_shard{name[-1]}.log'
                elif name.startswith('test_predict_') and name[13:] in TEST_RUNS:
                    path = TEST_OUT/f'predict_{name[13:]}.log'
                elif name == 'audit':
                    path = HERE / 'stage4_nonfinite_audit.log'
                elif name in ANALYSIS_LOGS:
                    path = HERE / ANALYSIS_LOGS[name]
                elif RUN_NAME.fullmatch(name) and name in {r['name'] for r in snapshot()['runs']}:
                    path = run_log(name)
                else:
                    self.respond(b'Unknown log', 'text/plain', 404)
                    return
                self.respond(json.dumps({'text': '\n'.join(deque(tail(path).splitlines(), maxlen=100))}, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
            elif url.path == '/healthz':
                self.respond(b'ok\n', 'text/plain')
            else:
                self.respond(b'Not found', 'text/plain', 404)
        except (OSError, ValueError, KeyError) as exc:
            self.respond(json.dumps({'error': type(exc).__name__}).encode(), 'application/json', 503)

    def log_message(self, format, *args):
        if args and args[1] not in ('200', '304'):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=6006, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Read-only dashboard listening on {args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
