#!/usr/bin/env python3
"""Read-only experiment dashboard. No training imports or scheduling actions."""
import argparse
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
FOLD = 'Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0'
RESFOLD = 'Dataset102_LiTS/nnUNetTrainer_200epochs__nnUNetResEncUNetPlans__3d_fullres/fold_0'
CST = timezone(timedelta(hours=8))
HTML = Path(__file__).with_name('index.html')
RUN_NAME = re.compile(r'(?:A(?:[0-9]|1[0-2])|ResEnc)_s[0-9]+|E16_cost')
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


def run_log(name):
    return PILOT / f'{name.split("_s")[0]}.log' if name in PILOT_RUNS else ROOT / name / 'launcher.log'


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
    total = manifest.get('num_epochs', 200) if not is_cost else None
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


def snapshot():
    with LOCK:
        now = time.time()
        if CACHE['data'] is not None and now - CACHE['time'] < 5:
            return CACHE['data']
        state = read_json(HERE / 'state.json', {})
        names = list(state.get('jobs', {}))
        names += [name_for(e) for e in state.get('stage6_queue', [])]
        pilots, pilot_jobs = pilot_info(now)
        jobs = dict(state.get('jobs', {}))
        jobs.update(pilot_jobs)
        names = list(pilot_jobs) + names
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
        CACHE['data'] = {
            'now': datetime.now(CST).isoformat(), 'updated_at': state.get('updated_at'),
            'state_age_seconds': round(now - mtime) if mtime else None,
            'controller_alive': process_alive(state.get('pid')), 'stage': state.get('stage', 'unknown'),
            'stages': stages, 'runs': list(runs.values()), 'analysis': analysis, 'pilots': pilots,
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
