#!/usr/bin/env python3
"""Run the eight authorized section-6c training jobs, validate, archive and push.

An isolated committed worktree is required. No test inference is scheduled.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import server_lowcnr_pilots as S

QUEUES = {0: [('A13b', 0), ('A13a', 0), ('A11', 1)],
          1: [('A13b', 1), ('A13a', 1), ('A11', 2)],
          2: [('A13b', 2), ('A13a', 2)]}
S.OUT = S.ROOT/'_lowcnr_replication'


def name(group, seed):
    return f'{group}_s{seed}'


def env_for(gpu):
    env = os.environ.copy()
    for k in list(env):
        if k.startswith('PHASER_') or k == 'EPOCHS':
            del env[k]
    env.update(GPU=str(gpu), RESULTS_ROOT=str(S.ROOT), ENV_BIN=str(S.PY.parent),
               VOCOMNI_NNUNET_CKPT=S.STATE['pretrained_path'], nnUNet_n_proc_DA='6',
               nnUNet_def_n_proc='2', EVAL_WORKERS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', TZ='Asia/Shanghai')
    return env


def spawn(args, gpu=None, logfile=None):
    S.check_stop()
    stream = logfile.open('w')
    proc = subprocess.Popen(args, cwd=S.REPO, env=env_for(gpu) if gpu is not None else env_for(0),
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    stream.close(); S.PROCESSES.append(proc)
    return proc


def wait_command(args, logfile):
    proc = spawn(args, logfile=logfile)
    while proc.poll() is None:
        S.check_stop(); S.save(); time.sleep(10)
    if proc.returncode:
        raise RuntimeError(f'Command failed ({proc.returncode}); see {logfile}')


def progress(run):
    job = S.STATE['jobs'][run]
    text = (S.OUT/f'{run}.log').read_text(errors='replace')
    epochs = re.findall(r'\bEpoch (\d+)\b', text)
    times = [float(x) for x in re.findall(r'Epoch time: ([0-9.]+) s', text)]
    job.update(latest_epoch=int(epochs[-1]) if epochs else None, completed_epoch_times=len(times))
    if times:
        if 'first_epoch_seconds' not in job:
            job['first_epoch_seconds'] = times[0]
            job['first_epoch_over_twice_A11_reference'] = times[0] > 2*95.29 if run.startswith('A13b_') else False
            S.log(f'{run}: first epoch {times[0]:.2f}s; peak GPU memory {job.get("peak_gpu_memory_mib", "unknown")} MiB')
        job['mean_recent_epoch_seconds'] = sum(times[-10:])/len(times[-10:])
        job['last_epoch_seconds'] = times[-1]
    job['validation_cases_written'] = len(list((S.ROOT/run/S.TRAINER/'validation').glob('*.nii.gz')))


def sample_gpu_memory(active):
    try:
        output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
        for line in output.splitlines():
            gpu, memory = (int(v.strip()) for v in line.split(','))
            if gpu in active:
                run, _ = active[gpu]; job = S.STATE['jobs'][run]
                job['peak_gpu_memory_mib'] = max(memory, job.get('peak_gpu_memory_mib',0))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass


def launch_next(gpu, pending, active):
    if not pending[gpu]:
        return
    group, seed = pending[gpu].pop(0); run = name(group, seed)
    proc = spawn(['bash',str(S.REPO/'experiments/run.sh'),group,str(seed)],gpu,S.OUT/f'{run}.log')
    active[gpu] = (run,proc)
    S.STATE['jobs'][run].update(pid=proc.pid,status='running',started_at=S.stamp())
    S.log(f'Started {run} on GPU {gpu}; PID {proc.pid}')


def archive(run):
    folder = S.ROOT/run/S.TRAINER
    manifest = json.loads((folder/'run_manifest.json').read_text())
    assert manifest['epochs_completed'] == manifest['num_epochs'] == 200
    assert manifest['fold'] == 0 and manifest['seed'] == int(run.rsplit('_s',1)[1])
    assert manifest['num_iterations_per_epoch'] == 250
    assert manifest['env']['VOCOMNI_NNUNET_CKPT'] == S.STATE['pretrained_path']
    cases = (S.REPO/'experiments/eval/val_cases.txt').read_text().split()
    assert len(cases) == 21
    assert all((folder/'validation'/f'{c}.nii.gz').is_file() for c in cases)
    assert (folder/'checkpoint_final.pth').is_file()
    for f in ('cases.csv','lesions.csv','pred_lesions.csv','summary.json'):
        assert (folder/'eval_val'/f).is_file()
    target = S.REPO/f'results/{run}'
    shutil.copytree(folder/'eval_val',target/'eval_val',dirs_exist_ok=True)
    for f in ('run_manifest.json','progress.png'):
        shutil.copyfile(folder/f,target/f)
    shutil.copyfile(S.OUT/f'{run}.log',target/'replication_run.log')


def curves(target):
    S.STATE['phase']='threshold_curves'; S.STATE['analysis_jobs']={}; S.save()
    out = S.OUT/'probability_curves'; processes=[]
    # Reuse verified A0 probabilities from the earlier diagnostic; only A1/A11
    # require new inference. No full-resolution probability volumes are saved.
    shutil.copytree(S.REPO/'results/missed_probability/cases',out/'cases',dirs_exist_ok=True)
    for gpu in range(3):
        logfile = S.OUT/f'curves_shard{gpu}.log'
        env = env_for(gpu); env['CUDA_VISIBLE_DEVICES']=str(gpu)
        stream = logfile.open('w')
        proc = subprocess.Popen([str(S.PY),'-u',str(S.REPO/'experiments/analysis/missed_lesion_probability.py'),
                                 'run','--runs','A1:0,1,2','A11:0,1,2','--shard',f'{gpu}/3',
                                 '--device','cuda','--out',str(out)],cwd=S.REPO,env=env,
                                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        stream.close(); S.PROCESSES.append(proc); processes.append(proc)
        S.STATE['analysis_jobs'][f'curves_shard{gpu}']=dict(pid=proc.pid,gpu=gpu,status='running')
    while any(p.poll() is None for p in processes):
        S.check_stop()
        for gpu, proc in enumerate(processes):
            if proc.poll() is not None:
                S.STATE['analysis_jobs'][f'curves_shard{gpu}'].update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
        S.STATE['probability_cases_written'] = len(list((out/'cases').glob('*/*.json')))
        S.save(); time.sleep(15)
    if any(p.returncode for p in processes):
        raise RuntimeError('Threshold-curve inference failed; outputs retained')
    for group in ('A0','A1','A11'):
        for seed in range(3):
            files=list((out/'cases'/name(group,seed)).glob('*.json'))
            assert len(files)==21, (group,seed,len(files))
    wait_command([str(S.PY),str(S.REPO/'experiments/analysis/missed_lesion_probability.py'),
                  'summarize','--out',str(out),'--results-dir',str(S.REPO/'results')],S.OUT/'curves_summary.log')
    import pandas as pd
    check=pd.read_csv(out/'check_argmax_vs_eval_val.csv')
    assert len(check)==9 and (check.lesions_compared==92).all() and (check.cases_compared==21).all()
    assert (check.lesion_detection_agree==1).all() and (check.fp_count_abs_diff==0).all()
    shutil.copytree(out,target/'probability_curves',dirs_exist_ok=True)
    for path in S.OUT.glob('curves*.log'):
        shutil.copyfile(path,target/path.name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-root',default='/root/autodl-tmp/Downstream_2')
    p.add_argument('--publish',action='store_true'); args=p.parse_args()
    S.CONTROL=Path(args.control_root); S.OUT.mkdir(parents=True,exist_ok=True)
    lock=(S.OUT/'controller.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (S.OUT/'state.json').exists() or (S.OUT/'STOP').exists():
        raise RuntimeError('Existing state/STOP; refuse an implicit second launch')
    S.START=time.time(); pretrained=S.CONTROL/'VoComni_nnunet.pt'; assert pretrained.is_file()
    jobs={name(g,s):dict(group=g,seed=s,gpu=gpu,status='pending') for gpu,q in QUEUES.items() for g,s in q}
    for run in jobs:
        folder=S.ROOT/run/S.TRAINER
        assert not (folder/'checkpoint_latest.pth').exists() and not (folder/'checkpoint_final.pth').exists(),run
    S.STATE.update(phase='starting',pid=os.getpid(),started_at=S.stamp(),jobs=jobs,
                   queues={str(gpu):[name(g,s) for g,s in q] for gpu,q in QUEUES.items()},
                   worktree=str(S.REPO),control_root=str(S.CONTROL),
                   source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.REPO,text=True).strip(),
                   pretrained_path=str(pretrained),pretrained_sha256=hashlib.sha256(pretrained.read_bytes()).hexdigest(),
                   publish_status='pending' if args.publish else 'disabled',
                   scope='Section 6c: eight fresh 200-epoch fold0 runs; fixed 4-mm metrics and validation threshold curves only; test set locked')
    S.save()
    try:
        pending={gpu:list(q) for gpu,q in QUEUES.items()};active={}
        for gpu in pending: launch_next(gpu,pending,active)
        S.STATE['phase']='running';S.save()
        while active:
            S.check_stop();sample_gpu_memory(active)
            for gpu,(run,proc) in list(active.items()):
                progress(run)
                if proc.poll() is not None:
                    S.STATE['jobs'][run].update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode,ended_at=S.stamp())
                    S.log(f'{run} exited: {proc.returncode}');del active[gpu]
                    if proc.returncode==0:
                        launch_next(gpu,pending,active)
            S.save();time.sleep(15)
        if not all(j['status']=='complete' for j in S.STATE['jobs'].values()):
            raise RuntimeError('Training failure; unstarted jobs remain pending, no incomplete result is published as complete')
        # Keep the source worktree clean throughout every training run so all
        # manifests refer to the same committed source without a dirty suffix.
        S.STATE['phase']='archiving';S.save()
        for run in S.STATE['jobs']:
            archive(run)
        target=S.REPO/'results/lowcnr_replication'; target.mkdir(parents=True,exist_ok=True)
        shutil.copytree(S.OUT/'preflight',target/'preflight',dirs_exist_ok=True)
        S.STATE['phase']='fixed_4mm_metrics';S.save()
        wait_command([str(S.PY),'-u',str(S.REPO/'experiments/analysis/lowcnr_replication_metrics.py'),
                      '--out',str(target)],S.OUT/'post4mm.log')
        shutil.copyfile(S.OUT/'post4mm.log',target/'post4mm.log')
        curves(target)
        S.STATE['phase']='complete';S.save()
        shutil.copyfile(S.OUT/'state.json',target/'state.json')
        shutil.copyfile(S.OUT/'controller.log',target/'controller.log')
        if args.publish:
            paths=[f'results/{run}' for run in S.STATE['jobs']]+['results/lowcnr_replication']
            wait_command(['git','add','--',*paths],S.OUT/'git_add.log')
            wait_command(['git','commit','--only','-m','Results: archive all section-6c seed replication and validation curves','--',*paths],S.OUT/'git_commit.log')
            wait_command(['git','fetch','origin'],S.OUT/'git_fetch.log')
            wait_command(['git','merge','--no-edit','origin/main'],S.OUT/'git_merge.log')
            wait_command(['git','push','origin','HEAD:main'],S.OUT/'git_push.log')
            S.STATE.update(publish_status='published',published_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.REPO,text=True).strip());S.save()
            S.log(f"Published all replication results to main: {S.STATE['published_commit']}")
    except KeyboardInterrupt as e:
        S.terminate_jobs();S.STATE.update(phase='cancelled',error=str(e));S.save();S.log(str(e))
    except Exception as e:
        S.terminate_jobs();S.STATE.update(phase='failed',error=str(e));S.save();S.log(f'Failed: {e}');raise


if __name__=='__main__':
    import signal
    def terminated(*_):raise KeyboardInterrupt('Replication controller terminated')
    signal.signal(signal.SIGTERM,terminated)
    main()
