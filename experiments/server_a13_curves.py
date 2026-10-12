#!/usr/bin/env python3
"""Export missing A13 validation curves, verify all requested artifacts, publish.

Each GPU shard starts after that GPU's existing A1/A11 curve shard exits.
No training, protocol freezing or test-set inference is performed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import server_lowcnr_pilots as S

BASE = S.ROOT/'_lowcnr_replication'
S.OUT = BASE/'probability_curves_A13'
BASE_STATE = BASE/'state.json'


def read_base():
    return json.loads(BASE_STATE.read_text())


def alive(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rpartition(')')[2].split()[0] not in ('Z','T','t')
    except (OSError, IndexError):
        return False


def wait_command(args, logfile):
    S.check_stop()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    stream = logfile.open('w')
    proc = subprocess.Popen(args,cwd=S.REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    stream.close();S.PROCESSES.append(proc)
    while proc.poll() is None:
        S.check_stop();S.save();time.sleep(10)
    if proc.returncode:
        raise RuntimeError(f'Command failed ({proc.returncode}): see {logfile}')


def verify_curves(folder, groups, expected_cases):
    import pandas as pd
    expected_runs={f'{g}_s{s}' for g in groups for s in range(3)}
    check=pd.read_csv(folder/'check_argmax_vs_eval_val.csv')
    assert set(check.run)==expected_runs and len(check)==len(expected_runs)
    assert (check.lesions_compared==92).all() and (check.cases_compared==21).all()
    assert (check.lesion_detection_agree==1).all() and (check.fp_count_abs_diff==0).all()
    for run in expected_runs:
        files=list((folder/'cases'/run).glob('*.json'))
        assert {f.stem for f in files}==expected_cases,(run,len(files))
    sweep=pd.read_csv(folder/'threshold_sweep_per_run.csv')
    assert set(sweep.run)==expected_runs
    assert (sweep.groupby('run').size()==14).all()  # argmax + six thresholds, raw/pp
    assert not list(folder.rglob('*.nii.gz')) and not list(folder.rglob('*.npz'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-root',default='/root/autodl-tmp/Downstream_2')
    p.add_argument('--publish',action='store_true');args=p.parse_args()
    S.CONTROL=Path(args.control_root);S.OUT.mkdir(parents=True,exist_ok=True)
    lock=(S.OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (S.OUT/'state.json').exists() and not (S.OUT/'STOP').exists()
    S.START=time.time();S.STATE.update(phase='waiting_for_gpu',pid=os.getpid(),started_at=S.stamp(),
                                    jobs={},worktree=str(S.REPO),scope='A13a/A13b seeds0/1/2; frozen 21 validation cases only',
                                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.REPO,text=True).strip(),
                                    publish_status='pending' if args.publish else 'disabled')
    processes={};S.save()
    try:
        while len(processes)<3 or any(x.poll() is None for x in processes.values()):
            S.check_stop();base=read_base()
            for gpu in range(3):
                if gpu in processes:continue
                job=base.get('analysis_jobs',{}).get(f'curves_shard{gpu}',{})
                ready=bool(job) and not alive(job.get('pid')) and job.get('exit_code')==0
                if base.get('phase')=='complete':ready=True
                if base.get('phase') in ('failed','cancelled'):
                    raise RuntimeError(f"Base controller {base['phase']}: {base.get('error')}")
                if not ready:continue
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),RESULTS_ROOT=str(S.ROOT),
                           OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                           NUMEXPR_NUM_THREADS='1',nnUNet_def_n_proc='2',PYTHONUNBUFFERED='1')
                logfile=S.OUT/f'curves_A13_shard{gpu}.log';stream=logfile.open('w')
                proc=subprocess.Popen([str(S.PY),'-u',str(S.REPO/'experiments/analysis/missed_lesion_probability.py'),
                                       'run','--runs','A13a:0,1,2','A13b:0,1,2','--shard',f'{gpu}/3',
                                       '--device','cuda','--out',str(S.OUT)],cwd=S.REPO,env=env,
                                      stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                stream.close();S.PROCESSES.append(proc);processes[gpu]=proc
                S.STATE['jobs'][f'curves_A13_shard{gpu}']=dict(pid=proc.pid,gpu=gpu,status='running',started_at=S.stamp())
                S.log(f'A13 shard {gpu}/3 started on GPU {gpu}, PID {proc.pid}')
            for gpu,proc in processes.items():
                if proc.poll() is not None:
                    S.STATE['jobs'][f'curves_A13_shard{gpu}'].update(status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
                    if proc.returncode:raise RuntimeError(f'A13 GPU {gpu} shard failed ({proc.returncode})')
            S.STATE['phase']='running' if processes else 'waiting_for_gpu'
            S.STATE['case_model_results_written']=len(list((S.OUT/'cases').glob('*/*.json')))
            S.save();time.sleep(15)
        S.STATE['phase']='summarizing';S.save()
        wait_command([str(S.PY),str(S.REPO/'experiments/analysis/missed_lesion_probability.py'),
                      'summarize','--out',str(S.OUT),'--results-dir',str(S.REPO/'results')],S.OUT/'curves_A13_summary.log')
        cases=set((S.REPO/'experiments/eval/val_cases.txt').read_text().split());assert len(cases)==21
        verify_curves(S.OUT,('A13a','A13b'),cases)
        # Wait for the original publisher to finish before copying its final
        # state/logs or merging its commits into this independent worktree.
        while True:
            S.check_stop();base=read_base()
            if base.get('phase') in ('failed','cancelled'):raise RuntimeError(base.get('error','Base controller failed'))
            if base.get('publish_status')=='published':break
            S.save();time.sleep(15)
        verify_curves(BASE/'probability_curves',('A0','A1','A11'),cases)
        S.STATE['phase']='archiving';S.save()
        if args.publish:
            wait_command(['git','fetch','origin'],S.OUT/'git_fetch.log')
            wait_command(['git','merge','--ff-only','origin/main'],S.OUT/'git_merge.log')
        target=S.REPO/'results/lowcnr_replication';target.mkdir(parents=True,exist_ok=True)
        a13=target/'probability_curves_A13'
        for path in S.OUT.iterdir():
            if path.name=='cases':shutil.copytree(path,a13/'cases',dirs_exist_ok=True)
            elif path.suffix in ('.csv','.json','.md','.log') and not path.name.startswith('git_'):
                a13.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,a13/path.name)
        for path in BASE.glob('curves*.log'):
            shutil.copyfile(path,target/'probability_curves'/path.name)
        shutil.copyfile(BASE_STATE,target/'state.json')
        shutil.copyfile(BASE/'controller.log',target/'controller.log')
        # Enforce the requested complete artifact contract, not just A13 files.
        for filename in ('REPLICATION_DECISION.md','replication_decisions.json','per_seed_pp4mm.csv',
                         'summary_pp4mm.json','cases_pp4mm.csv','lesions_pp4mm.csv','post4mm.log','state.json','controller.log'):
            assert (target/filename).is_file(),filename
        assert (target/'preflight').is_dir()
        files=list(target.rglob('*'))
        assert not any(p.is_file() and (p.name.endswith(('.nii.gz','.npz','.pth','.pt'))) for p in files)
        S.STATE.update(phase='complete',argmax_check_rows=6,lesions_per_run=92,cases_per_run=21,
                       all_detections_agree=True,fp_count_abs_diff=0,base_published_commit=base['published_commit'])
        S.save();shutil.copyfile(S.OUT/'state.json',a13/'state.json')
        shutil.copyfile(S.OUT/'controller.log',a13/'controller.log')
        if args.publish:
            paths=['results/lowcnr_replication']
            wait_command(['git','add','--',*paths],S.OUT/'git_add.log')
            wait_command(['git','commit','--only','-m','Results: add all A13 validation curves and verify complete replication artifacts','--',*paths],S.OUT/'git_commit.log')
            wait_command(['git','fetch','origin'],S.OUT/'git_fetch_final.log')
            wait_command(['git','merge','--no-edit','origin/main'],S.OUT/'git_merge_final.log')
            wait_command(['git','push','origin','HEAD:main'],S.OUT/'git_push.log')
            S.STATE.update(publish_status='published',published_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.REPO,text=True).strip());S.save()
            S.log(f"Published complete requested artifacts: {S.STATE['published_commit']}")
    except KeyboardInterrupt as e:
        S.terminate_jobs();S.STATE.update(phase='cancelled',error=str(e));S.save();S.log(str(e))
    except Exception as e:
        S.terminate_jobs();S.STATE.update(phase='failed',error=str(e));S.save();S.log(str(e));raise


if __name__=='__main__':
    import signal
    def terminated(*_):raise KeyboardInterrupt('A13 curve controller terminated')
    signal.signal(signal.SIGTERM,terminated)
    main()
