"""One audited retry after the original queue; no live code changes."""
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(r'G:\LLLM\MCDHGN-main')
S = ROOT/'research_runs/d0_48h_20261110_222624'
P = ROOT/'research_runs/d0_propagation_20261111'
R = Path(__file__).resolve().parent
JOB = 'C_v2_original_std1_raw1_s42_f2'
DEADLINE = dt.datetime.fromisoformat('2026-11-12T14:26:23+00:00')

def emit(event, **kw):
    with (R/'recovery.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), event=event, **kw))+'\n')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check_time():
    if dt.datetime.now(dt.timezone.utc) >= DEADLINE:
        raise RuntimeError('Research deadline reached')

def result_path(job):
    c = job['config']
    return S/'outputs'/c['experiment']/('seed'+str(c['seed']))/('fold'+str(c['fold']))/'result.json'

def frozen(root, plan):
    assert all(digest(root/p)==h for p,h in plan['execution_sha256'].items()), 'Execution hash changed'

def wait_handle(pid):
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    k.OpenProcess.restype=ctypes.c_void_p
    k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    k.CloseHandle.argtypes=[ctypes.c_void_p]
    h=k.OpenProcess(0x100000,False,pid)
    assert h, 'Expected process unavailable; manual review required'
    return k,h

def main():
    with (R/'recovery.lock').open('x') as f:
        json.dump({'pid':os.getpid()},f)
    try:
        # Capture both live process handles before waiting to avoid PID reuse.
        k,h=wait_handle(9680)
        ka,ha=wait_handle(26468)
        emit('waiting', scheduler_pid=9680, admission_pid=26468)
        try:
            for kernel,handle in [(k,h),(ka,ha)]:
                while True:
                    check_time()
                    state=kernel.WaitForSingleObject(handle,30000)
                    if state==0: break
                    assert state==258, 'Process wait failed'
        finally:
            k.CloseHandle(h);ka.CloseHandle(ha)
        plan=json.loads((S/'plan.json').read_text())
        assert len(plan['jobs'])==140
        assert not (S/'scheduler.lock').exists()
        events=[json.loads(x) for x in (S/'logs/scheduler.jsonl').read_text().splitlines()]
        assert events[-1]['event']=='queue_finished'
        missing=[j for j in plan['jobs'] if not result_path(j).exists()]
        assert [j['id'] for j in missing]==[JOB], 'Unexpected missing jobs'
        frozen(S,plan)
        assert not (P/'scheduler.lock').exists()
        assert not (P/'logs/scheduler.jsonl').exists(), 'Propagation already started'
        spec=importlib.util.spec_from_file_location('original_scheduler', S/'scheduler.py')
        scheduler=importlib.util.module_from_spec(spec);spec.loader.exec_module(scheduler)
        while True:
            check_time()
            free=[i for i in plan['gpu_ids'] if i not in scheduler.blocked()]
            if free: break
            time.sleep(30)
        dest=result_path(missing[0]).parent
        assert dest.resolve().is_relative_to((S/'outputs').resolve())
        archive=R/'failed_attempt'
        assert not archive.exists()
        manifest={str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()}
        (R/'failed_manifest.json').write_text(json.dumps(manifest,indent=2))
        dest.rename(archive)
        assert all(digest(archive/p)==h for p,h in manifest.items())
        emit('failed_attempt_archived', manifest=manifest)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',DGLBACKEND='pytorch',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (R/'retry.out').open('x') as out, (R/'retry.err').open('x') as err:
            proc=subprocess.Popen([sys.executable,'-B',str(S/'worker.py'),'--job',JOB,'--gpu',str(free[0])],cwd=S/'code',env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            emit('retry_started',pid=proc.pid,gpu=free[0],job=JOB)
            limit=min(plan['per_fold_timeout_seconds'],(DEADLINE-dt.datetime.now(dt.timezone.utc)).total_seconds())
            try: code=proc.wait(timeout=max(1,limit))
            except BaseException:
                proc.terminate()
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: proc.kill();proc.wait()
                raise
        assert code==0 and result_path(missing[0]).exists(), 'Retry failed; evidence retained'
        value=json.loads(result_path(missing[0]).read_text())
        assert all(value['config'][k]==v for k,v in missing[0]['config'].items())
        import numpy as np
        from sklearn.metrics import average_precision_score,roc_auc_score
        with np.load(dest/'validation_predictions.npz') as predictions:
            labels=predictions['labels']; probabilities=predictions['probabilities']
            assert np.isfinite(probabilities).all()
            ap=average_precision_score(labels,probabilities)
            auc=roc_auc_score(labels,probabilities)
            assert abs(ap-value['validation']['ap'])<1e-10
            assert abs(auc-value['validation']['auc'])<1e-10
        import torch
        checkpoint=torch.load(dest/'best.pt',map_location='cpu')
        assert checkpoint['best_epoch']==value['best_epoch']
        assert checkpoint['config']==value['config']
        assert checkpoint['data_sha256']==value['data_sha256']
        frozen(S,plan)
        assert all(result_path(j).exists() for j in plan['jobs'])
        emit('retry_completed',job=JOB,result_sha256=digest(dest/'result.json'),validation=value['validation'])
        check_time()
        pp=json.loads((P/'plan.json').read_text())
        assert dt.datetime.fromisoformat(pp['deadline_utc'])==DEADLINE
        frozen(P,pp)
        assert len(list((P/'outputs/smoke').glob('*/result.json')))==5
        assert not (P/'scheduler.lock').exists()
        emit('propagation_admission_passed', jobs=len(pp['jobs']))
        runpy.run_path(str(P/'scheduler.py'),run_name='__main__')
    except BaseException as exc:
        emit('recovery_stopped',error=repr(exc))
        raise
    finally:
        (R/'recovery.lock').unlink()

if __name__=='__main__':
    sys.stdout=(R/'detached.out').open('x',encoding='utf-8',buffering=1)
    sys.stderr=(R/'detached.err').open('x',encoding='utf-8',buffering=1)
    main()
