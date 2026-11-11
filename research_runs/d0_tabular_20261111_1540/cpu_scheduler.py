"""One task-owned CPU worker, two threads, original deadline and four-hour cap."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tabular_worker import digest

TASK = Path(__file__).resolve().parent


def emit(event, **kw):
    row = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), event=event, **kw)
    with (TASK / 'logs/scheduler.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def main():
    lock = TASK / 'scheduler.lock'
    with lock.open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid()}, f)
    proc = None
    try:
        plan = json.loads((TASK / 'plan.json').read_text(encoding='utf-8'))
        deadline = dt.datetime.fromisoformat(plan['deadline_utc'])
        emit('scheduler_started', pid=os.getpid(), jobs=len(plan['jobs']), deadline=plan['deadline_utc'])
        for job in plan['jobs']:
            if dt.datetime.now(dt.timezone.utc) >= deadline:
                emit('budget_finished')
                break
            assert all(digest(TASK / p) == h for p, h in plan['execution_sha256'].items())
            # Fail closed on pre-existing output, including incomplete output.
            config = job['config']
            dest = TASK / 'outputs' / config['experiment'] / ('seed' + str(config['seed'])) / ('fold' + str(config['fold']))
            if dest.exists():
                raise RuntimeError('Pre-existing output: ' + str(dest))
            with (TASK / 'logs' / (job['id'] + '.out')).open('x') as out, (TASK / 'logs' / (job['id'] + '.err')).open('x') as err:
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
                           CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                started = time.monotonic()
                proc = subprocess.Popen([sys.executable, '-B', str(TASK / 'tabular_worker.py'), '--job', job['id']],
                                        cwd=str(TASK), stdout=out, stderr=err, env=env,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
                emit('started', job=job['id'], pid=proc.pid)
                reason = None
                while proc.poll() is None:
                    if dt.datetime.now(dt.timezone.utc) >= deadline:
                        reason = 'budget_terminated'
                    elif time.monotonic() - started >= plan['per_fold_timeout_seconds']:
                        reason = 'timeout'
                    if reason:
                        proc.terminate()
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            proc.kill(); proc.wait(timeout=20)
                        break
                    time.sleep(3)
                ok = proc.returncode == 0 and (dest / 'result.json').exists()
                emit(reason or ('completed' if ok else 'failed'), job=job['id'], exit_code=proc.returncode,
                     seconds=time.monotonic() - started)
                proc = None
        else:
            emit('queue_finished')
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(timeout=20)
            emit('failed', reason='Scheduler interrupted; its worker stopped')
        lock.unlink()


if __name__ == '__main__':
    main()
