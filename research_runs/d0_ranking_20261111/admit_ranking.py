"""Queue behind the existing schedulers, without competing for their GPUs."""
import datetime as dt
import json
from pathlib import Path
import runpy
import sys
import time
import traceback

TASK=Path(__file__).resolve().parent
sys.stdout=(TASK/'logs/admission.out').open('x',encoding='utf-8',buffering=1)
sys.stderr=(TASK/'logs/admission.err').open('x',encoding='utf-8',buffering=1)


def main():
    deadline=dt.datetime.fromisoformat('2026-11-12T14:26:23+00:00')
    print('Waiting for original and propagation queues',flush=True)
    while dt.datetime.now(dt.timezone.utc)<deadline:
        ready=True
        for name,expected in [('d0_48h_20261110_222624',140),('d0_propagation_20261111',40)]:
            root=TASK.parent/name
            plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
            assert len(plan['jobs'])==expected,'Prior plan changed; review needed'
            ledger=root/'logs/scheduler.jsonl'
            if (root/'scheduler.lock').exists() or not ledger.exists():ready=False;break
            lines=ledger.read_text(encoding='utf-8').splitlines()
            if not lines or json.loads(lines[-1])['event']!='queue_finished':ready=False;break
            for job in plan['jobs']:
                c=job['config']
                if not (root/'outputs'/c['experiment']/('seed'+str(c['seed']))/('fold'+str(c['fold']))/'result.json').exists():
                    ready=False;break
        if ready:
            assert len(list((TASK/'outputs/smoke').glob('*/result.json')))==6
            print('Admission passed',dt.datetime.now(dt.timezone.utc).isoformat(),flush=True)
            runpy.run_path(str(TASK/'scheduler.py'),run_name='__main__')
            return
        time.sleep(30)
    print('Deadline reached before admission',flush=True)


try:main()
except BaseException:traceback.print_exc();raise
