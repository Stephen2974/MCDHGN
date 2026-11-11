"""Read-only, timestamped snapshots for this task's experiment directories."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(r'G:\LLLM\MCDHGN-main')
RUN_NAMES = ['d0_48h_20261110_222624', 'd0_propagation_20261111',
             'd0_tabular_20261111_1540', 'd0_ranking_20261111']
DEADLINE = dt.datetime.fromisoformat('2026-11-12T14:26:23+00:00')


def snapshot():
    report = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'deadline_utc': DEADLINE.isoformat(), 'runs': []}
    for name in RUN_NAMES:
        root = ROOT / 'research_runs' / name
        if not (root / 'plan.json').exists():
            continue
        plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
        groups, errors, active, events, completed = {}, [], {}, [], 0
        for line in (root / 'logs/scheduler.jsonl').read_text(encoding='utf-8').splitlines() if (root / 'logs/scheduler.jsonl').exists() else []:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # Current writer may be completing its last line.
            events.append(event)
            if event['event'] == 'started':
                active[event['job']] = event
            elif event['event'] in ('completed','failed','timeout','budget_terminated'):
                active.pop(event.get('job'), None)
        for job in plan['jobs']:
            c = job['config']
            path = root / 'outputs' / c['experiment'] / ('seed' + str(c['seed'])) / ('fold' + str(c['fold'])) / 'result.json'
            if not path.exists():
                continue
            try:
                row = json.loads(path.read_text(encoding='utf-8'))
                assert row['config']['experiment'] == c['experiment'] and row['config']['fold'] == c['fold']
            except (json.JSONDecodeError, AssertionError) as exc:
                errors.append(str(path) + ':' + type(exc).__name__)
                continue
            completed += 1
            groups.setdefault((c['experiment'],c['seed']), []).append(row)
        summaries = []
        for (experiment,seed), rows in groups.items():
            folds = sorted(row['config']['fold'] for row in rows)
            aps = [row['validation']['ap'] for row in rows]
            aucs = [row['validation']['auc'] for row in rows]
            summaries.append(dict(experiment=experiment,seed=seed,n=len(rows),folds=folds,
                                  complete=folds==list(range(5)),ap=statistics.mean(aps),auc=statistics.mean(aucs),
                                  ap_sd=statistics.stdev(aps) if len(aps)>1 else None,
                                  auc_sd=statistics.stdev(aucs) if len(aucs)>1 else None,
                                  seconds=sum(row.get('train_seconds',0) for row in rows)))
        mismatches=[path for path,h in plan['execution_sha256'].items()
                    if hashlib.sha256((root / path).read_bytes()).hexdigest()!=h]
        report['runs'].append(dict(name=name,completed=completed,planned=len(plan['jobs']),
                                   active=list(active.values()),groups=sorted(summaries,key=lambda g:(not g['complete'],-g['ap'])),
                                   failures=[e for e in events if e['event'] in ('failed','timeout','budget_terminated')],
                                   errors=errors,hash_mismatches=mismatches,
                                   last_event=events[-1] if events else None,
                                   has_lock=(root/'scheduler.lock').exists()))
    try:
        report['gpus'] = subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                                                   '--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()
    except Exception as exc:
        report['gpu_error'] = str(exc)
    return report


def save(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    target = directory / (stamp+'.json')
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    temp = directory/'latest.tmp'
    temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    temp.replace(directory/'latest.json')
    print(json.dumps({'report':str(target),'utc':report['utc'],
                      'counts':{r['name']:str(r['completed'])+'/'+str(r['planned']) for r in report['runs']}}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--json',action='store_true')
    args=p.parse_args(); directory=Path(__file__).resolve().parent/'reports'
    if args.json:
        print(json.dumps(snapshot(),indent=2));return
    if not args.watch:
        save(snapshot(),directory);return
    lock=directory.parent/'hourly.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'interval_seconds':3600},f)
    try:
        next_run=dt.datetime.now(dt.timezone.utc)
        while True:
            now=dt.datetime.now(dt.timezone.utc)
            if now>=next_run or now>=DEADLINE:
                save(snapshot(),directory)
                next_run+=dt.timedelta(hours=1)
            if now>=DEADLINE:break
            time.sleep(min(30,max(1,(next_run-now).total_seconds())))
    finally:lock.unlink()


if __name__=='__main__':main()
