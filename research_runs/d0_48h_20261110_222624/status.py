import json,datetime,statistics,hashlib
from pathlib import Path
T=Path(__file__).resolve().parent
p=json.loads((T/'plan.json').read_text(encoding='utf-8'))
groups={};completed=0
for j in p['jobs']:
 c=j['config'];path=T/'outputs'/c['experiment']/('seed'+str(c['seed']))/('fold'+str(c['fold']))/'result.json'
 if path.exists():
  row=json.loads(path.read_text(encoding='utf-8'));completed+=1
  groups.setdefault((c['experiment'],c['seed']),[]).append(row)
summary=[]
for (name,seed),rows in groups.items():
 summary.append(dict(experiment=name,seed=seed,n=len(rows),folds=sorted(r['config']['fold'] for r in rows),ap=statistics.mean(r['validation']['ap'] for r in rows),auc=statistics.mean(r['validation']['auc'] for r in rows),ap_sd=statistics.stdev(r['validation']['ap'] for r in rows) if len(rows)>1 else None,complete=len(rows)==5))
events=[]
if (T/'logs/scheduler.jsonl').exists():events=[json.loads(x) for x in (T/'logs/scheduler.jsonl').read_text(encoding='utf-8').splitlines()]
active={}
for e in events:
 if e['event']=='started':active[e['job']]=e
 elif e['event'] in ['completed','failed','timeout','budget_terminated']:active.pop(e['job'],None)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline=p['deadline_utc'],completed=completed,planned=len(p['jobs']),active=list(active.values()),failures=[e for e in events if e['event'] in ['failed','timeout','budget_terminated']],groups=sorted(summary,key=lambda x:(not x['complete'],-x['ap']))),indent=2))
