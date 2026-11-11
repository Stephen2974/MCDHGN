"""Read-only supplemental audit; writes a new evidence file only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import json, pickle, datetime as dt
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

project=Path('G:/LLLM/MCDHGN-main')
root=project/'research_runs/d0_tabular_20261111_1540'
plan=json.loads((root/'plan.json').read_text())
with (project/'Intermediate/label_set/train_batch.pkl').open('rb') as f: original=pickle.load(f)
folds=[]
with np.load(root/'cache/splits.npz') as z:
    for fold, source in enumerate(original):
        values=[]
        for key,value in zip(('train','val','train_y','val_y'),source):
            if hasattr(value,'cpu'):value=value.cpu().numpy()
            a=z[f'{fold}_{key}'];np.testing.assert_array_equal(a,np.asarray(value).reshape(-1));values.append(a)
        tr,va,ty,vy=values
        assert len(np.unique(tr))==len(tr) and len(np.unique(va))==len(va)
        assert not len(np.intersect1d(tr,va))
        assert len(tr)==len(ty) and len(va)==len(vy)
        folds.append(dict(fold=fold,train=len(tr),validation=len(va),overlap=0))
selection=[]
for job in plan['jobs']:
    c=job['config'];d=root/'outputs'/c['experiment']/f"seed{c['seed']}"/f"fold{c['fold']}"
    r=json.loads((d/'result.json').read_text());h=[json.loads(l) for l in (d/'history.jsonl').read_text().splitlines()]
    best=max(h,key=lambda v:v['ap'])
    assert r['best_epoch']==best['epoch'] and r['epochs_trained']==h[-1]['epoch']
    for m in ('ap','auc'):assert abs(best[m]-r['validation'][m])<1e-12
    if c['estimator']=='hgb':
        assert [v['epoch'] for v in h]==list(range(c['eval_every'],r['epochs_trained']+1,c['eval_every']))
        peak=-1;stale=0
        for i,v in enumerate(h):
            if v['ap']>peak:peak=v['ap'];stale=0
            else:stale+=1
            if i<len(h)-1:assert stale<c['patience']
        assert r['epochs_trained']==c['epochs'] or stale==c['patience']
    else:assert len(h)==1 and r['best_epoch']==500
    selection.append(dict(job=job['id'],best_epoch=r['best_epoch'],epochs_trained=r['epochs_trained']))
baseline=[]
for fold in range(5):
    d=project/'research_runs/d0_48h_20261110_222624/outputs/C_v2_original_std0_raw0/seed42'/f'fold{fold}'
    with np.load(d/'validation_predictions.npz') as p:
        assert np.isfinite(p['probabilities']).all() and ((p['probabilities']>=0)&(p['probabilities']<=1)).all()
        scores=dict(ap=float(average_precision_score(p['labels'],p['probabilities'])),auc=float(roc_auc_score(p['labels'],p['probabilities'])))
    r=json.loads((d/'result.json').read_text())
    for m in scores:assert abs(scores[m]-r['validation'][m])<1e-12
    baseline.append(dict(fold=fold,**scores))
events=[json.loads(l) for l in (root/'logs/scheduler.jsonl').read_text().splitlines()]
failures=[e for e in events if e['event'] in ('failed','timeout','budget_terminated')]
completed=[e for e in events if e['event']=='completed']
assert len(completed)==40 and {e['job'] for e in completed}=={j['id'] for j in plan['jobs']}
report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),folds=folds,selection=selection,baseline=baseline,failures=failures,completed_events=len(completed),last_event=events[-1],nonempty_stderr=[p.name for p in (root/'logs').glob('*.err') if p.stat().st_size],test_labels_read=False)
path=root/'audit'/('supplement_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
with path.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(dict(path=str(path),folds=folds,selection_verified=len(selection),failures=failures,nonempty_stderr=report['nonempty_stderr'])))
