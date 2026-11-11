"""Recompute saved validation metrics and verify exact original fold membership."""
import json,pickle,datetime,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
T=Path(__file__).resolve().parent
R=T.parent.parent
with (R/'Intermediate/label_set/train_batch.pkl').open('rb') as f:splits=pickle.load(f)
audit=json.loads((T/'audit/data_audit.json').read_text(encoding='utf-8'))
plan=json.loads((T/'plan.json').read_text(encoding='utf-8'))
checked=[]
for job in plan['jobs']:
 c=job['config'];d=T/'outputs'/c['experiment']/('seed'+str(c['seed']))/('fold'+str(c['fold']))
 if not (d/'result.json').exists():continue
 r=json.loads((d/'result.json').read_text(encoding='utf-8'))
 assert r['data_sha256']==audit['data_sha256']
 for k,v in c.items():assert r['config'][k]==v,(job['id'],k)
 with np.load(d/'validation_predictions.npz') as p:
  assert np.array_equal(p['node_ids'],np.asarray(splits[c['fold']][1]))
  assert np.array_equal(p['labels'],np.asarray(splits[c['fold']][3]))
  assert np.isfinite(p['probabilities']).all()
  ap=average_precision_score(p['labels'],p['probabilities'])
  auc=roc_auc_score(p['labels'],p['probabilities'])
  assert abs(ap-r['validation']['ap'])<1e-12
  assert abs(auc-r['validation']['auc'])<1e-12
 history=[json.loads(x) for x in (d/'history.jsonl').read_text(encoding='utf-8').splitlines()]
 best=max(history,key=lambda x:x['ap'])
 assert best['epoch']==r['best_epoch']
 assert abs(best['ap']-ap)<1e-7
 assert not (d/'test_result.json').exists()
 checked.append(job['id'])
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','checked':checked,'test_labels_read':False}
name='result_verification_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json'
(T/'audit'/name).write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
