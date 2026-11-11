import json,datetime,hashlib,copy
from pathlib import Path
T=Path(__file__).resolve().parent
assert not (T/'scheduler.lock').exists()
p=json.loads((T/'plan.json').read_text(encoding='utf-8'))
assert len(p['jobs'])==100
assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(p['deadline_utc'])
for j in p['jobs']:
 c=j['config'];assert (T/'outputs'/c['experiment']/('seed'+str(c['seed']))/('fold'+str(c['fold']))/'result.json').exists()
assert all(hashlib.sha256((T/k).read_bytes()).hexdigest()==v for k,v in p['execution_sha256'].items())
(T/'audit/plan_phase1.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
base=next(j['config'] for j in p['jobs'] if j['config']['experiment']=='B_GATv2node_raw0_norm0')
for head in ['original','linear']:
 for standardize in [False,True]:
  for raw in [False,True]:
   c=copy.deepcopy(base)
   c.update(lr=0.01,standardize=standardize,experiment='C_v2_%s_std%d_raw%d'%(head,standardize,raw))
   c['research'].update(head=head,raw_residual=raw,layer_norm=False)
   for fold in range(5):
    cf=dict(c,fold=fold)
    p['jobs'].append({'id':c['experiment']+'_s42_f'+str(fold),'config':cf})
note={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision':'Phase1 100/100 completed. No variant beat original GAT; do not claim improvement. Test head x train-only input standardization x raw residual at original lr=.01 on GATv2 node, full 2x2x2 design. Retain lr=.003 prior controls. Seed42 all five folds for every cell. Linear head is a capacity reduction, not added capacity. Select only after all cells complete. No test access. Deadline unchanged.','new_jobs':40}
(T/'audit/phase2_decision.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
tmp=T/'plan.phase2.tmp';tmp.write_text(json.dumps(p,indent=2),encoding='utf-8');tmp.replace(T/'plan.json')
print(json.dumps(note))
