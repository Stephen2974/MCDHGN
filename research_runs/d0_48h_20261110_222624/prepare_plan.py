import json,hashlib,datetime
from pathlib import Path
T=Path(__file__).resolve().parent
assert not (T/'plan.json').exists()
common=dict(epochs=1000,eval_every=10,patience=20,lr=0.003,weight_decay=0.0001,positive_weight=1.904,dropout=0.4,hidden_size=256,heads=4,fusion_dim=64,fusion_chunk_size=1024,smoke=False,protocol='softmax-ap-checkpoint-v1',standardize=False)
arch=dict(use_gatv2=False,semantic_mode='global',fusion_mode='none',fusion_alpha=0.6)
variants=[]
def add(name,research,**kw):variants.append(dict(common,experiment=name,architecture=dict(arch),research=research,**kw))
# Avoid passing duplicate keyword defaults; merge explicitly.
def add(name,research,**kw):
 c=dict(common);c.update(kw);c.update(experiment=name,architecture=dict(kw.get('architecture',arch)),research=research);variants.append(c)
add('A00_GAT_reference_lr01',{'family':'mcdhgn'},lr=0.01)
add('A01_raw_lr003',{'family':'raw'})
add('A02_raw_lr01',{'family':'raw'},lr=0.01)
add('A03_raw_zscore_lr003',{'family':'raw'},standardize=True)
for hops in [0,1,2]:
 for raw in [False,True]:
  add('A_SIGN_h%d_raw%d'%(hops,raw),{'family':'sign','hops':hops,'raw_branch':raw,'width':128})
add('A_SIGN_h2_raw1_zscore',{'family':'sign','hops':2,'raw_branch':True,'width':128},standardize=True)
add('A_SIGN_h2_raw1_global',{'family':'sign','hops':2,'raw_branch':True,'width':128,'semantic_mode':'global'})
# Full 2x2 raw-residual x LayerNorm within each encoder, same optimizer.
for gatv2 in [False,True]:
 for raw,norm in [(False,False),(True,False),(False,True),(True,True)]:
  add('B_%s_raw%d_norm%d'%('GATv2node' if gatv2 else 'GATglobal',raw,norm),{'family':'mcdhgn','raw_residual':raw,'layer_norm':norm},architecture=dict(arch,use_gatv2=gatv2,semantic_mode='node' if gatv2 else 'global'))
jobs=[]
for c in variants:
 for fold in range(5):jobs.append({'id':c['experiment']+'_s42_f'+str(fold),'config':dict(c,seed=42,fold=fold)})
files=[p for p in (T/'code').glob('*.py')]+[T/'worker.py',T/'scheduler.py']
plan=dict(schema=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization_start_utc='2026-11-10T14:26:23+00:00',deadline_utc='2026-11-12T14:26:23+00:00',per_fold_timeout_seconds=14400,gpu_ids=[0,1,2],jobs=jobs,execution_sha256={p.relative_to(T).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},selection_rule='Complete five folds per configuration, rank by mean AP; AUC must also exceed .9 for target. No foldwise cherry picking. Seed42 exploratory, repeat selected configurations on seed43/44 before robust claims.',failure_rule='Nonfinite loss, worker nonzero exit, missing result, four-hour fold timeout or global deadline. No automatic retries; retain all failed outputs.',next_phase='After all first-phase five-fold results, select best model and key ablations for seed43/44; append jobs only after prespecified decision record. No final test access.')
(T/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(T/'audit/protocol.md').write_text('''# D0 48-hour development experiment

Authorization: 2026-11-10 22:26:23 Taipei to 2026-11-12 22:26:23 Taipei. Maximum three idle RTX2080Ti, one fold per device. All task-owned workers have a four-hour timeout and are terminated at the deadline. Never stop foreign processes.

Data: frozen legacy original-project D0, 12944 x 48, original nine cached metapaths and five development folds. Audit verified 12 training-input SHA256 fingerprints against the baseline manifest, source TSV features exactly equal graph features, and split integrity. No test labels read. This verifies frozen-file identity, not missing historical source provenance.

Primary metric: mean of five validation APs (sklearn average_precision_score), secondary ROC AUC from the same AP-selected checkpoint. Target requires BOTH means >0.9. Report individual folds and sample SD; validation folds are dependent. Every variant uses all five folds, seed42, same epoch limit/eval/patience/loss. No per-fold model or learning-rate selection. These are development/model-selection results, not unbiased test estimates.

Initial hypotheses: (1) original feature signal bypasses graph smoothing; (2) incoming-mean multi-hop features improve over hop0; (3) raw residual and LayerNorm have separate effects in a 2x2 design. Raw MLP beta is only a placeholder. SIGN is inspired by https://arxiv.org/abs/2004.11198; implemented with frozen metapath incoming means, not a reproduction claim. Raw branch and normalization are separate controls. Additional raw residual parameters require capacity control before attribution.

No new dependencies or datasets, no label propagation, no external labels, no manifest/data changes. Standardization is optional and fitted on each fold's training nodes only; parameters saved. Existing source code is copied and never modified. Active experiment code is hash-frozen; later versions must use new code and outputs. Existing result directories are only read.

Initial validation: 22 existing ablation/summary tests passed in isolated code copy. Audit helper initially assumed comma-separated source despite .csv name; inspection showed TSV, corrected separator, reran exact comparison successfully. This was an audit parser failure before any training, not a data modification.

Literature: MCDHGN https://pmc.ncbi.nlm.nih.gov/articles/PMC11256962/; APPNP https://arxiv.org/abs/1810.05997; GPR-GNN https://arxiv.org/abs/2006.07988; GCNII https://proceedings.mlr.press/v119/chen20v.html. Later stages require complete first-stage results and recorded hypotheses, plus repeats of best and controls on seed43/44 within remaining budget. No guarantee of reaching .9 and no suppression of failures.
''',encoding='utf-8')
print(len(variants),'configurations',len(jobs),'fold jobs')
