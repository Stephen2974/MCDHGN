import os
os.environ['DGLBACKEND']='pytorch'
import sys,json,hashlib,datetime
from pathlib import Path
TASK=Path(__file__).resolve().parent
ROOT=TASK.parent.parent
sys.path.insert(0,str(TASK/'code'))
import run_ablation as r
r.dependencies()
r.torch.set_num_threads(4)
baseline=json.loads((ROOT/'data_versions/p0_v1/baseline_manifest.json').read_text(encoding='utf-8'))
frozen={x['path']:x for x in baseline['files']}
x,graphs,splits,fp=r.load_data(ROOT)
checks={p:{'current':h,'frozen':frozen.get(p,{}).get('sha256'),'matches':h==frozen.get(p,{}).get('sha256')} for p,h in fp.items()}
assert all(a['matches'] for a in checks.values()),checks
assert list(x.shape)==[12944,48]
assert len(splits)==5
# Compare original feature CSV with graph feature rows via original gene IDs.
import pickle,pandas as pd,numpy as np
with (ROOT/'data/step1/gene_toids.pkl').open('rb') as f: mapping=pickle.load(f)
csv=ROOT/'data/biological_features.csv'
assert r.sha256(csv)==frozen[csv.relative_to(ROOT).as_posix()]['sha256']
frame=pd.read_csv(csv,index_col=0,sep='\t')
ordered=[g for g,i in sorted(mapping.items(),key=lambda z:z[1])]
source=frame.loc[ordered].to_numpy(dtype=np.float32)
assert source.shape==tuple(x.shape),(source.shape,x.shape)
assert np.array_equal(source,x.numpy())
results=[]
for dirname in ['p0_legacy_D0','d0_E4_seed42_20261105_183857','D0_E6_seed42_20261107_124312','D0_E7_seed42_20261107_121618','e0_unified_20261106_170741']:
 for path in (ROOT/'results'/dirname).rglob('result.json'):
  row=json.loads(path.read_text(encoding='utf-8'))
  if row.get('config',{}).get('seed')!=42:continue
  results.append({'path':str(path),'data_match':row.get('data_sha256')==fp,'experiment':row['config']['experiment'],'fold':row['config']['fold'],'validation':row['validation'],'config':row['config'],'train_seconds':row.get('train_seconds')})
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','data_sha256':fp,'frozen_checks':checks,'features_exact_csv_match':True,'feature_shape':list(x.shape),'edges':[g.num_edges() for g in graphs],'folds':[{'train_n':len(t),'val_n':len(v),'train_positive':int(ty.sum()),'val_positive':int(vy.sum())} for t,v,ty,vy in splits],'test_labels_read':False,'provenance_limitations':['Frozen legacy original-project data, not independently authenticated publication release','Historical negative construction inputs unavailable','Historical test use unknown'],'baseline_results':results}
r.write_json(TASK/'audit/data_audit.json',out)
print(json.dumps({k:v for k,v in out.items() if k not in ['baseline_results','frozen_checks','data_sha256']},indent=2))
for exp in sorted(set(z['experiment'] for z in results)):
 rows=[z for z in results if z['experiment']==exp]
 print(exp,'n=',len(rows),'data_match=',all(z['data_match'] for z in rows),'AP=',np.mean([z['validation']['ap'] for z in rows]),'AUC=',np.mean([z['validation']['auc'] for z in rows]),'seconds=',sum(z['train_seconds'] or 0 for z in rows))
