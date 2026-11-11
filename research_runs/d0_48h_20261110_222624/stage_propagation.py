import json,hashlib,shutil,datetime
from pathlib import Path
S=Path(__file__).resolve().parent
T=S.parent/'d0_propagation_20261111'
assert not T.exists()
for d in ['code','audit','outputs','logs']:(T/d).mkdir(parents=True,exist_ok=True)
for p in (S/'code').glob('*.py'):
 if p.name!='research_models.py':shutil.copy2(p,T/'code'/p.name)
for p in (S/'candidate_v2').glob('*.py'):shutil.copy2(p,T/'code'/p.name)
(T/'code/research_models.py').write_text('from propagation_models import make_propagation_model as make_research_model\n',encoding='utf-8')
for n in ['worker.py','scheduler.py','launch_detached.py','status.py','verify_results.py']:shutil.copy2(S/n,T/n)
shutil.copy2(S/'audit/data_audit.json',T/'audit/data_audit.json')
old=json.loads((S/'plan.json').read_text(encoding='utf-8'))
base=dict(old['jobs'][0]['config'],lr=.003,standardize=False)
jobs=[]
matrix=[(0,'fixed','all'),(5,'fixed','all'),(10,'fixed','all'),(5,'signed','all'),(10,'signed','all'),(5,'softmax','all'),(5,'fixed','ppi'),(5,'signed','ppi')]
for hops,mode,paths in matrix:
 name='P_h%d_%s_%s'%(hops,mode,paths)
 for fold in range(5):
  c=dict(base,experiment=name,seed=42,fold=fold,research=dict(family='propagation',hops=hops,alpha=.1,gamma_mode=mode,width=128,path_mode=paths))
  jobs.append({'id':name+'_s42_f'+str(fold),'config':c})
files=list((T/'code').glob('*.py'))+[T/'worker.py',T/'scheduler.py']
p=dict(old,jobs=jobs,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_sha256={f.relative_to(T).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
p['next_phase']='Staged only. Admission requires original queue terminal and no foreign GPU workers; same immutable global deadline. Full five-fold comparisons.'
(T/'plan.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
print('STAGED ONLY',T,len(jobs),'jobs; no process launched')
