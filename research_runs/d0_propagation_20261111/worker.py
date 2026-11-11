"""Isolated worker: development labels only, exact legacy protocol."""
import os
os.environ['DGLBACKEND']='pytorch'
os.environ['PYTHONIOENCODING']='utf-8'
import sys,json,argparse,datetime
from pathlib import Path
TASK=Path(__file__).resolve().parent
ROOT=TASK.parent.parent
sys.path.insert(0,str(TASK/'code'))
import run_ablation as r

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--job',required=True)
 p.add_argument('--gpu',type=int,required=True)
 p.add_argument('--smoke',action='store_true')
 a=p.parse_args()
 r.dependencies()
 r.torch.set_num_threads(4)
 from research_models import make_research_model
 plan=json.loads((TASK/'plan.json').read_text(encoding='utf-8'))
 job=next(j for j in plan['jobs'] if j['id']==a.job)
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(plan['deadline_utc']):raise RuntimeError('Budget expired')
 config=dict(job['config'])
 config['code_sha256']={p.name:r.sha256(p) for p in sorted((TASK/'code').glob('*.py'))}
 config['code_sha256']['../worker.py']=r.sha256(Path(__file__))
 data=r.load_data(ROOT,smoke=a.smoke)
 if not a.smoke:
  audit=json.loads((TASK/'audit/data_audit.json').read_text(encoding='utf-8'))
  assert audit['status']=='passed' and data[3]==audit['data_sha256'],'Data changed after audit'
 x,graphs,splits,fp=data
 if config.get('standardize',False):
  tx=x[splits[config['fold']][0]]
  mean=tx.mean(0); scale=tx.std(0,unbiased=False).clamp_min(1e-6)
  x=(x-mean)/scale
  config['preprocessing']={'method':'train-nodes-only-zscore','mean':mean.tolist(),'scale':scale.tolist()}
 data=x,graphs,splits,fp
 if a.smoke:
  config.update(epochs=2,eval_every=1,patience=2,hidden_size=8,heads=2,fusion_dim=4,smoke=True,fold=0)
  output=TASK/'outputs'/'smoke'/job['id']
 else: output=TASK/'outputs'/config['experiment']/('seed'+str(config['seed']))/('fold'+str(config['fold']))
 primary=r.torch.device('cuda:'+str(a.gpu))
 r.make_model=lambda c,x,g,p,ids:make_research_model(c,x,g,p)
 r.train_one(config,data,output,primary,[a.gpu])

if __name__=='__main__':main()
