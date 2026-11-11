import copy
import json
from pathlib import Path
import sys
TASK=Path(__file__).resolve().parent
sys.path.insert(0,str(TASK/'code'))
import run_ablation as r
from research_models import make_research_model
r.dependencies();r.torch.set_num_threads(1)
data=r.load_data(TASK,smoke=True)
r.make_model=lambda c,x,g,p,ids:make_research_model(c,x,g,p)
plan=json.loads((TASK/'plan.json').read_text(encoding='utf-8'))
for experiment in sorted(set(j['config']['experiment'] for j in plan['jobs'])):
    c=copy.deepcopy(next(j['config'] for j in plan['jobs'] if j['config']['experiment']==experiment))
    c.update(epochs=2,eval_every=1,patience=2,smoke=True,fold=0,hidden_size=8,heads=2,fusion_dim=4)
    c['code_sha256']={p.name:r.sha256(p) for p in (TASK/'code').glob('*.py')}
    r.train_one(c,data,TASK/'outputs/smoke'/experiment,r.torch.device('cpu'),[])
print('PASS: six CPU synthetic end-to-end ranking variants')
