import sys,json,hashlib
from pathlib import Path
T=Path(__file__).resolve().parent
sys.path.insert(0,str(T/'code'))
import run_ablation as r
from propagation_models import make_propagation_model
r.dependencies();r.torch.set_num_threads(1)
data=r.load_data(T,smoke=True)
r.make_model=lambda c,x,g,p,ids:make_propagation_model(c,x,g,p)
plan=json.loads((T/'plan.json').read_text(encoding='utf-8'))
for family in ['P_h0_fixed_all','P_h5_fixed_all','P_h5_signed_all','P_h5_softmax_all','P_h5_fixed_ppi']:
 c=dict(next(j['config'] for j in plan['jobs'] if j['config']['experiment']==family))
 c.update(epochs=2,eval_every=1,patience=2,smoke=True,fold=0)
 c['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (T/'code').glob('*.py')}
 r.train_one(c,data,T/'outputs/smoke'/family,r.torch.device('cpu'),[])
print('PASS: five propagation end-to-end CPU smoke variants')
