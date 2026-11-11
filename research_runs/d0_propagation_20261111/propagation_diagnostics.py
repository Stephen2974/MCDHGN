import sys,json,datetime,argparse
from pathlib import Path
import torch
T=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
root=T/'outputs/smoke' if a.smoke else T/'outputs'
rows=[]
for path in root.rglob('result.json'):
 if not a.smoke and 'smoke' in path.parts:continue
 r=json.loads(path.read_text(encoding='utf-8'))
 ck=torch.load(path.parent/'best.pt',map_location='cpu')
 assert ck['config']==r['config'] and ck['data_sha256']==r['data_sha256']
 state=ck['state_dict'];mode=r['config']['research']['gamma_mode']
 gamma=state['gamma_logits'].softmax(0) if mode=='softmax' else state['gamma']
 assert torch.isfinite(gamma).all()
 weights=state['path_logits'].softmax(0)
 rows.append({'experiment':r['config']['experiment'],'seed':r['config']['seed'],'fold':r['config']['fold'],'best_epoch':r['best_epoch'],'gamma_mode':mode,'gamma':gamma.tolist(),'gamma_sum':float(gamma.sum()),'gamma_abs_sum':float(gamma.abs().sum()),'negative_coefficients':int((gamma<0).sum()),'selected_path_weights':weights.tolist(),'path_mode':r['config']['research']['path_mode'],'validation':r['validation'],'checkpoint_matches_result':True})
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'synthetic_smoke':a.smoke,'runs':rows,'interpretation':'Signed weights can change scale as well as propagation mixture. These diagnostics do not establish causation. Use paired controls; no test labels accessed.'}
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
with (T/'audit'/('gamma_'+('smoke_' if a.smoke else '')+stamp+'.json')).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print('Verified and extracted coefficients from',len(rows),'completed checkpoints; synthetic=',a.smoke)
