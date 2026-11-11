import json,statistics,datetime
from pathlib import Path
T=Path(__file__).resolve().parent
def rows(name):
 return [json.loads((T/'outputs'/name/'seed42'/('fold'+str(i))/'result.json').read_text(encoding='utf-8')) for i in range(5)]
pairs=[('SIGN_h1_vs_h0','A_SIGN_h1_raw0','A_SIGN_h0_raw0'),('SIGN_h2_vs_h1','A_SIGN_h2_raw0','A_SIGN_h1_raw0'),('SIGN_raw_branch','A_SIGN_h2_raw1','A_SIGN_h2_raw0'),('SIGN_global_vs_node','A_SIGN_h2_raw1_global','A_SIGN_h2_raw1'),('raw_standardization','A03_raw_zscore_lr003','A01_raw_lr003')]
for family in ['GATglobal','GATv2node']:
 for norm in [0,1]:pairs.append((family+'_raw_at_norm'+str(norm),'B_'+family+'_raw1_norm'+str(norm),'B_'+family+'_raw0_norm'+str(norm)))
 for raw in [0,1]:pairs.append((family+'_norm_at_raw'+str(raw),'B_'+family+'_raw'+str(raw)+'_norm1','B_'+family+'_raw'+str(raw)+'_norm0'))
out=[]
for label,a,b in pairs:
 ar,br=rows(a),rows(b)
 assert all(x['data_sha256']==y['data_sha256'] for x,y in zip(ar,br))
 delta={metric:[x['validation'][metric]-y['validation'][metric] for x,y in zip(ar,br)] for metric in ['ap','auc']}
 out.append({'comparison':label,'candidate':a,'control':b,'delta':delta,'mean_delta':{k:statistics.mean(v) for k,v in delta.items()},'ap_wins':sum(v>0 for v in delta['ap']),'candidate_train_seconds':sum(r['train_seconds'] for r in ar),'control_train_seconds':sum(r['train_seconds'] for r in br)})
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':42,'folds':list(range(5)),'interpretation':'Exploratory paired development-fold differences, not independent samples or significance tests. Architectural changes can alter parameter count; report capacity caveat. No test labels accessed.','comparisons':out}
target=T/'audit/phase1_paired_audit.json'
with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
for r in out:print(r['comparison'],r['mean_delta'],'AP wins',r['ap_wins'])
