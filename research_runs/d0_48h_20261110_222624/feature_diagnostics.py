import json,pickle,datetime
from pathlib import Path
import pandas as pd
import numpy as np
T=Path(__file__).resolve().parent;R=T.parent.parent
with (R/'data/step1/gene_toids.pkl').open('rb') as f:m=pickle.load(f)
frame=pd.read_csv(R/'data/biological_features.csv',sep='\t',index_col=0)
x=frame.loc[[g for g,i in sorted(m.items(),key=lambda z:z[1])]].to_numpy(dtype=np.float32)
with (R/'Intermediate/label_set/train_batch.pkl').open('rb') as f:splits=pickle.load(f)
folds=[]
for fold,(train,val,ty,vy) in enumerate(splits):
 a=x[np.asarray(train)];std=a.std(0)
 folds.append({'fold':fold,'train_only':True,'std_min':float(std.min()),'std_max':float(std.max()),'max_abs':float(np.abs(a).max()),'zero_variance_columns':int((std==0).sum()),'blocks':[{'columns':list(frame.columns[start:start+16]),'quantiles':np.quantile(a[:,start:start+16],[0,.01,.5,.99,1]).tolist(),'negative_fraction':float((a[:,start:start+16]<0).mean())} for start in [0,16,32]]})
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'folds':folds,'test_labels_read':False,'changes_made_to_data':False}
with (T/'audit/feature_diagnostics.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps(folds[0],indent=2))
