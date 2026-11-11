"""Recompute saved scores, verify checkpoint predictions, and paired ablations."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import sys
import joblib
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def score(y,p):
    return {'ap':float(average_precision_score(y,p)),'auc':float(roc_auc_score(y,p))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='d0_tabular_20261111_1540');parser.add_argument('--complete',action='store_true')
    args=parser.parse_args()
    project=Path(r'G:\LLLM\MCDHGN-main');root=project/'research_runs'/args.run
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    meta=json.loads((root/'cache/metadata.json').read_text(encoding='utf-8'))
    assert all(digest(project/p)==h for p,h in meta['data_sha256'].items())
    assert all(digest(root/p)==h for p,h in plan['execution_sha256'].items())
    assert all(digest(root/'cache'/p)==h for p,h in meta['cache_sha256'].items())
    rows=[];groups={};ensembles=[]
    with threadpool_limits(limits=2),np.load(root/'cache/splits.npz') as splits:
        for job in plan['jobs']:
            c=job['config'];fold=c['fold'];key=str(fold)
            dest=root/'outputs'/c['experiment']/('seed'+str(c['seed']))/('fold'+key)
            if not (dest/'result.json').exists():
                if args.complete:raise RuntimeError('Missing result: '+job['id'])
                continue
            result=json.loads((dest/'result.json').read_text(encoding='utf-8'))
            assert result['config']==c and result['data_sha256']==meta['data_sha256']
            assert result['code_sha256']==plan['execution_sha256']
            assert digest(dest/'best.joblib')==result['checkpoint_sha256']
            with np.load(dest/'validation_predictions.npz') as pred:
                ids,labels,probs=[pred[k] for k in ('node_ids','labels','probabilities')]
            np.testing.assert_array_equal(ids,splits[key+'_val'])
            np.testing.assert_array_equal(labels,splits[key+'_val_y'])
            assert np.isfinite(probs).all() and np.all((probs>=0)&(probs<=1))
            recomputed=score(labels,probs)
            for metric in ('ap','auc'):assert abs(recomputed[metric]-result['validation'][metric])<1e-12
            model=joblib.load(dest/'best.joblib')
            if hasattr(model,'n_jobs'):model.n_jobs=2
            x=np.load(root/'cache'/(c['features']+'.npy'),mmap_mode='r')
            np.testing.assert_allclose(model.predict_proba(x[ids])[:,1],probs,rtol=0,atol=1e-14)
            del model,x
            row=dict(experiment=c['experiment'],seed=c['seed'],fold=fold,**recomputed,
                     checkpoint_verified=True,predictions_sha256=digest(dest/'validation_predictions.npz'))
            rows.append(row);groups.setdefault((c['experiment'],c['seed']),[]).append(row)
            # Frozen equal-weight diagnostic, paired with the same baseline fold.
            bp=project/'research_runs/d0_48h_20261110_222624/outputs/C_v2_original_std0_raw0/seed42'/('fold'+key)/'validation_predictions.npz'
            if c['seed']==42 and bp.exists():
                with np.load(bp) as baseline:
                    np.testing.assert_array_equal(ids,baseline['node_ids'])
                    np.testing.assert_array_equal(labels,baseline['labels'])
                    blend=.5*probs+.5*baseline['probabilities']
                ensembles.append(dict(experiment=c['experiment'],seed=42,fold=fold,**score(labels,blend),
                                      baseline_predictions_sha256=digest(bp)))
    summary=[];paired=[]
    for (experiment,seed),values in groups.items():
        folds=sorted(v['fold'] for v in values)
        if folds!=list(range(5)):continue
        summary.append(dict(experiment=experiment,seed=seed,n=5,
                            **{m:statistics.mean(v[m] for v in values) for m in ('ap','auc')},
                            **{m+'_sd':statistics.stdev(v[m] for v in values) for m in ('ap','auc')}))
    for estimator in ('hgb','extra_trees'):
        for base,target in [('raw','mean1'),('mean1','mean2'),('mean2','moments2')]:
            for seed in sorted(set(r['seed'] for r in rows)):
                a=groups.get(('T_'+estimator+'_'+base,seed),[]);b=groups.get(('T_'+estimator+'_'+target,seed),[])
                if len(a)!=5 or len(b)!=5:continue
                aa={v['fold']:v for v in a};bb={v['fold']:v for v in b}
                delta=[dict(fold=f,**{m:bb[f][m]-aa[f][m] for m in ('ap','auc')}) for f in range(5)]
                paired.append(dict(estimator=estimator,comparison=target+' - '+base,seed=seed,fold_deltas=delta,
                                   mean_delta={m:statistics.mean(d[m] for d in delta) for m in ('ap','auc')}))
    blend_summary=[]
    for experiment in sorted(set(e['experiment'] for e in ensembles)):
        values=[e for e in ensembles if e['experiment']==experiment]
        if sorted(e['fold'] for e in values)!=list(range(5)):continue
        blend_summary.append(dict(experiment=experiment,n=5,**{m:statistics.mean(e[m] for e in values) for m in ('ap','auc')}))
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),verified=len(rows),planned=len(plan['jobs']),
                rows=rows,summary=sorted(summary,key=lambda r:-r['ap']),paired=paired,
                fixed_blend=sorted(blend_summary,key=lambda r:-r['ap']),
                caveat='Exploratory development validation; paired folds are dependent; no significance or test claim.',test_labels_read=False)
    path=root/'audit'/('verified_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({'path':str(path),'verified':len(rows),'planned':len(plan['jobs']),
                      'summary':report['summary'],'paired':paired,'fixed_blend':report['fixed_blend']},indent=2))


if __name__=='__main__':main()
