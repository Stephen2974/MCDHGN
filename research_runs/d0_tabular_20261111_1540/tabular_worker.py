"""Two-thread CPU tabular comparison; never reads held-out test labels."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['DGLBACKEND'] = 'pytorch'
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from threadpoolctl import threadpool_limits

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def check_deadline(plan):
    if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(plan['deadline_utc']):
        raise RuntimeError('Authorized experiment budget expired')


def metric_pair(truth, scores):
    if len(truth) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid prediction array')
    return {'ap': float(average_precision_score(truth, scores)),
            'auc': float(roc_auc_score(truth, scores)),
            'loss': float(log_loss(truth, scores, labels=[0, 1]))}


def fit_model(config, train_x, train_y, val_x, val_y, dest, deadline_check=lambda: None):
    """Only train_x/train_y enter fit. Validation selects HGB iteration count."""
    weighted = np.where(train_y == 1, config['positive_weight'], 1.0)
    seed = config['seed'] + 1000 * config['fold']
    params = dict(config['estimator_params'], random_state=seed)
    if config['estimator'] == 'extra_trees':
        model = ExtraTreesClassifier(**params)
        deadline_check()
        model.fit(train_x, train_y, sample_weight=weighted)
        scores = model.predict_proba(val_x)[:, 1]
        best_epoch = config['estimator_params']['n_estimators']
        history = [dict(epoch=best_epoch, **metric_pair(val_y, scores))]
        joblib.dump(model, dest / 'best.joblib')
    elif config['estimator'] == 'hgb':
        model = HistGradientBoostingClassifier(**params)
        best_ap, best_epoch, stale, history = -1.0, 0, 0, []
        for iteration in range(config['eval_every'], config['epochs'] + 1, config['eval_every']):
            deadline_check()
            model.set_params(max_iter=iteration)
            model.fit(train_x, train_y, sample_weight=weighted)
            scores = model.predict_proba(val_x)[:, 1]
            row = dict(epoch=iteration, **metric_pair(val_y, scores))
            history.append(row)
            if row['ap'] > best_ap:
                best_ap, best_epoch, stale = row['ap'], iteration, 0
                joblib.dump(model, dest / 'best.joblib')
            else:
                stale += 1
            if config['patience'] and stale >= config['patience']:
                break
    else:
        raise ValueError('Unknown estimator')
    deadline_check()
    best = joblib.load(dest / 'best.joblib')
    scores = best.predict_proba(val_x)[:, 1]
    metrics = metric_pair(val_y, scores)
    selected = next(h for h in history if h['epoch'] == best_epoch)
    assert abs(selected['ap'] - metrics['ap']) < 1e-12
    assert abs(selected['auc'] - metrics['auc']) < 1e-12
    with (dest / 'history.jsonl').open('x', encoding='utf-8') as f:
        for row in history:
            f.write(json.dumps(row) + '\n')
    return best, scores, metrics, best_epoch, history[-1]['epoch']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    plan = json.loads((TASK / 'plan.json').read_text(encoding='utf-8'))
    check_deadline(plan)
    assert all(digest(TASK / p) == h for p, h in plan['execution_sha256'].items())
    meta = json.loads((TASK / 'cache/metadata.json').read_text(encoding='utf-8'))
    assert all(digest(ROOT / p) == h for p, h in meta['data_sha256'].items()), 'Input data changed'
    assert all(digest(TASK / 'cache' / p) == h for p, h in meta['cache_sha256'].items()), 'Cache changed'
    job = next(j for j in plan['jobs'] if j['id'] == args.job)
    config = dict(job['config'])
    dest = TASK / 'outputs' / config['experiment'] / ('seed' + str(config['seed'])) / ('fold' + str(config['fold']))
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest / 'config.json', {'config': config, 'data_sha256': meta['data_sha256']})
    x = np.load(TASK / 'cache' / (config['features'] + '.npy'), mmap_mode='r')
    with np.load(TASK / 'cache/splits.npz') as z:
        fold = str(config['fold'])
        train, val, train_y, val_y = [z[fold + '_' + k] for k in ('train', 'val', 'train_y', 'val_y')]
    assert len(np.intersect1d(train, val)) == 0
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        best, scores, metrics, best_epoch, last_epoch = fit_model(
            config, x[train], train_y, x[val], val_y, dest, lambda: check_deadline(plan))
    np.savez_compressed(dest / 'validation_predictions.npz', node_ids=val, labels=val_y, probabilities=scores)
    result = {'config': config, 'data_sha256': meta['data_sha256'],
              'code_sha256': plan['execution_sha256'], 'validation': metrics,
              'best_epoch': best_epoch, 'epochs_trained': last_epoch,
              'train_seconds': time.perf_counter() - started, 'feature_count': x.shape[1],
              'checkpoint_sha256': digest(dest / 'best.joblib'),
              'environment': {'python': sys.version, 'sklearn': sklearn.__version__, 'cpu_threads': 2},
              'test_labels_read': False, 'fitting_scope': 'fold training nodes only',
              'selection': 'AP-selected iteration' if config['estimator'] == 'hgb' else 'fixed tree count'}
    write_json(dest / 'result.json', result)
    print(json.dumps({'job': args.job, 'validation': metrics, 'seconds': result['train_seconds']}), flush=True)


if __name__ == '__main__':
    main()
