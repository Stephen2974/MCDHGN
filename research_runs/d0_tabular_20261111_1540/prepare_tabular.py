"""Build immutable derived features and preregister before fitting any model."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['OMP_NUM_THREADS'] = '2'
import datetime as dt
import json
from pathlib import Path
import sys
import numpy as np
from tabular_features import feature_views
from tabular_worker import digest, write_json, check_deadline

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent.parent
sys.path.insert(0, str(TASK / 'code'))
import run_ablation as r


def main():
    assert not (TASK / 'plan.json').exists()
    r.dependencies()
    r.torch.set_num_threads(2)
    x, graphs, splits, fp = r.load_data(ROOT)
    audit = json.loads((TASK / 'audit/data_audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'passed' and audit['data_sha256'] == fp
    views = feature_views(x.numpy(), [(s.numpy(), d.numpy()) for g in graphs for s, d in [g.edges(order='eid')]])
    for name, values in views.items():
        np.save(TASK / 'cache' / (name + '.npy'), values)
    arrays = {str(fold) + '_' + key: value.numpy() for fold, split in enumerate(splits)
              for key, value in zip(('train', 'val', 'train_y', 'val_y'), split)}
    np.savez_compressed(TASK / 'cache/splits.npz', **arrays)
    write_json(TASK / 'cache/metadata.json', {
        'data_sha256': fp, 'cache_sha256': {p.name: digest(p) for p in (TASK / 'cache').glob('*')},
        'features': {k: list(v.shape) for k, v in views.items()}, 'test_labels_read': False,
        'feature_construction': 'original directed edge multiplicity; incoming means; no labels'})
    jobs = []
    # Two estimators x four cumulative feature views; complete all five folds.
    for estimator in ('hgb', 'extra_trees'):
        for features in ('raw', 'mean1', 'mean2', 'moments2'):
            experiment = 'T_' + estimator + '_' + features
            params = (dict(loss='log_loss', learning_rate=.05, max_leaf_nodes=15,
                           min_samples_leaf=20, l2_regularization=1.0, early_stopping=False,
                           warm_start=True) if estimator == 'hgb' else
                      dict(n_estimators=500, min_samples_leaf=2, max_features='sqrt', n_jobs=2))
            for fold in range(5):
                config = dict(experiment=experiment, estimator=estimator, features=features,
                              estimator_params=params, seed=42, fold=fold, positive_weight=1.904,
                              epochs=1000, eval_every=10, patience=20, smoke=False)
                jobs.append(dict(id=experiment + '_s42_f' + str(fold), config=config))
    files = list(TASK.glob('*.py')) + list((TASK / 'code').glob('*.py'))
    plan = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                deadline_utc='2026-11-12T14:26:23+00:00', per_fold_timeout_seconds=14400,
                jobs=jobs, cpu_workers=1, cpu_threads=2, gpu_ids=[],
                execution_sha256={p.relative_to(TASK).as_posix(): digest(p) for p in files},
                selection_rule='All five folds, same config; rank complete groups by mean AP; target both AP/AUC > .9.',
                repeat_rule='Best full group and its raw-feature control, seeds43/44, within original budget.',
                paired_controls='mean1-raw; mean2-mean1; moments2-mean2 within estimator; HGB vs ExtraTrees is a separate family comparison.',
                ensemble_rule='Exploratory fixed 0.5 probability mean with C_v2_original_std0_raw0, aligned per fold; no learned weights, no per-fold selection.',
                limitations='Development validation, not unbiased test. HGB has AP-selected iterations; ExtraTrees uses a fixed 500 trees. No equal-compute claim across families.')
    check_deadline(plan)
    write_json(TASK / 'plan.json', plan)
    print(json.dumps({'jobs': len(jobs), 'features': {k: list(v.shape) for k, v in views.items()}}))


if __name__ == '__main__':
    main()
