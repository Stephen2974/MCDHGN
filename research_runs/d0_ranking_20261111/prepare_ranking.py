"""Stage a new code copy. Never alters a running experiment's files."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

TASK=Path(__file__).resolve().parent
SOURCE=TASK.parent/'d0_48h_20261110_222624'


def main():
    assert not (TASK/'plan.json').exists()
    old=json.loads((SOURCE/'plan.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()==h for p,h in old['execution_sha256'].items())
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(old['deadline_utc'])
    for name in ('worker.py','scheduler.py','status.py','verify_results.py'):
        shutil.copy2(SOURCE/name,TASK/name)
    runner=TASK/'code/run_ablation.py'
    original=runner.read_text(encoding='utf-8')
    old_line='            loss = F.cross_entropy(logits[train_ids], train_y, weight=class_weight)'
    new_line='            loss = training_objective(logits[train_ids], train_y, class_weight, config, epoch)'
    assert original.count(old_line)==1
    changed=original.replace(old_line,new_line)
    changed=changed.replace('from ablation_config import EXPERIMENTS, METAPATHS',
                            'from ablation_config import EXPERIMENTS, METAPATHS\nfrom ranking_losses import training_objective')
    runner.write_text(changed,encoding='utf-8')
    jobs=[]
    for baseline in ('A00_GAT_reference_lr01','C_v2_original_std0_raw0'):
        base=next(j['config'] for j in old['jobs'] if j['config']['experiment']==baseline)
        for coefficient in (0,.1,1.):
            name='R_'+('GAT' if baseline.startswith('A00') else 'GATv2node')+'_rank'+str(coefficient).replace('.','p')
            for fold in range(5):
                c=copy.deepcopy(base)
                c.update(experiment=name,seed=42,fold=fold,ranking=dict(coefficient=coefficient,pairs=8192))
                jobs.append(dict(id=name+'_s42_f'+str(fold),config=c))
    files=list((TASK/'code').glob('*.py'))+[TASK/n for n in ('worker.py','scheduler.py')]
    plan=dict(old,jobs=jobs,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
              execution_sha256={p.relative_to(TASK).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              next_phase='Run only after original and propagation queues end successfully; preserve original deadline.',
              hypothesis='RankNet-style pairwise logistic auxiliary loss; 8192 uniform train-only pairs; CE control exact at coefficient zero.',
              selection_rule='Complete five folds per cell; same AP checkpoint for AP/AUC. Compare coefficient .1 and 1 to zero within architecture.',
              limitations='Pairwise surrogate does not optimize AP exactly. Development validation only. No test labels.')
    with (TASK/'plan.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2)
    print('Staged',len(jobs),'jobs; no GPU training started')


if __name__=='__main__':main()
