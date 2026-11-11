# Cross-domain continuation, 2026-11-11

Continue the existing D0 research task within its original authorization, ending
2026-11-12 22:26:23 Asia/Taipei. Remote root G:/LLLM/MCDHGN-main. No data or
manifest changes, dependency installation, test-label access, git commit/push,
or termination of foreign processes. Existing GPU queues remain untouched.

Hypothesis: the nonlinear, sparse, differently scaled omics measurements may
benefit from tabular tree inductive biases. Evidence motivating this experiment:
Grinsztajn et al. https://arxiv.org/abs/2207.08815. This is a transfer hypothesis,
not a claim that their benchmark establishes performance on this dataset.

The installed scikit-learn 1.6.1 provides histogram gradient boosting and
ExtraTrees. Reference API:
https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
Only one CPU worker, with two CPU threads, runs alongside existing GPU work.
All input and executable fingerprints are checked. All artifacts are isolated.

Eight configurations cross two estimators with four nested feature sets:
raw; raw + each path's one-hop mean; add two-hop means; add one-hop standard
deviations. Features use the original directed edges and retain multiplicity.
Derived features use no labels; the transductive graph matches the prior task.
Fit uses only the original fold's training nodes. No fold or label changes.
HGB uses 1000 iterations maximum, validation every 10, patience 20, learning
rate .05, 15 leaves, min leaf 20, L2 1. ExtraTrees uses a fixed 500 trees,
minimum leaf 2, square-root feature subsampling. Both weight positives 1.904.
Different model families have different fitting budgets; report this explicitly.
AP and AUC always use the same selected model's saved validation probabilities.
Complete all five folds for each config; seed42 for exploration. Repeat best
and its raw control at seeds43/44 as the original deadline permits.

Prespecified secondary diagnostic: equal probability blend of each complete
tree configuration with C_v2_original_std0_raw0, paired by original fold and
node ID. No trained blending weights or per-fold model choice. These are
exploratory development scores. Inspect train/validation isolation and record
all failures. Neither this nor any validation search is an unbiased test.

Next independent hypothesis: RankNet-style pairwise logistic ranking loss can
align training with ranking metrics. Source:
https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/
Implement in a new frozen neural copy, with a zero-coefficient CE control,
train-node pairs only, unchanged AP checkpoint selection, and matched seeds.
No claim of AP optimization follows from a pairwise ranking surrogate alone.
