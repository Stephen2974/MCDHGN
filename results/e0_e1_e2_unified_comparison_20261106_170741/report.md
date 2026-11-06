# E0/E1/E2 unified-protocol comparison
## Verdict
- E2−E0 AP mean delta (15 paired folds): 0.024304 (2.4304 percentage points); wins/ties/losses=12/0/3; worst regression=-0.024062.
- E2−E0 AUC mean delta: 0.014610; W/T/L=12/0/3.
- E2−E1 AP mean delta: 0.000425 (0.0425 pp); E1−E0 AP mean delta: 0.023879 (2.3879 pp).

## Across-seed means (mean of 5-fold means; sample SD over 3 seeds)
- E0: AP=0.787094 (SD=0.016057); AUC=0.859827 (SD=0.010969)
- E1: AP=0.810973 (SD=0.001638); AUC=0.876172 (SD=0.001378)
- E2: AP=0.811399 (SD=0.002718); AUC=0.874437 (SD=0.001269)

## Seed directions (E2−E0 fold-mean)
- seed 42: ΔAP=0.013981; ΔAUC=0.006004
- seed 43: ΔAP=0.040592; ΔAUC=0.025245
- seed 44: ΔAP=0.018340; ΔAUC=0.012583

## Fold mean ΔAP (E2−E0, averaged over 3 seeds)
- fold 0: 0.005477 (0.5477 pp)
- fold 1: 0.029393 (2.9393 pp)
- fold 2: 0.018705 (1.8705 pp)
- fold 3: 0.065073 (6.5073 pp)
- fold 4: 0.002875 (0.2875 pp)

## Ranking pair changes (strict inequality; ties excluded from flips)
- E1_vs_E0: mean net pair gain=5009.1; folds +/0/− = 15/0/0
- E2_vs_E1: mean net pair gain=-528.3; folds +/0/− = 7/0/8
- E2_vs_E0: mean net pair gain=4480.8; folds +/0/− = 12/0/3

## Answers A–F
A. E2 vs E0 AP Δ=0.024304 (2.4304 pp); AUC Δ=0.014610.
B. AP seed directions consistent? True; AUC? True.
C. Fold mean ΔAP values: {0: 0.005476980192899809, 1: 0.02939268864827871, 2: 0.0187052738215836, 3: 0.06507263632376614, 4: 0.0028746945070218244}. Interpret whether one fold dominates.
D. E2 vs E1 net pair-gain folds +/0/− = 7/0/8; mean net=-528.3.
E. Folds with AP↑ but AUC↓ or P@100↓ vs E0: 2 (see summary.json).
F. Held-out test set was not used; all selection used validation AP only. Independent test confirmation is still required before claiming generalization.

## Limits
- Only 3 seeds; do not overclaim statistical significance.
- 15 folds share overlapping training sets; not 15 independent samples.
- Transductive full-graph setting: test-node features/structure may participate in message passing; test labels are unused in training/selection.
- Focus-gene list is post-hoc diagnostic only.
