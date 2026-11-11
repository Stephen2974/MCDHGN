# Candidate v2: multi-metapath logit propagation

This is a prespecified, finite development experiment within the existing authorization. The hard deadline remains **2026-11-12 22:26:23 Asia/Taipei (2026-11-12 14:26:23 UTC)**. This document does not extend the budget or authorize installation, new data, test-label access, changes to original files, or interference with another session.

## Scope and input contract

Use the verified frozen original-project D0 inputs: the original 48 feature columns, original node ordering, original nine cached metapath graphs, and the same five development folds. Check the input SHA256 fingerprints against the task audit before each worker starts. Do not change graph files, labels, splits, raw data, or manifests. `path_mode=ppi` selects the already existing graph at index 0; it creates no new graph or data version.

This experiment is transductive: graph topology and features of all existing nodes are available, while the loss uses only the current fold's training labels. Neither validation nor test labels enter propagation. Validation labels are used solely by the existing checkpoint selection and metric calculation. There is no test evaluation, label propagation, validation-label feature, or new label source.

The implementation is an APPNP/GPR-inspired multi-metapath variant, not a reproduction of either original method. References: APPNP, https://arxiv.org/abs/1810.05997 ; GPR-GNN, https://arxiv.org/abs/2006.07988 . Scores reported by other papers are not directly comparable to this data and split protocol.

## Model held constant

One shared 48 -> 128 -> 2 MLP, using ReLU and dropout after its hidden layer, produces logits Z. For each selected path m, build a sparse incoming-mean operator P_m with P_m[dst,src] equal to edge multiplicity divided by destination indegree. Preserve directed edges, duplicate edges, and existing self loops. Reject zero-indegree graphs rather than silently changing them.

The path output is sum over k=0..K of gamma[k] * P_m^k Z. One gamma vector is shared across paths. Global softmax path weights mix selected path outputs. The word "fixed" refers to hop coefficients only: path weights and the MLP still train.

Use alpha=0.1 and width=128 throughout. Initial PPR coefficients are gamma[k]=alpha*(1-alpha)^k for k<K and gamma[K]=(1-alpha)^K. They sum to one before training. Signed parameters start at these coefficients without constraints. Softmax parameters start at their logarithms, yielding the same initial polynomial up to floating-point rounding. At K=0 the implementation returns the MLP logits directly and freezes unused propagation parameters, so it cannot introduce a trainable scalar gain or a path-averaging rounding difference.

The topology buffers are sparse and excluded from checkpoints. Restore a checkpoint only after reconstructing its operators from the same fingerprinted input graphs. The forward argument named graphs is not used after construction. The returned representation h is the pre-propagation MLP hidden state; logits are the propagated predictions.

## Complete finite matrix

Run the following **eight configurations**, each on **folds 0,1,2,3,4 with seed 42**, for **40 fold jobs** in total. All research dictionaries also set `family=propagation`, `width=128`, and `alpha=0.1`.

1. `C00_MLP`: hops=0, gamma_mode=fixed, path_mode=all.
2. `C01_PPR5_all`: hops=5, gamma_mode=fixed, path_mode=all.
3. `C02_PPR10_all`: hops=10, gamma_mode=fixed, path_mode=all.
4. `C03_signed5_all`: hops=5, gamma_mode=signed, path_mode=all.
5. `C04_signed10_all`: hops=10, gamma_mode=signed, path_mode=all.
6. `C05_softmax5_all`: hops=5, gamma_mode=softmax, path_mode=all.
7. `C06_PPR5_ppi`: hops=5, gamma_mode=fixed, path_mode=ppi.
8. `C07_signed5_ppi`: hops=5, gamma_mode=signed, path_mode=ppi.

The K=5 fixed and signed models are prespecified representatives for PPI comparisons. They are not retrospectively chosen because of their validation scores. Do not replace them with a different hop count after inspecting results. No softmax K=10, further alpha grid, or extra optimizer grid belongs to this matrix.

Use the same optimization protocol for every row: Adam lr=0.003, weight_decay=0.0001, positive class weight=1.904, dropout=0.4, at most 1000 epochs, validation every 10 epochs, patience=20 validation checks, and gradient norm clipping at 1.0. No feature standardization is introduced in this matrix. Preserve the existing runner's deterministic seed rule, seed + 1000*fold, and note residual CUDA nondeterminism. Shared MLP initialization is paired across variants for a given seed/fold. Any departure must be documented before a new, separately named experiment is launched.

These models have a one-hidden-layer shared MLP. They must be compared primarily with C00, not treated as an exact replacement for an earlier, differently sized or deeper raw-feature MLP. All variants use the same loss, validation policy, and checkpoint export logic.

## Prespecified comparisons and limits of attribution

- Propagation: C01 versus C00 asks whether fixed PPR propagation over the nine paths improves this shared MLP. This changes topology use and introduces learned path weights; it does not isolate a single biological mechanism.
- Hop count: C02 versus C01, and C04 versus C03, change K=5 to K=10 within their coefficient family. Truncated-PPR initialization changes with K. Signed models also gain five coefficient parameters. Interpret these as hop-budget comparisons, not pure depth-only causal effects.
- Learned coefficients: C03 versus C01 and C04 versus C02 compare unconstrained learned coefficients with their fixed initialization at matching K. C05 versus C01 compares learned nonnegative normalized coefficients with fixed coefficients. C03 versus C05 compares two different parameterizations and constraints, not only the existence of negative weights.
- Path selection: C06 versus C01, and C07 versus C03, compare PPI alone with all nine fixed paths at matching K and coefficient mode. Path count, available topology, and the number of effective path-weight parameters differ. A benefit cannot be uniquely attributed to non-PPI biological information without additional controls.

Signed gamma is neither normalized nor bounded. Its sum and absolute sum may drift, altering the scale of logits and optimization as well as the propagation filter. Negative coefficients alone are not proof that high-pass filtering caused a gain. Weight decay acts on signed gamma directly, but on the logits that parameterize softmax gamma; regularization is therefore not equivalent across these modes. Keep these differences visible in the interpretation. Do not normalize signed coefficients after training to improve a reported score.

## Metrics, checkpointing, and selection

Keep the existing protocol: select each fold's single checkpoint by validation average precision, then obtain both AP and ROC AUC from that same checkpoint's two-class softmax probabilities. Never select a different checkpoint for AUC. Average the five fold APs and the five fold AUCs separately. The stated target requires **both means strictly greater than 0.9**.

Publish each fold's AP, AUC, best epoch, trained epochs, runtime, peak memory, and status. Also publish five-fold means, sample standard deviations, and paired fold differences for the prespecified comparisons. Five folds are dependent, so do not present them as independent experimental replicates for an ordinary significance test. Do not select a model or learning rate separately for each fold, merge incomplete variants, or choose predictions from different models for a favorable composite result.

Rank complete configurations by mean AP; if the means are exactly tied, use mean AUC, then lower compute cost. The 0.9 target is a separate joint threshold, not a reason to suppress a configuration with higher AP. Results are development/model-selection results. Searching multiple models and epochs on the same validation folds creates selection optimism; additional seeds on the same folds probe training stability but do not eliminate that optimism. Final test evaluation requires separate authorization.

Within remaining time, a later decision record may nominate the selected model and relevant controls for seeds 43 and 44. Those runs are not automatically added by this protocol and must not delay honest reporting of the full seed-42 matrix. No robust improvement claim should rely only on one favorable seed or fold.

## Recommended diagnostic export interface (proposal only)

No model code is changed by this document. In a new, hash-frozen worker version, a label-free helper such as `propagation_diagnostics(model)` may return JSON-serializable values. It must be called on the AP-selected checkpoint, not merely the last training epoch. Suggested fields are:

- schema_version, hops, gamma_mode, alpha_initial, selected_paths, and original_path_count.
- initial_gamma and effective_gamma from `model.coefficients().detach().cpu()`.
- gamma_sum, gamma_abs_sum, gamma_min, gamma_max, negative_coefficient_count, gamma_l2, and gamma_distance_from_initial_l2.
- selected_path_weights from `softmax(model.path_logits)`, expanded to all original paths with zeros for unselected paths, plus path-weight entropy.
- total parameter count, trainable parameter count, effective learned gamma count, and effective learned path-weight count. A one-element PPI path weight is fixed at one; K=0 propagation parameters are inactive.
- Optionally, at a documented fixed feature-only evaluation sample: raw and propagated logit RMS, mean absolute binary logit margin, and their ratios. This helps identify output-scale changes without reading any labels.

For a short diagnostic trajectory, save coefficients, their sum/absolute sum, and path weights at the ordinary validation checkpoints. These records are descriptive and do not create an additional checkpoint-selection criterion. Raw softmax gamma logits alone are not the effective coefficients; log the transformed coefficients too. K=0 beta is a metadata placeholder and does not explain prediction importance. PPI-only beta contains zeros for paths that were not used, and all-path beta is global rather than gene-specific.

## Execution, safety, and immutable evidence

Deploy into a new task-owned code/output version. Do not edit any code imported by active workers. Before launch, freeze the model, worker, scheduler, plan, and protocol hashes; record the unchanged input fingerprints and environment versions. Use independent outputs for every configuration, seed, and fold; refuse existing output directories. Keep all failures, partial outputs, stderr, and attempted configurations.

Use at most three currently available RTX2080Ti devices and at most one task worker per device. Respect foreign GPU users. The existing four-hour per-fold timeout and total deadline remain in force. Only terminate processes positively owned by this task when a prespecified failure, timeout, or budget condition occurs. A missing/nonfinite result or worker failure is a failure, not evidence of low predictive performance. No automatic retry or silent hyperparameter adjustment belongs to this matrix.

Stop launching jobs at the fixed total deadline and enforce the previously authorized deadline rule for owned workers. If the budget cannot finish all five folds of a configuration, mark it incomplete and exclude it from complete-model rankings and target claims. Do not alter the deadline to complete this protocol.

## Static implementation review

The inspected `propagation_models.py` implements the stated incoming direction, duplicate-edge weighting, shared MLP, shared hop coefficients, global path weights, K=0 shortcut, PPI selection, and nonpersistent sparse topology buffers. It accepts no label argument. It does not write or mutate graphs. Its fixed/signed/softmax initializations have matching effective PPR coefficients by construction. The model caches topology only, so current transformed features would still be used if a separately authorized preprocessing experiment were created.

This is a static review, not a claim that runtime tests, remote input checks, or GPU smoke tests passed. Those must be recorded separately before training. The existing unit suite should verify directed duplicate-edge propagation, hand-computed PPR, K=0 equivalence, initialization equivalence, finite gradients, PPI isolation, and rejection of label arguments. External worker fingerprint checks remain necessary: the model validates feature shape but does not itself verify node identities or file hashes.
