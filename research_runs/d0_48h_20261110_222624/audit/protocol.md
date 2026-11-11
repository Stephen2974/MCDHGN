# D0 48-hour development experiment

Authorization: 2026-11-10 22:26:23 Taipei to 2026-11-12 22:26:23 Taipei. Maximum three idle RTX2080Ti, one fold per device. All task-owned workers have a four-hour timeout and are terminated at the deadline. Never stop foreign processes.

Data: frozen legacy original-project D0, 12944 x 48, original nine cached metapaths and five development folds. Audit verified 12 training-input SHA256 fingerprints against the baseline manifest, source TSV features exactly equal graph features, and split integrity. No test labels read. This verifies frozen-file identity, not missing historical source provenance.

Primary metric: mean of five validation APs (sklearn average_precision_score), secondary ROC AUC from the same AP-selected checkpoint. Target requires BOTH means >0.9. Report individual folds and sample SD; validation folds are dependent. Every variant uses all five folds, seed42, same epoch limit/eval/patience/loss. No per-fold model or learning-rate selection. These are development/model-selection results, not unbiased test estimates.

Initial hypotheses: (1) original feature signal bypasses graph smoothing; (2) incoming-mean multi-hop features improve over hop0; (3) raw residual and LayerNorm have separate effects in a 2x2 design. Raw MLP beta is only a placeholder. SIGN is inspired by https://arxiv.org/abs/2004.11198; implemented with frozen metapath incoming means, not a reproduction claim. Raw branch and normalization are separate controls. Additional raw residual parameters require capacity control before attribution.

No new dependencies or datasets, no label propagation, no external labels, no manifest/data changes. Standardization is optional and fitted on each fold's training nodes only; parameters saved. Existing source code is copied and never modified. Active experiment code is hash-frozen; later versions must use new code and outputs. Existing result directories are only read.

Initial validation: 22 existing ablation/summary tests passed in isolated code copy. Audit helper initially assumed comma-separated source despite .csv name; inspection showed TSV, corrected separator, reran exact comparison successfully. This was an audit parser failure before any training, not a data modification.

Literature: MCDHGN https://pmc.ncbi.nlm.nih.gov/articles/PMC11256962/; APPNP https://arxiv.org/abs/1810.05997; GPR-GNN https://arxiv.org/abs/2006.07988; GCNII https://proceedings.mlr.press/v119/chen20v.html. Later stages require complete first-stage results and recorded hypotheses, plus repeats of best and controls on seed43/44 within remaining budget. No guarantee of reaching .9 and no suppression of failures.
