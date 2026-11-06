# Protocol audit: unified E0 vs legacy E1/E2

Date: 2026-11-06  
Machine: lab-gpu (`G:\LLLM\MCDHGN-main`)  
E0 output: `results/e0_unified_20261106_170741`  
E1/E2 source: `results/p0_legacy_D0/{E1,E2}`

## Environment

- No `AGENTS.md` found under the project or `G:\LLLM`.
- Git branch `main`; local modifications retained (D1/D2 logs, unverified D1 experiment dirs, queue script). Nothing was reverted.
- Concurrent process at launch: `data_processing.d2.pipeline download` only (CPU). No training jobs were stopped.
- GPUs 0/1/2 were idle (0 / 0 / ~145 MiB display processes) before E0 launch.

## Completeness of existing E1/E2

All 30 runs present (`seeds∈{42,43,44}` × `folds∈{0..4}`) with:

- `result.json`, `config.json`, `best.pt`, `validation_predictions.npz`

No missing artifacts.

## Code / data hash consistency

All 30 E1/E2 runs share one code fingerprint and one data fingerprint. Current working-tree hashes **match** sealed E1/E2 hashes exactly:

| File | SHA256 |
|------|--------|
| model.py | `0c667afd9e9ffd00bf2f04290d3b880c35b303cb0ca19e24230097f9be3c18c0` |
| fusion.py | `a8cad7be137a9705052d9c88c3fd269e1058a28825e85fb4a63987ffb06e8ee5` |
| run_ablation.py | `909802329d37513858d03391628e8cb520c9e6d50f3f28147d2fc0c1dccdea0c` |
| ablation_config.py | `6f5e81a470d86048d61a57ab6ada8c8eba11c4e2aa2f9c86136ea72c1b560280` |

Data artifacts (`new_9nodes_graph.bin`, `gene_toids.pkl`, `train_batch.pkl`, `Intermediate/blocks/{0..8}gs.bin`) also match all 30 runs. **No isolated code snapshot was required**; E0 runs against the verified current tree.

## Architecture (only intended differences)

| Exp | use_gatv2 | semantic_mode | fusion_mode |
|-----|-----------|---------------|-------------|
| E0 | False (GATConv) | global | none |
| E1 | True (GATv2) | global | none |
| E2 | True (GATv2) | node | none |

Runtime check: E0 model modules include `GATConv` and `SemanticAttention(mode=global)`; no `GATv2`/`MemEfficientGATv2`. `ViewInteraction(mode=none)` returns identity (`fusion.py`). IEC is not invoked by `run_ablation.py`.

## Training / evaluation protocol (verified from saved configs)

Identical across E1/E2 (and reused for E0 CLI defaults, which match saved values):

- `epochs=1000`, `eval_every=10`, `patience=20`
- `lr=0.01`, `weight_decay=1e-4`, `positive_weight=1.904`
- `hidden_size=256`, `heads=4`, `dropout=0.4`
- `fusion_dim=64`, `fusion_chunk_size=1024`
- Adam; grad clip norm 1.0
- Fold seed: `seed + 1000*fold`
- Positive score: `softmax[:,1]`; AP via `sklearn.average_precision_score`
- Single best checkpoint by validation AP; all reported metrics from that checkpoint
- Protocol tag: `softmax-ap-checkpoint-v1`
- No `--test-run` in this campaign

## Split audit (actual IDs)

Development set size 5820; held-out test 646; **dev ∩ test = ∅**.

Per fold: `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`.  
Five validation folds partition the development set (sizes 1164×5).  
Labels consistent across folds for each node; gene mapping contiguous 0..12943 (12944 genes).

## Transductive setting (explicit)

Training uses the full gene graph and all node features for message passing. Held-out **test labels** are not read during training or model selection. Test-node features and graph structure may still participate in neighborhood aggregation. Do **not** claim test nodes are structurally invisible.

## Non-interference

- Existing `results/p0_legacy_D0` not overwritten.
- E0 written only under `results/e0_unified_20261106_170741`.
- No other model experiments auto-started.
- D2 download left running.

## Limits

- Three seeds only; overlapping training folds → not 15 i.i.d. samples for naive t-tests.
- Focus-gene list (IEC false-positive diagnostic set) is post-hoc only.
- Independent test-set confirmation is out of scope for this task.
