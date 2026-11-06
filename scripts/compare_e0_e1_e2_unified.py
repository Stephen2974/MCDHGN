"""Strict paired comparison of unified-protocol E0/E1/E2 validation runs.

Recomputes AP/AUC from validation_predictions.npz aligned by node_id.
Does not mix probabilities across folds for the primary AP.
Focus-gene list is diagnostic only and never used for training or selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[1]

# Diagnostic-only list from iec.py; never enters training weights or selection.
FOCUS_GENES = (
    "ATN1", "BLTP1", "COL4A5", "DNAH11", "DPF3", "EXOSC7", "FBP2", "FEM1B",
    "FRAS1", "HELZ", "HR", "LRP5", "MBD5", "MTMR6", "NLGN1", "NUP155",
    "PCDHA9", "PLXNA4", "ROBO4", "SLITRK1", "SVIL", "TAF1A", "WDHD1",
)

EXPECTED_CODE = {
    "model.py": "0c667afd9e9ffd00bf2f04290d3b880c35b303cb0ca19e24230097f9be3c18c0",
    "fusion.py": "a8cad7be137a9705052d9c88c3fd269e1058a28825e85fb4a63987ffb06e8ee5",
    "run_ablation.py": "909802329d37513858d03391628e8cb520c9e6d50f3f28147d2fc0c1dccdea0c",
    "ablation_config.py": "6f5e81a470d86048d61a57ab6ada8c8eba11c4e2aa2f9c86136ea72c1b560280",
}

TRAIN_KEYS = (
    "epochs", "eval_every", "patience", "lr", "weight_decay", "positive_weight",
    "dropout", "hidden_size", "heads", "fusion_dim", "fusion_chunk_size", "protocol",
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value) -> None:
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_id_to_symbol(data_root: Path) -> dict[int, str]:
    with (data_root / "data/step1/gene_toids.pkl").open("rb") as fh:
        gene_to_id = pickle.load(fh)
    return {int(v): str(k) for k, v in gene_to_id.items()}


def recompute_from_npz(path: Path) -> dict:
    data = np.load(path)
    node_ids = np.asarray(data["node_ids"]).reshape(-1)
    labels = np.asarray(data["labels"]).reshape(-1)
    scores = np.asarray(data["probabilities"]).reshape(-1)
    order = np.argsort(node_ids, kind="stable")
    node_ids, labels, scores = node_ids[order], labels[order], scores[order]
    if len(set(node_ids.tolist())) != len(node_ids):
        raise ValueError(f"Duplicate node_ids in {path}")
    if set(labels.tolist()) - {0, 1}:
        raise ValueError(f"Non-binary labels in {path}")
    ap = float(metrics.average_precision_score(labels, scores))
    auc = float(metrics.roc_auc_score(labels, scores))
    out = {"ap": ap, "auc": auc, "n": int(len(labels)), "n_pos": int((labels == 1).sum())}
    for k in (50, 100):
        count = min(k, len(labels))
        top = np.argsort(-scores, kind="stable")[:count]
        out[f"precision_at_{k}"] = float(labels[top].mean())
        out[f"recall_at_{k}"] = float(labels[top].sum() / labels.sum())
    return out, node_ids, labels, scores


def collect_run(exp: str, seed: int, fold: int, run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    res_path = run_dir / "result.json"
    pred_path = run_dir / "validation_predictions.npz"
    best_path = run_dir / "best.pt"
    for p in (cfg_path, res_path, pred_path, best_path):
        if not p.exists():
            raise FileNotFoundError(p)
    cfg_wrap = json.loads(cfg_path.read_text(encoding="utf-8"))
    result = json.loads(res_path.read_text(encoding="utf-8"))
    config = result["config"]
    recomputed, node_ids, labels, scores = recompute_from_npz(pred_path)
    reported = result["validation"]
    for key in ("ap", "auc"):
        if abs(recomputed[key] - reported[key]) > 1e-10:
            raise ValueError(
                f"{run_dir}: recomputed {key}={recomputed[key]} != reported {reported[key]}"
            )
    ranks = scores.argsort(kind="stable")[::-1].argsort() + 1  # 1-based dense by stable sort of -score
    # Stable rank with ties: average? Task asks to record tie rule. Use competition rank via
    # argsort(-score, stable) then position; equal scores get different ranks by original order.
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores), dtype=np.int32)
    rank[order] = np.arange(1, len(scores) + 1)
    return {
        "experiment": exp,
        "seed": seed,
        "fold": fold,
        "run_dir": str(run_dir.resolve()),
        "best_epoch": result["best_epoch"],
        "epochs_trained": result["epochs_trained"],
        "ap": recomputed["ap"],
        "auc": recomputed["auc"],
        "precision_at_50": reported.get("precision_at_50", recomputed["precision_at_50"]),
        "precision_at_100": reported.get("precision_at_100", recomputed["precision_at_100"]),
        "train_seconds": result["train_seconds"],
        "code_sha256": config["code_sha256"],
        "data_sha256": result["data_sha256"],
        "architecture": config["architecture"],
        "train_protocol": {k: config[k] for k in TRAIN_KEYS},
        "parameters": result["parameters"],
        "node_ids": node_ids,
        "labels": labels,
        "scores": scores,
        "ranks": rank,
        "config_file": cfg_wrap,
    }


def discover_runs(e0_root: Path, e1_root: Path, e2_root: Path) -> dict:
    roots = {"E0": e0_root, "E1": e1_root, "E2": e2_root}
    runs = {}
    for exp, root in roots.items():
        for seed in (42, 43, 44):
            for fold in range(5):
                run_dir = root / exp / f"seed{seed}" / f"fold{fold}"
                if exp == "E0" and not run_dir.exists():
                    # queue writes E0/seedX/foldY under e0_root which already includes E0 parent? 
                    # e0_root is results/e0_unified_... and contains E0/seed...
                    alt = root / f"seed{seed}" / f"fold{fold}"
                    run_dir = run_dir if run_dir.exists() else alt
                runs[(exp, seed, fold)] = collect_run(exp, seed, fold, run_dir)
    return runs


def mean_sd(values):
    values = list(values)
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else None,
        "n": len(values),
        "values": values,
    }


def pair_stats(deltas):
    wins = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if d == 0)
    losses = sum(1 for d in deltas if d < 0)
    return {
        "mean_delta": float(statistics.mean(deltas)),
        "sd_delta": float(statistics.stdev(deltas)) if len(deltas) > 1 else None,
        "n": len(deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "worst_regression": float(min(deltas)),
        "best_gain": float(max(deltas)),
        "mean_delta_pp": float(statistics.mean(deltas)) * 100.0,
        "deltas": deltas,
    }


def ranking_pair_stats(scores_a, labels, scores_b, tie_rule: str):
    """Compare positive/negative ordering between models A(baseline) and B(candidate).

    Tie rule: a positive is 'strictly outranked' by a negative only when score_neg > score_pos.
    Equal scores are recorded as ties and do not count as outranked or repaired.
    """
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    # For each positive, count negatives with strictly higher score
    def invert_counts(scores):
        # positives outranked by negatives
        pos_outranked = []
        for i in pos:
            pos_outranked.append(int(np.sum(scores[neg] > scores[i])))
        # negatives that outrank positives
        neg_outrank = []
        for j in neg:
            neg_outrank.append(int(np.sum(scores[pos] < scores[j])))
        return np.asarray(pos_outranked), np.asarray(neg_outrank)

    a_pos, a_neg = invert_counts(scores_a)
    b_pos, b_neg = invert_counts(scores_b)

    # Pair flips over all (pos,neg) pairs
    sa_p = scores_a[pos][:, None]
    sa_n = scores_a[neg][None, :]
    sb_p = scores_b[pos][:, None]
    sb_n = scores_b[neg][None, :]
    a_correct = sa_p > sa_n
    a_wrong = sa_p < sa_n
    a_tie = sa_p == sa_n
    b_correct = sb_p > sb_n
    b_wrong = sb_p < sb_n
    b_tie = sb_p == sb_n
    repaired = int(np.sum(a_wrong & b_correct))
    newly_wrong = int(np.sum(a_correct & b_wrong))
    stayed_correct = int(np.sum(a_correct & b_correct))
    stayed_wrong = int(np.sum(a_wrong & b_wrong))
    return {
        "tie_rule": tie_rule,
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "n_pairs": int(len(pos) * len(neg)),
        "baseline_pos_mean_neg_above": float(a_pos.mean()) if len(a_pos) else None,
        "candidate_pos_mean_neg_above": float(b_pos.mean()) if len(b_pos) else None,
        "baseline_neg_mean_pos_below": float(a_neg.mean()) if len(a_neg) else None,
        "candidate_neg_mean_pos_below": float(b_neg.mean()) if len(b_neg) else None,
        "pairs_repaired": repaired,
        "pairs_newly_wrong": newly_wrong,
        "pairs_stayed_correct": stayed_correct,
        "pairs_stayed_wrong": stayed_wrong,
        "baseline_tie_pairs": int(a_tie.sum()),
        "candidate_tie_pairs": int(b_tie.sum()),
        "net_pair_gain": repaired - newly_wrong,
    }


def build_gene_rows(runs, id_to_symbol):
    rows = []
    for seed in (42, 43, 44):
        for fold in range(5):
            e0 = runs[("E0", seed, fold)]
            e1 = runs[("E1", seed, fold)]
            e2 = runs[("E2", seed, fold)]
            if not (np.array_equal(e0["node_ids"], e1["node_ids"]) and np.array_equal(e0["node_ids"], e2["node_ids"])):
                raise ValueError(f"node_id mismatch seed={seed} fold={fold}")
            if not (np.array_equal(e0["labels"], e1["labels"]) and np.array_equal(e0["labels"], e2["labels"])):
                raise ValueError(f"label mismatch seed={seed} fold={fold}")
            n_neg = int((e0["labels"] == 0).sum())
            n_pos = int((e0["labels"] == 1).sum())
            for i, nid in enumerate(e0["node_ids"]):
                label = int(e0["labels"][i])
                s0, s1, s2 = float(e0["scores"][i]), float(e1["scores"][i]), float(e2["scores"][i])
                r0, r1, r2 = int(e0["ranks"][i]), int(e1["ranks"][i]), int(e2["ranks"][i])
                # inversion counts for this node
                if label == 1:
                    inv0 = int(np.sum(e0["scores"][e0["labels"] == 0] > s0))
                    inv1 = int(np.sum(e1["scores"][e1["labels"] == 0] > s1))
                    inv2 = int(np.sum(e2["scores"][e2["labels"] == 0] > s2))
                else:
                    inv0 = int(np.sum(e0["scores"][e0["labels"] == 1] < s0))
                    inv1 = int(np.sum(e1["scores"][e1["labels"] == 1] < s1))
                    inv2 = int(np.sum(e2["scores"][e2["labels"] == 1] < s2))
                # repair / new-error vs E0 and vs E1 for this gene's pairwise role is aggregate;
                # per-gene flags: for positives, whether inversion count improved/worsened.
                def flag(base_inv, new_inv, lab):
                    if lab not in (0, 1):
                        return "na"
                    if new_inv < base_inv:
                        return "repaired"
                    if new_inv > base_inv:
                        return "new_error"
                    return "unchanged"

                symbol = id_to_symbol.get(int(nid), "")
                rows.append({
                    "node_id": int(nid),
                    "gene_symbol": symbol,
                    "label": label,
                    "seed": seed,
                    "fold": fold,
                    "is_focus_gene": int(symbol in FOCUS_GENES),
                    "score_E0": s0,
                    "score_E1": s1,
                    "score_E2": s2,
                    "rank_E0": r0,
                    "rank_E1": r1,
                    "rank_E2": r2,
                    "n_opposite_strictly_above_E0": inv0,
                    "n_opposite_strictly_above_E1": inv1,
                    "n_opposite_strictly_above_E2": inv2,
                    "E1_vs_E0_error_flag": flag(inv0, inv1, label),
                    "E2_vs_E1_error_flag": flag(inv1, inv2, label),
                    "E2_vs_E0_error_flag": flag(inv0, inv2, label),
                    "n_pos_in_fold": n_pos,
                    "n_neg_in_fold": n_neg,
                    "tie_rule": "strict_inequality_only; equal_scores_not_counted_as_outrank",
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0-root", type=Path, required=True)
    parser.add_argument("--e1-root", type=Path, default=ROOT / "results/p0_legacy_D0")
    parser.add_argument("--e2-root", type=Path, default=ROOT / "results/p0_legacy_D0")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-notes", type=Path, default=None,
                        help="optional prewritten protocol_audit.md to copy/append")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    id_to_symbol = load_id_to_symbol(args.data_root.resolve())
    runs = discover_runs(args.e0_root.resolve(), args.e1_root.resolve(), args.e2_root.resolve())

    # Protocol consistency checks
    protocol_issues = []
    data_hashes = {json.dumps(r["data_sha256"], sort_keys=True) for r in runs.values()}
    code_hashes = {json.dumps(r["code_sha256"], sort_keys=True) for r in runs.values()}
    train_protos = {json.dumps(r["train_protocol"], sort_keys=True) for r in runs.values()}
    if len(data_hashes) != 1:
        protocol_issues.append(f"multiple data hashes: {len(data_hashes)}")
    if len(code_hashes) != 1:
        protocol_issues.append(f"multiple code hashes: {len(code_hashes)}")
    if len(train_protos) != 1:
        protocol_issues.append(f"multiple train protocols: {len(train_protos)}")
    sample_code = next(iter(runs.values()))["code_sha256"]
    if sample_code != EXPECTED_CODE:
        protocol_issues.append("code hash != sealed E1/E2 expected hashes")

    expected_arch = {
        "E0": {"use_gatv2": False, "semantic_mode": "global", "fusion_mode": "none", "fusion_alpha": 0.6},
        "E1": {"use_gatv2": True, "semantic_mode": "global", "fusion_mode": "none", "fusion_alpha": 0.6},
        "E2": {"use_gatv2": True, "semantic_mode": "node", "fusion_mode": "none", "fusion_alpha": 0.6},
    }
    for (exp, seed, fold), r in runs.items():
        if r["architecture"] != expected_arch[exp]:
            protocol_issues.append(f"arch mismatch {exp} s{seed} f{fold}: {r['architecture']}")

    # per_run
    per_run = []
    for exp in ("E0", "E1", "E2"):
        for seed in (42, 43, 44):
            for fold in range(5):
                r = runs[(exp, seed, fold)]
                per_run.append({
                    "experiment": exp,
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": r["best_epoch"],
                    "epochs_trained": r["epochs_trained"],
                    "AP": f"{r['ap']:.10f}",
                    "AUC": f"{r['auc']:.10f}",
                    "precision_at_50": f"{r['precision_at_50']:.10f}",
                    "precision_at_100": f"{r['precision_at_100']:.10f}",
                    "train_seconds": f"{r['train_seconds']:.3f}",
                    "source_dir": r["run_dir"],
                    "code_sha256_model_py": r["code_sha256"]["model.py"],
                    "code_sha256_run_ablation_py": r["code_sha256"]["run_ablation.py"],
                    "data_sha256_graph": r["data_sha256"]["data/network/hetero/new_9nodes_graph.bin"],
                    "data_sha256_splits": r["data_sha256"]["Intermediate/label_set/train_batch.pkl"],
                })
    write_tsv(out / "per_run.tsv", per_run, list(per_run[0].keys()))

    # per_seed
    per_seed = []
    seed_means = defaultdict(dict)
    for exp in ("E0", "E1", "E2"):
        for seed in (42, 43, 44):
            folds = [runs[(exp, seed, f)] for f in range(5)]
            row = {
                "experiment": exp,
                "seed": seed,
                "AP_mean": statistics.mean(r["ap"] for r in folds),
                "AUC_mean": statistics.mean(r["auc"] for r in folds),
                "precision_at_50_mean": statistics.mean(r["precision_at_50"] for r in folds),
                "precision_at_100_mean": statistics.mean(r["precision_at_100"] for r in folds),
                "best_epoch_mean": statistics.mean(r["best_epoch"] for r in folds),
                "train_seconds_sum": sum(r["train_seconds"] for r in folds),
            }
            seed_means[exp][seed] = row
            per_seed.append({k: (f"{v:.10f}" if isinstance(v, float) else v) for k, v in row.items()})
    write_tsv(out / "per_seed.tsv", per_seed, list(per_seed[0].keys()))

    # across-seed summary
    across = {}
    for exp in ("E0", "E1", "E2"):
        across[exp] = {}
        for metric in ("AP_mean", "AUC_mean", "precision_at_50_mean", "precision_at_100_mean"):
            vals = [seed_means[exp][s][metric] for s in (42, 43, 44)]
            across[exp][metric] = mean_sd(vals)

    # paired deltas
    contrasts = [("E1", "E0"), ("E2", "E1"), ("E2", "E0")]
    paired_rows = []
    paired_summary = {}
    for cand, base in contrasts:
        name = f"{cand}-{base}"
        paired_summary[name] = {}
        for metric, key in (("AP", "ap"), ("AUC", "auc"), ("precision_at_50", "precision_at_50"), ("precision_at_100", "precision_at_100")):
            deltas = []
            per_seed_delta = {}
            for seed in (42, 43, 44):
                seed_d = []
                for fold in range(5):
                    d = runs[(cand, seed, fold)][key] - runs[(base, seed, fold)][key]
                    deltas.append(d)
                    seed_d.append(d)
                    paired_rows.append({
                        "contrast": name,
                        "metric": metric,
                        "seed": seed,
                        "fold": fold,
                        "baseline": runs[(base, seed, fold)][key],
                        "candidate": runs[(cand, seed, fold)][key],
                        "delta": d,
                        "delta_pp": d * 100.0 if metric == "AP" else "",
                    })
                per_seed_delta[seed] = float(statistics.mean(seed_d))
            st = pair_stats(deltas)
            st["per_seed_mean_delta"] = per_seed_delta
            st["direction_consistent"] = all(v > 0 for v in per_seed_delta.values()) or all(v < 0 for v in per_seed_delta.values())
            paired_summary[name][metric] = st
    write_tsv(out / "paired_deltas.tsv", paired_rows, list(paired_rows[0].keys()))

    # gene predictions + ranking changes
    gene_rows = build_gene_rows(runs, id_to_symbol)
    write_tsv(out / "gene_predictions_comparison.tsv", gene_rows, list(gene_rows[0].keys()))

    ranking_rows = []
    ranking_summary = {}
    tie_rule = "strict_inequality_only; equal_scores_count_as_ties_not_flips"
    for cand, base in contrasts:
        name = f"{cand}_vs_{base}"
        nets = []
        for seed in (42, 43, 44):
            for fold in range(5):
                a = runs[(base, seed, fold)]
                b = runs[(cand, seed, fold)]
                st = ranking_pair_stats(a["scores"], a["labels"], b["scores"], tie_rule)
                st.update({"contrast": name, "seed": seed, "fold": fold})
                ranking_rows.append(st)
                nets.append(st["net_pair_gain"])
        ranking_summary[name] = {
            "mean_net_pair_gain": float(statistics.mean(nets)),
            "sd_net_pair_gain": float(statistics.stdev(nets)) if len(nets) > 1 else None,
            "folds_net_positive": sum(1 for n in nets if n > 0),
            "folds_net_negative": sum(1 for n in nets if n < 0),
            "folds_net_zero": sum(1 for n in nets if n == 0),
            "tie_rule": tie_rule,
        }
    # flatten ranking_rows for tsv
    ranking_tsv = []
    for r in ranking_rows:
        ranking_tsv.append({k: r[k] for k in [
            "contrast", "seed", "fold", "n_pos", "n_neg", "n_pairs",
            "baseline_pos_mean_neg_above", "candidate_pos_mean_neg_above",
            "baseline_neg_mean_pos_below", "candidate_neg_mean_pos_below",
            "pairs_repaired", "pairs_newly_wrong", "pairs_stayed_correct", "pairs_stayed_wrong",
            "baseline_tie_pairs", "candidate_tie_pairs", "net_pair_gain", "tie_rule",
        ]})
    write_tsv(out / "ranking_changes.tsv", ranking_tsv, list(ranking_tsv[0].keys()))

    # fold-level contribution: is E2-E0 driven by one bad fold?
    e2e0_ap = []
    for seed in (42, 43, 44):
        for fold in range(5):
            e2e0_ap.append({
                "seed": seed, "fold": fold,
                "E0_AP": runs[("E0", seed, fold)]["ap"],
                "E2_AP": runs[("E2", seed, fold)]["ap"],
                "delta": runs[("E2", seed, fold)]["ap"] - runs[("E0", seed, fold)]["ap"],
            })
    # leave-one-fold-out mean delta across remaining 14? Better: per-fold mean delta across seeds
    fold_mean_delta = {}
    for fold in range(5):
        fold_mean_delta[fold] = float(statistics.mean(
            runs[("E2", s, fold)]["ap"] - runs[("E0", s, fold)]["ap"] for s in (42, 43, 44)
        ))

    manifest = {
        "n_runs": 45,
        "experiments": ["E0", "E1", "E2"],
        "seeds": [42, 43, 44],
        "folds": [0, 1, 2, 3, 4],
        "protocol_issues": protocol_issues,
        "code_sha256": sample_code,
        "data_sha256": next(iter(runs.values()))["data_sha256"],
        "train_protocol": next(iter(runs.values()))["train_protocol"],
        "architectures": expected_arch,
        "sources": {
            "E0": str(args.e0_root.resolve()),
            "E1": str(args.e1_root.resolve()),
            "E2": str(args.e2_root.resolve()),
        },
        "runs": [
            {
                "experiment": exp, "seed": seed, "fold": fold,
                "source_dir": runs[(exp, seed, fold)]["run_dir"],
                "best_epoch": runs[(exp, seed, fold)]["best_epoch"],
                "AP": runs[(exp, seed, fold)]["ap"],
                "AUC": runs[(exp, seed, fold)]["auc"],
                "parameters": runs[(exp, seed, fold)]["parameters"],
            }
            for exp in ("E0", "E1", "E2") for seed in (42, 43, 44) for fold in range(5)
        ],
    }
    write_json(out / "run_manifest.json", manifest)

    # Answers A-F scaffolding
    e2e0 = paired_summary["E2-E0"]
    e2e1 = paired_summary["E2-E1"]
    e1e0 = paired_summary["E1-E0"]

    def seed_dirs(metric_key="ap"):
        dirs = {}
        for seed in (42, 43, 44):
            d = statistics.mean(
                runs[("E2", seed, f)][metric_key] - runs[("E0", seed, f)][metric_key] for f in range(5)
            )
            dirs[seed] = d
        return dirs

    ap_seed_dirs = seed_dirs("ap")
    auc_seed_dirs = seed_dirs("auc")

    # Check tradeoffs: AP up but AUC/Top100/pairs down
    tradeoffs = []
    for seed in (42, 43, 44):
        for fold in range(5):
            dap = runs[("E2", seed, fold)]["ap"] - runs[("E0", seed, fold)]["ap"]
            dauc = runs[("E2", seed, fold)]["auc"] - runs[("E0", seed, fold)]["auc"]
            d100 = runs[("E2", seed, fold)]["precision_at_100"] - runs[("E0", seed, fold)]["precision_at_100"]
            if dap > 0 and (dauc < 0 or d100 < 0):
                tradeoffs.append({"seed": seed, "fold": fold, "d_ap": dap, "d_auc": dauc, "d_p100": d100})

    summary = {
        "across_seed_means": across,
        "paired_summary": {
            k: {m: {kk: vv for kk, vv in st.items() if kk != "deltas"} for m, st in v.items()}
            for k, v in paired_summary.items()
        },
        "ranking_summary": ranking_summary,
        "fold_mean_delta_E2_minus_E0_AP": fold_mean_delta,
        "E2_minus_E0_AP_by_seed": ap_seed_dirs,
        "E2_minus_E0_AUC_by_seed": auc_seed_dirs,
        "ap_up_auc_or_p100_down_folds": tradeoffs,
        "protocol_issues": protocol_issues,
        "focus_genes_diagnostic_only": list(FOCUS_GENES),
        "answers": {
            "A_E2_vs_E0_AP_mean_delta": e2e0["AP"]["mean_delta"],
            "A_E2_vs_E0_AP_mean_delta_pp": e2e0["AP"]["mean_delta_pp"],
            "A_E2_vs_E0_AUC_mean_delta": e2e0["AUC"]["mean_delta"],
            "B_AP_seed_direction_consistent": e2e0["AP"]["direction_consistent"],
            "B_AUC_seed_direction_consistent": e2e0["AUC"]["direction_consistent"],
            "C_fold_mean_deltas_AP": fold_mean_delta,
            "D_E2_vs_E1_ranking": ranking_summary["E2_vs_E1"],
            "E_tradeoff_folds": tradeoffs,
            "F_needs_heldout_test": True,
        },
    }
    write_json(out / "summary.json", summary)

    # report.md
    lines = []
    lines.append("# E0/E1/E2 unified-protocol comparison\n")
    lines.append("## Verdict\n")
    lines.append(
        f"- E2−E0 AP mean delta (15 paired folds): {e2e0['AP']['mean_delta']:.6f} "
        f"({e2e0['AP']['mean_delta_pp']:.4f} percentage points); "
        f"wins/ties/losses={e2e0['AP']['wins']}/{e2e0['AP']['ties']}/{e2e0['AP']['losses']}; "
        f"worst regression={e2e0['AP']['worst_regression']:.6f}.\n"
    )
    lines.append(
        f"- E2−E0 AUC mean delta: {e2e0['AUC']['mean_delta']:.6f}; "
        f"W/T/L={e2e0['AUC']['wins']}/{e2e0['AUC']['ties']}/{e2e0['AUC']['losses']}.\n"
    )
    lines.append(
        f"- E2−E1 AP mean delta: {e2e1['AP']['mean_delta']:.6f} "
        f"({e2e1['AP']['mean_delta_pp']:.4f} pp); "
        f"E1−E0 AP mean delta: {e1e0['AP']['mean_delta']:.6f} ({e1e0['AP']['mean_delta_pp']:.4f} pp).\n"
    )
    lines.append("\n## Across-seed means (mean of 5-fold means; sample SD over 3 seeds)\n")
    for exp in ("E0", "E1", "E2"):
        ap = across[exp]["AP_mean"]
        auc = across[exp]["AUC_mean"]
        lines.append(
            f"- {exp}: AP={ap['mean']:.6f} (SD={ap['sd']:.6f}); "
            f"AUC={auc['mean']:.6f} (SD={auc['sd']:.6f})\n"
        )
    lines.append("\n## Seed directions (E2−E0 fold-mean)\n")
    for seed in (42, 43, 44):
        lines.append(f"- seed {seed}: ΔAP={ap_seed_dirs[seed]:.6f}; ΔAUC={auc_seed_dirs[seed]:.6f}\n")
    lines.append("\n## Fold mean ΔAP (E2−E0, averaged over 3 seeds)\n")
    for fold, d in fold_mean_delta.items():
        lines.append(f"- fold {fold}: {d:.6f} ({d*100:.4f} pp)\n")
    lines.append("\n## Ranking pair changes (strict inequality; ties excluded from flips)\n")
    for name, st in ranking_summary.items():
        lines.append(
            f"- {name}: mean net pair gain={st['mean_net_pair_gain']:.1f}; "
            f"folds +/0/− = {st['folds_net_positive']}/{st['folds_net_zero']}/{st['folds_net_negative']}\n"
        )
    lines.append("\n## Answers A–F\n")
    lines.append(f"A. E2 vs E0 AP Δ={e2e0['AP']['mean_delta']:.6f} ({e2e0['AP']['mean_delta_pp']:.4f} pp); "
                 f"AUC Δ={e2e0['AUC']['mean_delta']:.6f}.\n")
    lines.append(f"B. AP seed directions consistent? {e2e0['AP']['direction_consistent']}; "
                 f"AUC? {e2e0['AUC']['direction_consistent']}.\n")
    lines.append(f"C. Fold mean ΔAP values: {fold_mean_delta}. "
                 "Interpret whether one fold dominates.\n")
    lines.append(f"D. E2 vs E1 net pair-gain folds +/0/− = "
                 f"{ranking_summary['E2_vs_E1']['folds_net_positive']}/"
                 f"{ranking_summary['E2_vs_E1']['folds_net_zero']}/"
                 f"{ranking_summary['E2_vs_E1']['folds_net_negative']}; "
                 f"mean net={ranking_summary['E2_vs_E1']['mean_net_pair_gain']:.1f}.\n")
    lines.append(f"E. Folds with AP↑ but AUC↓ or P@100↓ vs E0: {len(tradeoffs)} "
                 f"(see summary.json).\n")
    lines.append("F. Held-out test set was not used; all selection used validation AP only. "
                 "Independent test confirmation is still required before claiming generalization.\n")
    lines.append("\n## Limits\n")
    lines.append("- Only 3 seeds; do not overclaim statistical significance.\n")
    lines.append("- 15 folds share overlapping training sets; not 15 independent samples.\n")
    lines.append("- Transductive full-graph setting: test-node features/structure may participate in message passing; test labels are unused in training/selection.\n")
    lines.append("- Focus-gene list is post-hoc diagnostic only.\n")
    if protocol_issues:
        lines.append("\n## Protocol issues\n")
        for issue in protocol_issues:
            lines.append(f"- {issue}\n")
    write_text(out / "report.md", "".join(lines))

    if args.audit_notes and args.audit_notes.exists():
        # keep caller-provided audit as protocol_audit.md if not already written
        dest = out / "protocol_audit.md"
        if not dest.exists():
            dest.write_text(args.audit_notes.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps({
        "output": str(out),
        "protocol_issues": protocol_issues,
        "E2_minus_E0_AP": e2e0["AP"]["mean_delta"],
        "E2_minus_E0_AUC": e2e0["AUC"]["mean_delta"],
    }, indent=2))


if __name__ == "__main__":
    main()
