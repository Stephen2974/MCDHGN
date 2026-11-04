"""Cross-learning-rate comparison. Does not modify stored results or the summarizer."""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


ROOT = Path(r"G:\LLLM\MCDHGN-main")
OUT = Path(sys.argv[1]).resolve()
OLD = {
    "E1": ROOT / "results" / "p0_legacy_D0" / "E1" / "seed42",
    "E3": ROOT / "results" / "d0_E3_seed42_20261103_153158" / "train" / "E3" / "seed42",
}
METRICS = ("ap", "auc", "accuracy", "macro_f1")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recompute(path):
    stored = np.load(path)
    labels = stored["labels"].astype(int)
    probabilities = stored["probabilities"].astype(float)
    logits = stored["logits"].astype(float)
    node_ids = stored["node_ids"].astype(int)
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted)
    expected = expected[:, 1] / expected.sum(axis=1)
    predicted = logits.argmax(axis=1)
    return {
        "node_ids": node_ids,
        "labels": labels,
        "ap": float(average_precision_score(labels, probabilities)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "probability_max_abs_diff": float(np.max(np.abs(probabilities - expected))),
    }


def splits():
    import torch
    path = ROOT / "Intermediate" / "label_set" / "train_batch.pkl"
    with path.open("rb") as stream:
        rows = pickle.load(stream)
    parsed = []
    for row in rows:
        parsed.append(tuple(torch.as_tensor(value, dtype=torch.long).reshape(-1).numpy() for value in row))
    return parsed


def load_run(directory):
    row = read_json(directory / "result.json")
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text(encoding="utf-8").splitlines() if line]
    metrics = recompute(directory / "validation_predictions.npz")
    best_rows = [item for item in history if item["ap"] == max(item["ap"] for item in history)]
    return row, history, metrics, best_rows[0]


def check_pair(experiment, fold, audit):
    new_dir = OUT / "train" / experiment / "seed42" / ("fold" + str(fold))
    old_dir = OLD[experiment] / ("fold" + str(fold))
    new, new_history, new_metrics, new_best = load_run(new_dir)
    old, old_history, old_metrics, old_best = load_run(old_dir)
    if new["config"]["experiment"] != experiment or old["config"]["experiment"] != experiment:
        raise RuntimeError("Experiment name mismatch")
    if new["config"]["seed"] != 42 or old["config"]["seed"] != 42 or new["config"]["fold"] != fold or old["config"]["fold"] != fold:
        raise RuntimeError("Seed or fold mismatch")
    if new["config"]["smoke"] or old["config"]["smoke"] or old["config"]["lr"] != 0.01 or new["config"]["lr"] != 0.003:
        raise RuntimeError("Unexpected lr or smoke setting")
    new_rest = {key: value for key, value in new["config"].items() if key != "lr"}
    old_rest = {key: value for key, value in old["config"].items() if key != "lr"}
    if new_rest != old_rest or new["data_sha256"] != old["data_sha256"] or new["environment"] != old["environment"]:
        raise RuntimeError("More than lr differs for %s fold %s" % (experiment, fold))
    train_ids, val_ids, _, val_y = splits()[fold]
    if set(np.asarray(new_metrics["node_ids"], dtype=np.int64)).intersection(np.asarray(train_ids, dtype=np.int64).tolist()):
        raise RuntimeError("Validation IDs overlap training IDs")
    if not np.array_equal(np.asarray(new_metrics["node_ids"], dtype=np.int64), np.asarray(val_ids, dtype=np.int64)) or not np.array_equal(np.asarray(new_metrics["labels"], dtype=np.int64), np.asarray(val_y, dtype=np.int64)):
        raise RuntimeError("Recovered validation IDs or labels differ from the stored split")
    if new_best["epoch"] != new["best_epoch"] or old_best["epoch"] != old["best_epoch"]:
        raise RuntimeError("History best epoch differs from result.json")
    for label, row, metrics in (("new", new, new_metrics), ("old", old, old_metrics)):
        diffs = {name: abs(metrics[name] - row["validation"][name]) for name in METRICS}
        audit.append({
            "experiment": experiment, "fold": fold, "which": label,
            "best_epoch": row["best_epoch"], "epochs_trained": row["epochs_trained"],
            "metric_abs_diff": diffs,
            "probability_max_abs_diff": metrics["probability_max_abs_diff"],
            "history_best_ap": new_best["ap"] if label == "new" else old_best["ap"],
            "restored_ap": metrics["ap"],
        })
    return old, new, old_history, new_history


def delta_block(pairs):
    values = {}
    for metric in METRICS:
        deltas = [new[metric] - old[metric] for _, old, new in pairs]
        values[metric] = {
            "mean_delta_pp": 100 * float(np.mean(deltas)),
            "wins": int(sum(delta > 0 for delta in deltas)),
            "ties": int(sum(delta == 0 for delta in deltas)),
            "pairs": [{"fold": fold, "delta_pp": 100 * delta} for (fold, _, _), delta in zip(pairs, deltas)],
        }
    return values


def late_drop(row, history):
    return row["validation"]["ap"] - history[-1]["ap"] >= 0.05


def main():
    fold_splits = splits()
    if len(fold_splits) != 5:
        raise RuntimeError("Expected five stored folds")
    audit = []
    runs = {}
    histories = {}
    for experiment in ("E1", "E3"):
        for fold in range(5):
            old, new, old_history, new_history = check_pair(experiment, fold, audit)
            runs[(experiment, 0.01, fold)] = old
            runs[(experiment, 0.003, fold)] = new
            histories[(experiment, 0.01, fold)] = old_history
            histories[(experiment, 0.003, fold)] = new_history
    e1_lr = [(fold, runs[("E1", 0.01, fold)]["validation"], runs[("E1", 0.003, fold)]["validation"]) for fold in range(5)]
    e3_lr = [(fold, runs[("E3", 0.01, fold)]["validation"], runs[("E3", 0.003, fold)]["validation"]) for fold in range(5)]
    within_new = [(fold, runs[("E1", 0.003, fold)]["validation"], runs[("E3", 0.003, fold)]["validation"]) for fold in range(5)]
    within_old = [(fold, runs[("E1", 0.01, fold)]["validation"], runs[("E3", 0.01, fold)]["validation"]) for fold in range(5)]
    interaction_pairs = []
    for fold in range(5):
        new_gap = {metric: runs[("E3", 0.003, fold)]["validation"][metric] - runs[("E1", 0.003, fold)]["validation"][metric] for metric in METRICS}
        old_gap = {metric: runs[("E3", 0.01, fold)]["validation"][metric] - runs[("E1", 0.01, fold)]["validation"][metric] for metric in METRICS}
        interaction_pairs.append((fold, old_gap, new_gap))
    contrasts = {
        "E1_lr003_minus_E1_lr001": delta_block(e1_lr),
        "E3_lr003_minus_E3_lr001": delta_block(e3_lr),
        "E3_minus_E1_at_lr003": delta_block(within_new),
        "interaction_lr003_gap_minus_lr001_gap": delta_block(interaction_pairs),
    }
    records = []
    for experiment in ("E1", "E3"):
        for lr in (0.01, 0.003):
            for fold in range(5):
                row = runs[(experiment, lr, fold)]
                history = histories[(experiment, lr, fold)]
                peak = max(row["peak_allocated_bytes"].values()) if row["peak_allocated_bytes"] else None
                records.append({
                    "experiment": experiment, "lr": lr, "seed": 42, "fold": fold,
                    "best_epoch": row["best_epoch"], "epochs_trained": row["epochs_trained"],
                    "hit_epoch_cap": row["epochs_trained"] >= 1000,
                    "late_drop_0_05": late_drop(row, history),
                    "best_minus_last_ap": row["validation"]["ap"] - history[-1]["ap"],
                    "peak_allocated_bytes": peak, "train_seconds": row["train_seconds"],
                    **{metric: row["validation"][metric] for metric in METRICS},
                })
    if len(records) != 20:
        raise RuntimeError("Expected 20 formal runs")
    destination = OUT / "lr_comparison"
    destination.mkdir(parents=True)
    (destination / "summary.json").write_text(json.dumps({
        "contrasts": contrasts, "recompute_audit": audit, "runs": records,
        "note": "One base seed and validation model selection only. No multi-seed or independent-test claim.",
    }, indent=2), encoding="utf-8")
    with (destination / "paired_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "Learning-rate comparison for E1 and E3, base seed 42, five validation folds.",
        "Differences are percentage points. This is one seed used for screening, not a multi-seed or independent-test result.",
        "",
    ]
    titles = (
        ("A. E1(lr=0.003) - E1(lr=0.01)", "E1_lr003_minus_E1_lr001"),
        ("B. E3(lr=0.003) - E3(lr=0.01)", "E3_lr003_minus_E3_lr001"),
        ("C. E3(lr=0.003) - E1(lr=0.003)", "E3_minus_E1_at_lr003"),
        ("D. [E3-E1 at 0.003] - [E3-E1 at 0.01]", "interaction_lr003_gap_minus_lr001_gap"),
    )
    for title, key in titles:
        lines.append(title)
        for metric in METRICS:
            block = contrasts[key][metric]
            lines.append("  %s: %+.4f percentage points; wins %d/5" % (metric, block["mean_delta_pp"], block["wins"]))
        for pair in contrasts[key]["ap"]["pairs"]:
            lines.append("  fold %s AP: %+.4f" % (pair["fold"], pair["delta_pp"]))
        lines.append("")
    lines.append("fold3")
    for experiment in ("E1", "E3"):
        old_ap = runs[(experiment, 0.01, 3)]["validation"]["ap"]
        new_ap = runs[(experiment, 0.003, 3)]["validation"]["ap"]
        lines.append("%s fold3 AP lr0.003-lr0.01: %+.4f percentage points" % (experiment, 100 * (new_ap - old_ap)))
    gap_old = runs[("E3", 0.01, 3)]["validation"]["ap"] - runs[("E1", 0.01, 3)]["validation"]["ap"]
    gap_new = runs[("E3", 0.003, 3)]["validation"]["ap"] - runs[("E1", 0.003, 3)]["validation"]["ap"]
    lines.append("fold3 E3-E1 at lr0.01: %+.4f percentage points" % (100 * gap_old))
    lines.append("fold3 E3-E1 at lr0.003: %+.4f percentage points" % (100 * gap_new))
    lines.append("")
    drops_old = sum(item["late_drop_0_05"] for item in records if item["lr"] == 0.01)
    drops_new = sum(item["late_drop_0_05"] for item in records if item["lr"] == 0.003)
    caps_old = sum(item["hit_epoch_cap"] for item in records if item["lr"] == 0.01)
    caps_new = sum(item["hit_epoch_cap"] for item in records if item["lr"] == 0.003)
    lines.append("Runs whose last logged AP is at least 0.05 below the selected checkpoint: lr0.01 %d/10, lr0.003 %d/10" % (drops_old, drops_new))
    lines.append("Runs that reached the 1000-epoch cap: lr0.01 %d/10, lr0.003 %d/10" % (caps_old, caps_new))
    lines.append("Mean best epoch: lr0.01 %.1f, lr0.003 %.1f" % (
        np.mean([item["best_epoch"] for item in records if item["lr"] == 0.01]),
        np.mean([item["best_epoch"] for item in records if item["lr"] == 0.003])))
    lines.append("")
    for item in records:
        if item["lr"] == 0.003:
            lines.append("%s fold %s: best_epoch=%s epochs_trained=%s seconds=%.1f peak_allocated_bytes=%s best_minus_last_ap=%+.4f" % (
                item["experiment"], item["fold"], item["best_epoch"], item["epochs_trained"], item["train_seconds"],
                item["peak_allocated_bytes"], item["best_minus_last_ap"]))
    (destination / "comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "recompute_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
