"""Read-only comparison of D1 E2/E4 seed42 lr=0.01 vs lr=0.003."""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(r"G:\LLLM\MCDHGN-main\results")
OLD = {
    "E2": ROOT / "d1_unverified_E2_seed42_20261108",
    "E4": ROOT / "D1_unverified_E4_seed42_20261108_104318",
}
NEW = ROOT / "D1_unverified_lr003_E2_E4_seed42_20261108_212833"
PATHS = ["gg", "ga-ag", "gb-bg", "gc-cg", "gd-dg", "ge-eg", "gf-fg", "gh-hg", "gi-ig"]
HP = [
    "epochs", "eval_every", "patience", "weight_decay", "positive_weight", "dropout",
    "hidden_size", "heads", "fusion_dim", "fusion_chunk_size", "smoke", "protocol",
    "seed", "architecture", "code_sha256",
]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def history(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_dir(base, exp, fold):
    return base / exp / "seed42" / f"fold{fold}"


def pstdev(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    return float(values.std(ddof=1))


def fmt(value, digits=4):
    if value is None:
        return "null"
    return f"{value:.{digits}f}"


def prediction_bias(path):
    data = np.load(path)
    labels = data["labels"].astype(float)
    probs = data["probabilities"].astype(float)
    prevalence = float(labels.mean())
    return {
        "n": int(labels.size),
        "prevalence": prevalence,
        "mean_probability": float(probs.mean()),
        "pred_positive_rate": float((probs > 0.5).mean()),
        "min_probability": float(probs.min()),
        "max_probability": float(probs.max()),
    }


def series_stats(rows):
    aps = np.array([row["ap"] for row in rows], dtype=float)
    diffs = np.diff(aps) if len(aps) > 1 else np.array([0.0])
    return {
        "n_checks": int(len(aps)),
        "ap_std": float(aps.std(ddof=1)) if len(aps) > 1 else None,
        "ap_min": float(aps.min()),
        "ap_max": float(aps.max()),
        "max_abs_step": float(np.max(np.abs(diffs))),
        "steps_ge_0.05": int(np.sum(np.abs(diffs) >= 0.05)),
        "steps_ge_0.10": int(np.sum(np.abs(diffs) >= 0.10)),
    }


def main():
    lines = []
    problems = []
    results = []
    for exp in ("E2", "E4"):
        for fold in range(5):
            new_path = run_dir(NEW, exp, fold) / "result.json"
            old_path = run_dir(OLD[exp], exp, fold) / "result.json"
            if not new_path.is_file() or not old_path.is_file():
                problems.append(f"missing result {exp} fold{fold}")
                continue
            new = load_json(new_path)
            old = load_json(old_path)
            if new["config"]["lr"] != 0.003 or old["config"]["lr"] != 0.01:
                problems.append(f"unexpected lr {exp} fold{fold}")
            for key in HP:
                if new["config"].get(key) != old["config"].get(key):
                    problems.append(f"config {key} differs {exp} fold{fold}")
            if new["config"]["experiment"] != exp or new["config"]["fold"] != fold:
                problems.append(f"identity {exp} fold{fold}")
            if new["data_sha256"] != old["data_sha256"]:
                problems.append(f"data hash differs {exp} fold{fold}")
            if list(new["config"]) != list(old["config"]):
                problems.append(f"config keys differ {exp} fold{fold}: {list(new['config'])} vs {list(old['config'])}")
            test_file = run_dir(NEW, exp, fold) / "test_result.json"
            if test_file.exists():
                problems.append(f"test result present {exp} fold{fold}")
            new_hist = history(run_dir(NEW, exp, fold) / "history.jsonl")
            old_hist = history(run_dir(OLD[exp], exp, fold) / "history.jsonl")
            results.append({
                "exp": exp, "fold": fold, "new": new, "old": old,
                "new_hist": new_hist, "old_hist": old_hist,
            })

    lines.append(f"runs={len(results)} problems={len(problems)}")
    for item in problems:
        lines.append("PROBLEM " + item)

    for exp in ("E2", "E4"):
        group = [row for row in results if row["exp"] == exp]
        lines.append("")
        lines.append(f"## {exp} best-checkpoint validation")
        lines.append("fold old_ap new_ap d_ap old_auc new_auc d_auc old_f1 new_f1 d_f1 old_best new_best old_epochs new_epochs old_sec new_sec")
        old_aps, new_aps = [], []
        improved = 0
        for row in group:
            o, n = row["old"]["validation"], row["new"]["validation"]
            d_ap = n["ap"] - o["ap"]
            if d_ap > 0:
                improved += 1
            old_aps.append(o["ap"])
            new_aps.append(n["ap"])
            lines.append(
                f"{row['fold']} {fmt(o['ap'])} {fmt(n['ap'])} {fmt(d_ap)} "
                f"{fmt(o['auc'])} {fmt(n['auc'])} {fmt(n['auc']-o['auc'])} "
                f"{fmt(o['macro_f1'])} {fmt(n['macro_f1'])} {fmt(n['macro_f1']-o['macro_f1'])} "
                f"{row['old']['best_epoch']} {row['new']['best_epoch']} "
                f"{row['old']['epochs_trained']} {row['new']['epochs_trained']} "
                f"{row['old']['train_seconds']:.1f} {row['new']['train_seconds']:.1f}"
            )
        lines.append(
            f"mean_ap old={fmt(float(np.mean(old_aps)))} new={fmt(float(np.mean(new_aps)))} "
            f"delta={fmt(float(np.mean(new_aps)-np.mean(old_aps)))} improved_folds={improved}/5 "
            f"sample_std old={fmt(pstdev(old_aps))} new={fmt(pstdev(new_aps))}"
        )
        lines.append("threshold_mean_ap_gain_0.003=" + ("yes" if float(np.mean(new_aps)-np.mean(old_aps)) >= 0.003 else "no"))

        lines.append(f"## {exp} end-of-training validation (last history row, not best checkpoint)")
        lines.append("fold old_ep new_ep old_ap new_ap d_ap old_auc new_auc old_f1 new_f1 old_val_loss new_val_loss old_train_loss new_train_loss")
        old_final, new_final = [], []
        for row in group:
            oh, nh = row["old_hist"][-1], row["new_hist"][-1]
            old_final.append(oh["ap"])
            new_final.append(nh["ap"])
            lines.append(
                f"{row['fold']} {oh['epoch']} {nh['epoch']} {fmt(oh['ap'])} {fmt(nh['ap'])} {fmt(nh['ap']-oh['ap'])} "
                f"{fmt(oh['auc'])} {fmt(nh['auc'])} {fmt(oh['macro_f1'])} {fmt(nh['macro_f1'])} "
                f"{fmt(oh['loss'])} {fmt(nh['loss'])} {fmt(oh['train_loss'])} {fmt(nh['train_loss'])}"
            )
        lines.append(
            f"final_mean_ap old={fmt(float(np.mean(old_final)))} new={fmt(float(np.mean(new_final)))} "
            f"delta={fmt(float(np.mean(new_final)-np.mean(old_final)))} "
            f"sample_std old={fmt(pstdev(old_final))} new={fmt(pstdev(new_final))}"
        )

    lines.append("")
    lines.append("## budget")
    for row in results:
        trained = row["new"]["epochs_trained"]
        best = row["new"]["best_epoch"]
        hist = row["new_hist"]
        late = [item["ap"] for item in hist if item["epoch"] >= 800]
        rising_late = len(late) >= 2 and max(late) > late[0] + 1e-6 and best >= 800 and trained == 1000
        flag = "可能受训练预算限制" if trained == 1000 else "stopped_before_cap"
        lines.append(
            f"{row['exp']} fold{row['fold']} epochs={trained} best={best} {flag} "
            f"last_ap={fmt(hist[-1]['ap'])} best_ap={fmt(row['new']['validation']['ap'])}"
        )
        if rising_late:
            lines.append(f"  late_ap_from_epoch800 max={fmt(max(late))} first={fmt(late[0])}")

    lines.append("")
    lines.append("## diagnostic details")
    wanted = {("E2", 4), ("E4", 0), ("E4", 4)}
    for row in results:
        if (row["exp"], row["fold"]) not in wanted:
            continue
        lines.append(f"### {row['exp']} fold{row['fold']}")
        for label, res, hist in (("lr0.01", row["old"], row["old_hist"]), ("lr0.003", row["new"], row["new_hist"])):
            best_rows = [item for item in hist if item["epoch"] == res["best_epoch"]]
            best_train = best_rows[-1]["train_loss"] if best_rows else None
            gg = res["weight_diagnostics"]["mean_by_path"][0]
            ent = res["weight_diagnostics"]["mean_entropy"]
            bias = prediction_bias(run_dir(NEW if label == "lr0.003" else OLD[row["exp"]], row["exp"], row["fold"]) / "validation_predictions.npz")
            stats = series_stats(hist)
            final = hist[-1]
            lines.append(
                f"{label} best_ep={res['best_epoch']} trained={res['epochs_trained']} "
                f"best_ap={fmt(res['validation']['ap'])} best_auc={fmt(res['validation']['auc'])} "
                f"best_f1={fmt(res['validation']['macro_f1'])} best_val_loss={fmt(res['validation']['loss'])} "
                f"best_train_loss={fmt(best_train)} final_ep={final['epoch']} final_ap={fmt(final['ap'])} "
                f"final_auc={fmt(final['auc'])} final_f1={fmt(final['macro_f1'])} final_val_loss={fmt(final['loss'])} "
                f"final_train_loss={fmt(final['train_loss'])} gg={fmt(gg)} entropy={fmt(ent)} "
                f"prev={fmt(bias['prevalence'])} mean_p={fmt(bias['mean_probability'])} "
                f"pred_pos={fmt(bias['pred_positive_rate'])} p_range={fmt(bias['min_probability'])}-{fmt(bias['max_probability'])} "
                f"ap_std={fmt(stats['ap_std'])} ap_range={fmt(stats['ap_min'])}-{fmt(stats['ap_max'])} "
                f"max_step={fmt(stats['max_abs_step'])} steps>=0.05={stats['steps_ge_0.05']} steps>=0.10={stats['steps_ge_0.10']}"
            )
            near_prev = abs(final["ap"] - bias["prevalence"]) < 0.01
            collapsed = abs(final["auc"] - 0.5) < 0.02 and abs(final["loss"] - math.log(2)) < 0.05
            lines.append(
                f"  final_ap_near_prevalence={near_prev} final_auc_near_0.5_and_loss_near_ln2={collapsed} "
                f"ln2={fmt(math.log(2))}"
            )

    lines.append("")
    lines.append("## E2 gg weight at best checkpoint, all folds")
    for row in results:
        if row["exp"] != "E2":
            continue
        lines.append(
            f"fold{row['fold']} old_gg={fmt(row['old']['weight_diagnostics']['mean_by_path'][0])} "
            f"new_gg={fmt(row['new']['weight_diagnostics']['mean_by_path'][0])} "
            f"old_H={fmt(row['old']['weight_diagnostics']['mean_entropy'])} "
            f"new_H={fmt(row['new']['weight_diagnostics']['mean_entropy'])}"
        )

    text = "\n".join(lines) + "\n"
    out = NEW / "comparison_report.txt"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
