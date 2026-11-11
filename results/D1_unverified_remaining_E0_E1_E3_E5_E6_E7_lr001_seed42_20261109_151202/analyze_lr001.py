"""Read-only E0-E7 lr=0.01 report. Does not modify historical results or the summarizer."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(r"G:\LLLM\MCDHGN-main")
NEW = ROOT / "results" / "D1_unverified_remaining_E0_E1_E3_E5_E6_E7_lr001_seed42_20261109_151202"
SOURCES = {
    "E0": NEW / "E0",
    "E1": NEW / "E1",
    "E2": ROOT / "results" / "d1_unverified_E2_seed42_20261108" / "E2",
    "E3": NEW / "E3",
    "E4": ROOT / "results" / "D1_unverified_E4_seed42_20261108_104318" / "E4",
    "E5": NEW / "E5",
    "E6": NEW / "E6",
    "E7": NEW / "E7",
}
LR003 = ROOT / "results" / "D1_unverified_lr003_E2_E4_seed42_20261108_212833"
EXPECTED_CODE = {
    "model.py": "0c667afd9e9ffd00bf2f04290d3b880c35b303cb0ca19e24230097f9be3c18c0",
    "fusion.py": "a8cad7be137a9705052d9c88c3fd269e1058a28825e85fb4a63987ffb06e8ee5",
    "run_ablation.py": "909802329d37513858d03391628e8cb520c9e6d50f3f28147d2fc0c1dccdea0c",
    "ablation_config.py": "6f5e81a470d86048d61a57ab6ada8c8eba11c4e2aa2f9c86136ea72c1b560280",
}
EXPECTED_DATA_KEYS = 12
CONTRASTS = [
    ("E1-E0", "E1", "E0", "GATv2，相对 GAT；两边都是 global + none"),
    ("E2-E1", "E2", "E1", "node 语义权重，相对 global；两边都是 GATv2 + none"),
    ("E3-E1", "E3", "E1", "cross 融合，相对无融合；两边都是 GATv2 + global"),
    ("E4-E2", "E4", "E2", "cross 融合，相对无融合；两边都是 GATv2 + node"),
    ("E4-E3", "E4", "E3", "融合之后的 node 权重，相对 global"),
    ("E4-E6", "E4", "E6", "cross 相对等参数量的共享 pointwise MLP"),
    ("E4-E5", "E4", "E5", "保留图编码分支，alpha 0.6 相对 alpha 1"),
    ("E7-E0", "E7", "E0", "GAT 上加入 node + cross，相对 GAT + global + none"),
    ("E4-E7", "E4", "E7", "融合配置相同时，GATv2 相对 GAT"),
]


def load_result(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    history = []
    hist_path = path.parent / "history.jsonl"
    if hist_path.is_file():
        for line in hist_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                history.append(json.loads(line))
    record["_history"] = history
    record["_path"] = str(path)
    return record


def protocol_key(record):
    cfg = record["config"]
    ignored = {"experiment", "architecture", "seed", "fold"}
    kept = {k: v for k, v in cfg.items() if k not in ignored}
    return json.dumps({"config": kept, "data": record["data_sha256"]}, sort_keys=True)


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def mean(values):
    return statistics.mean(values)


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def curve_note(record):
    history = record["_history"]
    best_ap = record["validation"]["ap"]
    best_epoch = record["best_epoch"]
    if not history:
        return "无 history"
    last = history[-1]
    peak = max(history, key=lambda row: row["ap"])
    late = [row for row in history if row["epoch"] >= best_epoch]
    late_min_auc = min(row["auc"] for row in late) if late else last["auc"]
    drop = peak["ap"] - last["ap"]
    notes = []
    if last["auc"] <= 0.55 or late_min_auc <= 0.55:
        notes.append(f"最佳 epoch {best_epoch} 之后 AUC 最低到 {late_min_auc:.3f}，末次 AUC {last['auc']:.3f}")
    if drop >= 0.15:
        notes.append(f"末次 AP {last['ap']:.3f}，比曲线峰值 {peak['ap']:.3f} 低 {drop:.3f}")
    if not notes:
        return "末段没有明显塌缩"
    if best_ap >= 0.75:
        kind = "最佳检查点仍可用，后期验证失稳"
    elif best_ap >= 0.65:
        kind = "最佳检查点已经偏弱，后期还进一步变差"
    else:
        kind = "最佳检查点本身就差"
    return kind + "：" + "；".join(notes)


def weight_note(record):
    weights = record["weight_diagnostics"]["mean_by_path"]
    top = max(weights)
    entropy = record["weight_diagnostics"]["mean_entropy"]
    return top, entropy


def main():
    groups = {}
    problems = []
    protocols = {}
    for exp, folder in SOURCES.items():
        folds = {}
        for fold in range(5):
            path = folder / "seed42" / f"fold{fold}" / "result.json"
            if not path.is_file():
                problems.append(f"缺少 {exp} seed42 fold{fold}")
                continue
            record = load_result(path)
            cfg = record["config"]
            if cfg["experiment"] != exp or cfg["seed"] != 42 or cfg["fold"] != fold or cfg["lr"] != 0.01:
                problems.append(f"身份不符 {path}")
            if cfg["code_sha256"] != EXPECTED_CODE:
                problems.append(f"代码哈希不符 {exp} fold{fold}")
            if len(record["data_sha256"]) != EXPECTED_DATA_KEYS:
                problems.append(f"数据指纹文件数不是 12：{exp} fold{fold}")
            protocols.setdefault(protocol_key(record), []).append(f"{exp} fold{fold}")
            folds[fold] = record
        groups[exp] = folds

    lr003 = {}
    for exp in ("E2", "E4"):
        lr003[exp] = {}
        for fold in range(5):
            path = LR003 / exp / "seed42" / f"fold{fold}" / "result.json"
            if path.is_file():
                lr003[exp][fold] = load_result(path)

    lines = []
    add = lines.append
    add("D1 unverified，seed42，lr=0.01，E0–E7 验证集对照")
    add("数据身份仍是 unverified_temporary，本报告不把它写成已正式核验。")
    add("只使用 seed42。折间差异是描述统计，不是跨种子显著性。")
    add("E2、E4 的 lr=0.01 结果留在原目录，这里只读取。")
    add("lr=0.003 单独列在最后，不并进主表，也不按折挑选学习率。")
    add("")
    add("队列：2026-11-09 15:12:32 启动，22:38:04 结束。30 次授权运行均有 result.json，runner 无 FAILED。")
    add("新结果目录：results/D1_unverified_remaining_E0_E1_E3_E5_E6_E7_lr001_seed42_20261109_151202")
    add("E2：results/d1_unverified_E2_seed42_20261108")
    add("E4：results/D1_unverified_E4_seed42_20261108_104318")
    if problems:
        add("核对问题：")
        for item in problems:
            add("  " + item)
    else:
        add("核对：八组各五折齐全；实验名、seed、fold、lr=0.01、四份代码哈希、12 份数据指纹一致。")
    add(f"主表协议种类：{len(protocols)}。不同实验的架构字段不计入协议，因此八组可以对照。")
    add("")
    add("每折验证指标。AP/AUC/Macro-F1 来自最佳检查点，不是最后一个 epoch。")
    add("折间 SD 是五折 AP 的样本标准差（除以 n-1）。")
    add("")
    summary = {}
    for exp in ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]:
        folds = groups[exp]
        aps = [folds[i]["validation"]["ap"] for i in range(5)]
        summary[exp] = {
            "ap": mean(aps),
            "sd": sample_sd(aps),
            "auc": mean(folds[i]["validation"]["auc"] for i in range(5)),
            "f1": mean(folds[i]["validation"]["macro_f1"] for i in range(5)),
            "params": sorted({folds[i]["parameters"] for i in range(5)}),
        }
        arch = folds[0]["config"]["architecture"]
        add(f"{exp}  {arch['use_gatv2'] and 'GATv2' or 'GAT'}  {arch['semantic_mode']}  {arch['fusion_mode']}  alpha={arch['fusion_alpha']}  参数 {summary[exp]['params'][0]}")
        add(f"  五折平均 AP {summary[exp]['ap']:.4f}  样本 SD {summary[exp]['sd']:.4f}  平均 AUC {summary[exp]['auc']:.4f}  平均 Macro-F1 {summary[exp]['f1']:.4f}")
        add("  fold   AP      AUC     MacroF1  bestEp  trained  minutes  maxPath  entropy  曲线")
        for fold in range(5):
            record = folds[fold]
            top, entropy = weight_note(record)
            add(
                f"  {fold:<4d}  {record['validation']['ap']:.4f}  {record['validation']['auc']:.4f}  "
                f"{record['validation']['macro_f1']:.4f}  {record['best_epoch']:6d}  {record['epochs_trained']:7d}  "
                f"{record['train_seconds']/60:7.1f}  {top:7.3f}  {entropy:7.3f}  {curve_note(record)}"
            )
        add("")

    add("结构对照。差值是五折 AP 平均差，正值表示前一项更高。单位是 AP，不是百分点。")
    add("同一对折也列出，便于看到差别是否集中在个别折。")
    add("")
    for name, left, right, meaning in CONTRASTS:
        deltas = [groups[left][i]["validation"]["ap"] - groups[right][i]["validation"]["ap"] for i in range(5)]
        add(f"{name}  {meaning}")
        add(f"  平均差 {mean(deltas):+.4f}  五折差的样本 SD {sample_sd(deltas):.4f}  逐折 " + "  ".join(f"f{i}:{d:+.4f}" for i, d in enumerate(deltas)))
    inter = []
    for fold in range(5):
        score = lambda exp, fold=fold: groups[exp][fold]["validation"]["ap"]
        inter.append((score("E4") - score("E3")) - (score("E2") - score("E1")))
    add("交互项 (E4-E3)-(E2-E1)，只作描述：平均 " + f"{mean(inter):+.4f}" + "  逐折 " + "  ".join(f"f{i}:{d:+.4f}" for i, d in enumerate(inter)))
    add("")
    add("读法。seed42 只有一套划分，不能把五折差当成独立样本做显著性。")
    add("E6 的 pointwise 是一套共享 MLP，按视图逐个作用，不是每个视图各有一套参数。")
    add("lr=0.01 用来对齐结构，不表示它对每个架构都合适。低 AP、权重集中或 AUC 接近 0.5 仍按原协议保留。")
    add("")
    add("明显偏差的折，分开看最佳检查点和后期曲线：")
    flagged = []
    for exp, folds in groups.items():
        for fold, record in folds.items():
            note = curve_note(record)
            ap = record["validation"]["ap"]
            if ap < 0.75 or "失稳" in note or "本身就差" in note or "进一步变差" in note:
                flagged.append((exp, fold, ap, note, record["best_epoch"], record["epochs_trained"]))
    for exp, fold, ap, note, best_epoch, trained in sorted(flagged):
        add(f"  {exp} fold{fold}  最佳 AP {ap:.4f}  best {best_epoch} / trained {trained}  {note}")
    add("")
    add("lr=0.003 敏感性对照，目录 results/D1_unverified_lr003_E2_E4_seed42_20261108_212833。")
    add("这里只并列整组，不用某一折的较低学习率替换 lr=0.01 主表。")
    for exp in ("E2", "E4"):
        rows = lr003[exp]
        if len(rows) != 5:
            add(f"  {exp} lr=0.003 折数 {len(rows)}")
            continue
        aps = [rows[i]["validation"]["ap"] for i in range(5)]
        add(f"  {exp} lr=0.003  平均 AP {mean(aps):.4f}  样本 SD {sample_sd(aps):.4f}  平均 AUC {mean(rows[i]['validation']['auc'] for i in range(5)):.4f}  平均 Macro-F1 {mean(rows[i]['validation']['macro_f1'] for i in range(5)):.4f}")
        for fold in range(5):
            record = rows[fold]
            add(f"    fold{fold}  AP {record['validation']['ap']:.4f}  AUC {record['validation']['auc']:.4f}  MacroF1 {record['validation']['macro_f1']:.4f}  best {record['best_epoch']}  trained {record['epochs_trained']}  {curve_note(record)}")
        main_aps = [groups[exp][i]["validation"]["ap"] for i in range(5)]
        add(f"    同折 lr0.003-lr0.01：" + "  ".join(f"f{i}:{aps[i]-main_aps[i]:+.4f}" for i in range(5)) + f"  平均 {mean(aps)-mean(main_aps):+.4f}")
    add("")
    text = "\n".join(lines) + "\n"
    out = NEW / "comparison_report.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("WROTE", out)


if __name__ == "__main__":
    main()
