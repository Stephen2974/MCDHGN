"""Summarize paired validation ablations; uses only the Python standard library.

Fold deltas are descriptive, not independent observations for a naive t-test.
Results of different seeds share a split and are also not new test datasets.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

from ablation_config import CONTRASTS

METRICS = ("ap", "auc", "accuracy", "macro_f1")


def stats(values):
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def protocol(record):
    ignored = {"experiment", "architecture", "seed", "fold"}
    cfg = {k: v for k, v in record["config"].items() if k not in ignored}
    return json.dumps({"config": cfg, "data": record["data_sha256"]}, sort_keys=True)


def collect(roots):
    groups, protocols = {}, set()
    for root in roots:
        for path in sorted(Path(root).rglob("result.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            cfg = record["config"]
            key = (cfg["seed"], cfg["fold"])
            exp = cfg["experiment"]
            group = groups.setdefault(exp, {})
            if key in group:
                raise ValueError(f"Duplicate {exp}, seed/fold {key}; do not count a run twice")
            group[key] = record
            protocols.add(protocol(record))
    if not groups:
        raise ValueError("No completed result.json files found")
    if len(protocols) != 1:
        raise ValueError("Mixed data/code/training protocols: summarize them separately")
    return groups


def summarize(groups, expected_folds=5, allow_partial=False):
    summaries, contrasts = {}, {}
    all_keys = set.union(*(set(g) for g in groups.values()))
    for exp, group in sorted(groups.items()):
        if not allow_partial and set(group) != all_keys:
            raise ValueError(f"{exp} lacks matching seed/fold pairs; use --allow-partial only for progress checks")
        by_seed = {}
        for seed in sorted({s for s, _ in group}):
            rows = [r for (s, _), r in group.items() if s == seed]
            folds = sorted(f for s, f in group if s == seed)
            if not allow_partial and folds != list(range(expected_folds)):
                raise ValueError(f"{exp} seed {seed}: expected all {expected_folds} folds; got {folds}")
            by_seed[str(seed)] = {m: statistics.mean(r["validation"][m] for r in rows) for m in METRICS}
        summaries[exp] = {
            "coverage": {str(s): sorted(f for ss, f in group if ss == s) for s in sorted({s for s, _ in group})},
            "all_run_metrics": {m: stats([r["validation"][m] for r in group.values()]) for m in METRICS},
            "per_seed_fold_mean": by_seed,
            "across_seed_means": {m: stats([r[m] for r in by_seed.values()]) for m in METRICS},
            "parameters": sorted({r["parameters"] for r in group.values()}),
            "total_train_hours": sum(r["train_seconds"] for r in group.values()) / 3600,
        }
    for name, (candidate, baseline) in CONTRASTS.items():
        if candidate not in groups or baseline not in groups:
            continue
        keys = sorted(set(groups[candidate]) & set(groups[baseline]))
        if not keys:
            continue
        values = {}
        for metric in METRICS:
            deltas = [groups[candidate][k]["validation"][metric] - groups[baseline][k]["validation"][metric] for k in keys]
            values[metric] = {
                "delta": stats(deltas), "mean_delta_pp": 100 * statistics.mean(deltas),
                "wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas),
                "pairs": [{"seed": k[0], "fold": k[1], "delta": d} for k, d in zip(keys, deltas)],
                "per_seed_mean_delta": {
                    str(seed): statistics.mean(d for k, d in zip(keys, deltas) if k[0] == seed)
                    for seed in sorted({s for s, _ in keys})
                },
            }
        if keys:
            contrasts[name] = {"candidate": candidate, "baseline": baseline, "metrics": values}
    interaction = {}
    if all(exp in groups for exp in ("E1", "E2", "E3", "E4")):
        keys = sorted(set.intersection(*(set(groups[e]) for e in ("E1", "E2", "E3", "E4"))))
        if keys:
            for metric in METRICS:
                values = []
                for key in keys:
                    score = lambda e: groups[e][key]["validation"][metric]
                    values.append((score("E4") - score("E3")) - (score("E2") - score("E1")))
                interaction[metric] = dict(stats(values), mean_pp=100 * statistics.mean(values))
    return {"experiments": summaries, "paired_contrasts": contrasts,
            "interaction_E4_minus_E3_minus_E2_plus_E1": interaction,
            "note": "Validation model-selection results; folds overlap in training. No independence-based p-values computed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/ablation_summary"))
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.output.name == "p0_legacy_D0_summary" and (args.output / "summary.json").is_file():
        kept = args.output
        args.output = args.output.with_name("p0_legacy_D0_summary_seeds42_43_44")
        print(f"Keeping existing summary at {kept}; writing {args.output}")
    groups = collect(args.roots)
    result = summarize(groups, args.expected_folds, args.allow_partial)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment", "seed", "fold", "best_epoch", *METRICS, "parameters", "train_seconds"])
        writer.writeheader()
        for exp, group in sorted(groups.items()):
            for (seed, fold), record in sorted(group.items()):
                writer.writerow(dict(experiment=exp, seed=seed, fold=fold, best_epoch=record["best_epoch"],
                                     **{m: record["validation"][m] for m in METRICS},
                                     parameters=record["parameters"], train_seconds=record["train_seconds"]))
    for name, row in result["paired_contrasts"].items():
        ap, auc = row["metrics"]["ap"], row["metrics"]["auc"]
        print(f"{row['candidate']}-{row['baseline']} {name}: AP {ap['mean_delta_pp']:+.4f} pp "
              f"({ap['wins']}/{ap['delta']['n']} wins), AUC {auc['mean_delta_pp']:+.4f} pp")
    print(f"Saved {args.output / 'summary.json'} and runs.csv")


if __name__ == "__main__":
    main()
