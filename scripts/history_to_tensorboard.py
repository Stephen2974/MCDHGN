"""Build a TensorBoard logdir from finished history.jsonl runs.

Does not start training and does not read a campaign while its queue.pid
process is still running. Values that were never stored (gradient norm,
per-epoch path entropy, train AP) cannot be reconstructed.

Example:
  python scripts/history_to_tensorboard.py
  python scripts/history_to_tensorboard.py results/D1_unverified_E4_seed42_20261108_104318
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_log import convert_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path,
                        help="campaign directories; default is every finished campaign under --results")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--logdir", type=Path, default=ROOT / "results" / "tensorboard")
    args = parser.parse_args()
    report = convert_results(args.results.resolve(), args.logdir.resolve(), args.runs or None)
    for campaign in report["skipped"]:
        print(f"SKIP live {campaign}", flush=True)
    for history, destination, count in report["written"]:
        print(f"WROTE {count:5d}  {history} -> {destination}", flush=True)
    for history, message in report["errors"]:
        print(f"ERROR {history}: {message}", flush=True)
    print(f"runs={len(report['written'])} skipped_live={len(report['skipped'])} "
          f"errors={len(report['errors'])} logdir={args.logdir.resolve()}", flush=True)
    if report["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
