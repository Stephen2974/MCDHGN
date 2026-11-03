"""Preserve the old-summary redirect without editing the P0-sealed summarizer.

Accepts the same command-line arguments as summarize_ablations.py.
"""

import argparse
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=Path("results/ablation_summary"))
    args, _ = parser.parse_known_args()
    original_argv = sys.argv[:]
    original_path = sys.path[:]
    try:
        if args.output.name == "p0_legacy_D0_summary" and (args.output / "summary.json").is_file():
            target = args.output.with_name("p0_legacy_D0_summary_seeds42_43_44")
            print(f"Keeping existing summary at {args.output}; writing {target}", flush=True)
            sys.argv += ["--output", str(target)]
        sys.path.insert(0, str(ROOT))
        runpy.run_path(str(ROOT / "summarize_ablations.py"), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


if __name__ == "__main__":
    main()
