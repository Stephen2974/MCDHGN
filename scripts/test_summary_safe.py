"""Check the preserved redirect using the real sealed summarizer."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SafeSummaryTests(unittest.TestCase):
    def test_existing_summary_is_preserved_and_custom_output_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "input"
            inputs.mkdir()
            for fold in range(5):
                dest = inputs / str(fold)
                dest.mkdir()
                row = dict(config=dict(experiment="E1", architecture={}, seed=42, fold=fold),
                           data_sha256={}, parameters=10, train_seconds=1, best_epoch=1,
                           validation=dict(ap=.8, auc=.85, accuracy=.8, macro_f1=.75))
                (dest / "result.json").write_text(json.dumps(row), encoding="utf-8")
            protected = root / "p0_legacy_D0_summary"
            protected.mkdir()
            sentinel = b'{"keep": "original bytes"}\r\n'
            (protected / "summary.json").write_bytes(sentinel)
            for output in (protected, root / "custom output"):
                run = subprocess.run([sys.executable, str(ROOT / "scripts/summarize_ablations_safe.py"),
                                      str(inputs), "--output", str(output)],
                                     capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual((protected / "summary.json").read_bytes(), sentinel)
            for output in (root / "p0_legacy_D0_summary_seeds42_43_44", root / "custom output"):
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["experiments"]["E1"]["coverage"], {"42": list(range(5))})


if __name__ == "__main__":
    unittest.main()
