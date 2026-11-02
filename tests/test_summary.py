"""Dependency-free checks for experiment aggregation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_ablations import collect, summarize


def record(exp, seed, fold, score, protocol="fixed"):
    return {"config": {"experiment": exp, "architecture": {}, "seed": seed,
                       "fold": fold, "protocol": protocol},
            "data_sha256": {"input": "abc"}, "parameters": 100, "train_seconds": 10,
            "validation": {m: score for m in ("ap", "auc", "accuracy", "macro_f1")}}


class SummaryTests(unittest.TestCase):
    def test_paired_delta_and_seed_average(self):
        groups = {e: {} for e in ("E1", "E2", "E3", "E4")}
        for exp, gain in (("E1", 0), ("E2", .01), ("E3", .02), ("E4", .04)):
            for seed in (42, 43):
                for fold in range(5):
                    groups[exp][(seed, fold)] = record(exp, seed, fold, .7 + .001 * fold + gain)
        result = summarize(groups)
        pair = result["paired_contrasts"]["node_weights_only"]["metrics"]["ap"]
        self.assertAlmostEqual(pair["mean_delta_pp"], 1)
        self.assertEqual(pair["wins"], 10)
        self.assertEqual(result["experiments"]["E1"]["across_seed_means"]["ap"]["n"], 2)
        self.assertAlmostEqual(result["interaction_E4_minus_E3_minus_E2_plus_E1"]["ap"]["mean_pp"], 1)

    def test_incomplete_and_nonoverlapping_pairs(self):
        groups = {"E1": {(42, 0): record("E1", 42, 0, .7)},
                  "E2": {(43, 0): record("E2", 43, 0, .8)}}
        with self.assertRaisesRegex(ValueError, "matching"):
            summarize(groups)
        self.assertEqual(summarize(groups, allow_partial=True)["paired_contrasts"], {})

    def test_protocol_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i, p in enumerate(("old", "new")):
                run = root / str(i)
                run.mkdir()
                (run / "result.json").write_text(json.dumps(record("E" + str(i), 42, 0, .7, p)))
            with self.assertRaisesRegex(ValueError, "Mixed"):
                collect([root])

    def test_duplicate_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "result.json").write_text(json.dumps(record("E1", 42, 0, .7)))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collect([root, root])


if __name__ == "__main__":
    unittest.main()
