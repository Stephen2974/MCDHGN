"""D2 zero-fill packaging: alignment, unexpected-missing rejection, no overwrite."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_processing.d2.build_graph import (
    MissingPatternError, align_feature_rows, analyze_missing_pattern, apply_zero_fill,
)
from data_processing.d2.spec import feature_columns
from data_processing.d2.tables import IDENTITY


def gene(node_id, source, status="resolved", **extra):
    row = {
        "node_id": str(node_id),
        "source_id": source,
        "hgnc_id": extra.get("hgnc_id", ""),
        "symbol": extra.get("symbol", source),
        "entrez_id": extra.get("entrez_id", ""),
        "mapping_status": status,
        "ambiguity_reason": extra.get("ambiguity_reason", ""),
    }
    return row


def feature_row(g, values=None, blank_all=False):
    row = {key: g[key] for key in IDENTITY}
    for column in feature_columns():
        if blank_all:
            row[column] = ""
        else:
            row[column] = "0" if values is None else str(values.get(column, "0"))
    return row


def observed_row(g, observed_all=True, blank_all=False):
    row = {key: g[key] for key in IDENTITY}
    for column in feature_columns():
        if blank_all:
            row[column] = "0"
        else:
            row[column] = "1" if observed_all else "0"
    return row


class AlignTests(unittest.TestCase):
    def test_shuffled_rows_reorder_by_node_id(self):
        genes = [gene(0, "A"), gene(1, "B"), gene(2, "C")]
        rows = [
            feature_row(genes[2], {"MUT_MISSENSE:KIRC": "0.2"}),
            feature_row(genes[0], {"MUT_MISSENSE:KIRC": "0.1"}),
            feature_row(genes[1], {"MUT_MISSENSE:KIRC": "0.3"}),
        ]
        aligned = align_feature_rows(rows, genes)
        self.assertEqual([row["source_id"] for row in aligned], ["A", "B", "C"])
        self.assertEqual(aligned[0]["MUT_MISSENSE:KIRC"], "0.1")
        self.assertEqual(aligned[2]["MUT_MISSENSE:KIRC"], "0.2")


class MissingPatternTests(unittest.TestCase):
    def setUp(self):
        self.genes = [
            gene(0, "OK"),
            gene(1, "TAZ", status="ambiguous", ambiguity_reason="multiple_hgnc_candidates"),
            gene(2, "CBSL", status="unmapped"),
        ]

    def _ok_bundle(self):
        features = [
            feature_row(self.genes[0], {"MUT_MISSENSE:KIRC": "0.5"}),
            feature_row(self.genes[1], blank_all=True),
            feature_row(self.genes[2], blank_all=True),
        ]
        # fill non-blank columns for gene 0 with zeros for remaining cols already done
        observed = [
            observed_row(self.genes[0], observed_all=True),
            observed_row(self.genes[1], blank_all=True),
            observed_row(self.genes[2], blank_all=True),
        ]
        reasons = []
        for node in (1, 2):
            status = self.genes[node]["mapping_status"]
            for column in feature_columns():
                reasons.append({
                    "node_id": str(node),
                    "column": column,
                    "reason": "mapping_blocked:" + status,
                    "n_event_patients": "0",
                    "n_callable_patients": "10",
                })
        return features, observed, reasons

    def test_approved_pattern_fills_exactly_blocked_cells(self):
        features, observed, reasons = self._ok_bundle()
        analysis = analyze_missing_pattern(features, observed, reasons, self.genes)
        self.assertEqual(analysis["n_blank_cells"], 2 * 48)
        filled, n = apply_zero_fill(analysis)
        self.assertEqual(n, 96)
        self.assertEqual(filled[0]["MUT_MISSENSE:KIRC"], "0.5")
        self.assertEqual(filled[1]["MUT_MISSENSE:KIRC"], "0")
        self.assertEqual(observed[1]["MUT_MISSENSE:KIRC"], "0")

    def test_unexpected_extra_blank_is_rejected(self):
        features, observed, reasons = self._ok_bundle()
        features[0]["MUT_TRUNC:BRCA"] = ""
        observed[0]["MUT_TRUNC:BRCA"] = "0"
        with self.assertRaises(MissingPatternError):
            analyze_missing_pattern(features, observed, reasons, self.genes)

    def test_non_mapping_reason_is_rejected(self):
        features, observed, reasons = self._ok_bundle()
        reasons[0]["reason"] = "no_callable_denominator"
        with self.assertRaises(MissingPatternError):
            analyze_missing_pattern(features, observed, reasons, self.genes)

    def test_variant_root_exists_refuses_overwrite(self):
        from data_processing.d2 import build_graph as bg
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "data_variants" / "d2_mutation_zero_v1"
            target.mkdir(parents=True)
            (target / "marker.txt").write_text("existing", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                bg.build_variant(root=root, variant_root=target, d2_out=root / "missing")
            self.assertIn("already exists", str(ctx.exception))
            self.assertEqual((target / "marker.txt").read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
