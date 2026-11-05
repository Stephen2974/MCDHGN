"""D1 CNV rules: selection, rates, COAD/READ, mapping, integrity and graph packaging."""
import hashlib
import importlib.util
import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

from data_processing.d1.spec import CANCER_TYPES, feature_columns

from data_processing.d1.cna import CnaFormatError, accumulate_cna, frequency_from_calls, parse_gistic
from data_processing.d1.download import DownloadError, ensure_table_bytes, parse_lfsconfig, parse_pointer
from data_processing.d1.genes import map_source_row, nodes_by_hgnc
from data_processing.d1.select import classify_coadread, select_samples
from data_processing.d1.tables import apply_study_counts, build_tables, check_tables, empty_store, fill_absent_reasons, read_d1_tsv
from data_processing.p0.audit import HgncIndex


def hgnc(hid, symbol, entrez, alias="", previous=""):
    return {
        "hgnc_id": hid, "symbol": symbol, "status": "Approved", "alias_symbol": alias,
        "prev_symbol": previous, "entrez_id": entrez, "ensembl_gene_id": "", "uniprot_ids": "",
    }


def gene(node_id, source, hid, symbol, entrez, status="resolved"):
    return {
        "node_id": str(node_id), "source_id": source, "hgnc_id": hid, "symbol": symbol,
        "entrez_id": entrez, "mapping_status": status, "ambiguity_reason": "",
    }


def sample(sample_id, patient, **extra):
    row = {
        "sample_id": sample_id, "patient_id": patient, "study": "coadread_tcga_pan_can_atlas_2018",
        "study_cancer": "", "is_coadread": True, "oncotree_code": "COAD", "tumor_tissue_site": "Colon",
        "tumor_type": "Colon Adenocarcinoma", "cancer_type_detailed": "Colon Adenocarcinoma",
        "sample_type": "Primary", "has_cna": True, "in_clinical": True,
    }
    row.update(extra)
    return row


class SelectionTests(unittest.TestCase):
    def test_lexicographic_sample_ignores_event_strength(self):
        rows = select_samples([
            sample("TCGA-AA-0001-01B", "TCGA-AA-0001", gistic_amp=9),
            sample("TCGA-AA-0001-01A", "TCGA-AA-0001", gistic_amp=0),
        ])
        chosen = [row["sample_id"] for row in rows if row["selected"]]
        self.assertEqual(chosen, ["TCGA-AA-0001-01A"])
        rejected = next(row for row in rows if row["sample_id"].endswith("B"))
        self.assertEqual(rejected["selection_reason"], "duplicate_patient_not_lexicographic_minimum")

    def test_clinical_and_barcode_contradiction_is_excluded(self):
        rows = select_samples([sample("TCGA-AA-0001-06", "TCGA-AA-0001", sample_type="Primary")])
        self.assertFalse(rows[0]["selected"])
        self.assertIn("clinical_barcode_contradiction", rows[0]["selection_reason"])

    def test_metastasis_agreement_is_not_selected(self):
        rows = select_samples([sample("TCGA-AA-0001-06", "TCGA-AA-0001", sample_type="Metastasis")])
        self.assertEqual(rows[0]["selection_reason"], "not_primary_tumor:metastasis")


class CoadReadTests(unittest.TestCase):
    def test_oncotree_and_macr_rules(self):
        self.assertEqual(classify_coadread("COAD", "NA", "Colon Adenocarcinoma")[0], "COAD")
        self.assertEqual(classify_coadread("READ", "NA", "Rectum Adenocarcinoma")[0], "READ")
        self.assertEqual(classify_coadread("COAD", "Rectum", "Rectal Adenocarcinoma")[0], "")
        cancer, basis = classify_coadread(
            "MACR", "Colon", "Colon Adenocarcinoma, Mucinous Type",
            "Mucinous Adenocarcinoma of the Colon and Rectum")
        self.assertEqual(cancer, "COAD")
        self.assertIn("MACR", basis)
        self.assertEqual(classify_coadread("MACR", "Rectum", "Rectal Adenocarcinoma")[0], "READ")
        self.assertEqual(classify_coadread("MACR", "NA", "Mucinous Adenocarcinoma")[0], "")
        self.assertEqual(classify_coadread("MACR", "Liver", "Colon Adenocarcinoma")[0], "")

    def test_one_patient_cannot_enter_both_cancers(self):
        rows = select_samples([
            sample("TCGA-AA-0001-01", "TCGA-AA-0001", oncotree_code="COAD", tumor_tissue_site="Colon"),
            sample("TCGA-AA-0001-01A", "TCGA-AA-0001", oncotree_code="READ", tumor_tissue_site="Rectum",
                   tumor_type="Rectal Adenocarcinoma", cancer_type_detailed="Rectal Adenocarcinoma"),
        ])
        self.assertTrue(all(not row["selected"] for row in rows))
        self.assertTrue(all(row["selection_reason"] == "patient_assigned_to_both_coad_and_read" for row in rows))


class RateTests(unittest.TestCase):
    def test_missing_is_outside_denominator_and_zero_is_inside(self):
        result = frequency_from_calls(["2", "0", "NA", "-2", ""])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual((result["amp_count"], result["del_count"]), (1, 1))
        self.assertEqual(result["amp_rate"], format_rate_text(1, 3))
        self.assertIsNone(frequency_from_calls(["NA", "na"])["amp_rate"])
        self.assertEqual(frequency_from_calls(["NA"])["denominator"], 0)

    def test_illegal_gistic_is_rejected_from_counts(self):
        with self.assertRaises(CnaFormatError):
            parse_gistic("3")
        result = frequency_from_calls(["3", "2", "0"])
        self.assertEqual(result["illegal"], ["3"])
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["amp_count"], 1)


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.index = HgncIndex([
            hgnc("HGNC:1", "AAA", "1", alias="OLD"),
            hgnc("HGNC:2", "BBB", "2", alias="OLD"),
            hgnc("HGNC:3", "CCC", "3"),
        ])
        self.rows = [
            gene(0, "AAA", "HGNC:1", "AAA", "1"),
            gene(1, "BBB", "HGNC:2", "BBB", "2"),
            gene(2, "DUPA", "HGNC:9", "DUP", "9", "duplicate_identity"),
            gene(3, "DUPB", "HGNC:9", "DUP", "9", "duplicate_identity"),
        ]
        self.grouped = nodes_by_hgnc(self.rows)
        self.index.records["HGNC:9"] = hgnc("HGNC:9", "DUP", "9")
        self.index.index["entrez_id"]["9"].add("HGNC:9")
        self.index.index["symbol"]["DUP"].add("HGNC:9")

    def test_ambiguous_symbol_is_not_first_alias(self):
        mapped = map_source_row("OLD", "", self.index, self.grouped)
        self.assertEqual(mapped["status"], "rejected")
        self.assertEqual(mapped["node_id"], "")

    def test_entrez_disambiguates_without_switching_genes(self):
        mapped = map_source_row("OLD", "2", self.index, self.grouped)
        self.assertEqual(mapped["status"], "accepted")
        self.assertEqual(mapped["node_id"], "1")
        self.assertEqual(mapped["method"], "entrez_hgnc")

    def test_symbol_entrez_conflict_is_rejected(self):
        mapped = map_source_row("AAA", "2", self.index, self.grouped)
        self.assertEqual(mapped["status"], "rejected")
        self.assertIn("symbol_entrez_conflict", mapped["detail"])

    def test_duplicate_identity_is_not_assigned(self):
        mapped = map_source_row("DUP", "9", self.index, self.grouped)
        self.assertEqual(mapped["status"], "rejected")
        self.assertIn("identity_not_assignable", mapped["detail"])

    def test_unambiguous_symbol_fallback(self):
        mapped = map_source_row("CCC", "", self.index, self.grouped)
        self.assertEqual(mapped["status"], "rejected")
        self.assertIn("outside_node_set", mapped["detail"])
        self.rows.append(gene(4, "CCC", "HGNC:3", "CCC", "3"))
        grouped = nodes_by_hgnc(self.rows)
        mapped = map_source_row("CCC", "NA", self.index, grouped)
        self.assertEqual(mapped["node_id"], "4")
        self.assertTrue(mapped["method"].startswith("symbol_fallback"))


class TableTests(unittest.TestCase):
    def make_rows_and_tables(self):
        rows = [gene(0, "A", "HGNC:1", "A", "1"), gene(1, "B", "", "B", "", "unmapped"), gene(2, "C", "HGNC:3", "C", "3")]
        store = empty_store(3)
        apply_study_counts(store, {2: {"KIRC": (4, 1, 1), "BRCA": (2, 0, 0)}, 0: {"BRCA": "conflict"}})
        apply_study_counts(store, {0: {"KIRC": (3, 0, 3)}})
        fill_absent_reasons(store, rows)
        tables = build_tables(rows, store)
        reasons = []
        for index, row in enumerate(rows):
            reasons.append({
                "node_id": row["node_id"], "source_id": row["source_id"],
                **{cancer: (store["reason"][index][idx] if store["den"][index][idx] < 0 else "")
                   for idx, cancer in enumerate(CANCER_TYPES)},
            })
        sample_rows = []
        for index in range(1, 5):
            for cancer in ("KIRC", "BRCA"):
                sample_rows.append({
                    "sample_id": f"{cancer}-S{index}", "patient_id": f"{cancer}-P{index}", "study": "synthetic",
                    "cancer_type": cancer, "sample_type": "Primary", "has_cna": "true", "selected": "true",
                    "selection_reason": "selected", "classification_basis": "synthetic",
                    "oncotree_code": cancer, "tumor_tissue_site": "Kidney", "tumor_type": "Renal",
                    "barcode_sample_type_code": "01", "clinical_sample_class": "primary",
                })
        return rows, tables, reasons, store, sample_rows

    def test_node_order_conflict_and_missing_denominator(self):
        rows, (features, observed, patients, events), _reasons, store, sample_rows = self.make_rows_and_tables()
        self.assertEqual([row["node_id"] for row in features], ["0", "1", "2"])
        self.assertEqual(features[0]["CNV_AMP:BRCA"], "")
        self.assertEqual(patients[0]["KIRC"], "3")
        self.assertEqual(events[0]["AMP_COUNT:KIRC"], "0")
        self.assertEqual(events[0]["DEL_COUNT:KIRC"], "3")
        self.assertEqual(patients[1]["KIRC"], "")
        self.assertEqual(store["reason"][1][0], "node_mapping_status:unmapped")
        self.assertEqual(features[2]["CNV_AMP:KIRC"], format_rate_text(1, 4))
        self.assertEqual(observed[2]["CNV_DEL:KIRC"], "1")
        store_zero = empty_store(1)
        apply_study_counts(store_zero, {0: {"KIRC": (0, 0, 0)}})
        fill_absent_reasons(store_zero, rows[:1])
        zero_features, _zero_observed, zero_patients, _events = build_tables(rows[:1], store_zero)
        self.assertEqual(zero_features[0]["CNV_AMP:KIRC"], "")
        self.assertEqual(zero_patients[0]["KIRC"], "0")
        check_tables(features, observed, patients, events, rows, sample_rows=sample_rows)

    def test_empty_and_truncated_auxiliary_tables_fail(self):
        rows, (features, observed, patients, events), reasons, _store, _samples = self.make_rows_and_tables()
        with self.assertRaisesRegex(ValueError, "row count"):
            check_tables(features, [], patients, events, rows)
        with self.assertRaisesRegex(ValueError, "row count"):
            check_tables(features, observed[:-1], patients, events, rows)
        with self.assertRaisesRegex(ValueError, "row count"):
            check_tables(features, observed, patients, events, rows, reasons=reasons[:-1])

    def test_added_swapped_and_duplicate_rows_fail(self):
        rows, (features, observed, patients, events), _reasons, _store, _samples = self.make_rows_and_tables()
        added = list(observed) + [dict(observed[-1])]
        added[-1]["node_id"] = "3"
        with self.assertRaisesRegex(ValueError, "row count"):
            check_tables(features, added, patients, events, rows)
        swapped = list(observed)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            check_tables(features, swapped, patients, events, rows)
        duplicated = list(observed)
        duplicated[1] = dict(duplicated[1], node_id="0", source_id="B")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            check_tables(features, duplicated, patients, events, rows)

    def test_wrong_identity_missing_column_and_denominator_limit_fail(self):
        rows, (features, observed, patients, events), reasons, _store, sample_rows = self.make_rows_and_tables()
        wrong = [dict(row) for row in features]
        wrong[1]["source_id"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "source_id"):
            check_tables(wrong, observed, patients, events, rows)
        missing_column = []
        for row in observed:
            clone = dict(row)
            del clone["CNV_DEL:KIRP"]
            missing_column.append(clone)
        with self.assertRaisesRegex(ValueError, "column contract"):
            check_tables(features, missing_column, patients, events, rows)
        over = [dict(row) for row in patients]
        over[0]["KIRC"] = "999"
        with self.assertRaisesRegex(ValueError, "selected patients"):
            check_tables(features, observed, over, events, rows, reasons=reasons, sample_rows=sample_rows)

    def test_duplicate_header_from_reader_is_rejected(self):
        rows, (features, observed, patients, events), _reasons, _store, _samples = self.make_rows_and_tables()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dup.tsv"
            columns = list(observed[0])
            text = "\t".join(columns + ["CNV_AMP:KIRC"]) + "\n"
            text += "\t".join(observed[0][column] for column in columns) + "\t1\n"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate columns"):
                read_d1_tsv(path)

    def test_duplicate_rows_and_pointer_are_rejected(self):
        def mapper(symbol, entrez):
            if entrez == "5":
                return {"status": "accepted", "method": "entrez_hgnc", "node_id": "0", "hgnc_id": "HGNC:5", "detail": ""}
            return {"status": "rejected", "method": "", "node_id": "", "hgnc_id": "", "detail": "unmapped_symbol"}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "study" / "data_cna.txt"
            path.parent.mkdir()
            path.write_text("Hugo_Symbol\tEntrez_Gene_Id\tS1\tS2\nGENE\t5\t2\t0\nGENE\t5\t2\t1\n", encoding="utf-8")
            result = accumulate_cna(path, {"KIRC": ["S1", "S2"]}, mapper, 1, ("KIRC",))
            self.assertEqual(result["counts"][0]["KIRC"], "conflict")
            path.write_text("Hugo_Symbol\tEntrez_Gene_Id\tS1\tS2\nGENE\t5\t2\t0\nALIAS\t5\t2\t0\n", encoding="utf-8")
            result = accumulate_cna(path, {"KIRC": ["S1", "S2"]}, mapper, 1, ("KIRC",))
            self.assertEqual(result["counts"][0]["KIRC"], (2, 1, 0))
            self.assertEqual(result["stats"]["deduplicated_rows"], 1)
            path.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 4\n")
            with self.assertRaises(Exception):
                accumulate_cna(path, {"KIRC": ["S1"]}, mapper, 1, ("KIRC",))

    def test_lfs_pointer_is_not_a_table(self):
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"ab" * 32 + b"\nsize 12\n"
        oid, size = parse_pointer(pointer)
        self.assertEqual((oid, size), ("ab" * 32, 12))
        with self.assertRaises(DownloadError):
            ensure_table_bytes(pointer, "data_cna.txt")
        self.assertTrue(parse_lfsconfig("[lfs]\n\turl = https://example.invalid/lfs/\n").startswith("https://"))
        with self.assertRaises(DownloadError):
            parse_lfsconfig("[lfs]\n\turl = http://example.invalid/lfs/\n")


def format_rate_text(numerator, denominator):
    from data_processing.d1.cna import format_rate
    return format_rate(numerator, denominator)


class SourceContractTests(unittest.TestCase):
    def test_oncotree_codes_are_explicit_for_project_abbreviations(self):
        from data_processing.d1.spec import ONCOTREE_TYPE
        self.assertEqual(ONCOTREE_TYPE["kirc_tcga_pan_can_atlas_2018"], "ccrcc")
        self.assertEqual(ONCOTREE_TYPE["kirp_tcga_pan_can_atlas_2018"], "prcc")
        self.assertEqual(ONCOTREE_TYPE["lihc_tcga_pan_can_atlas_2018"], "hcc")
        self.assertEqual(ONCOTREE_TYPE["thca_tcga_pan_can_atlas_2018"], "thpa")


class GraphRefusalTests(unittest.TestCase):
    def test_missing_matrix_is_refused(self):
        from data_processing.d1.build_graph import assert_ready
        with self.assertRaises(ValueError):
            assert_ready([{"CNV_AMP:KIRC": "", "CNV_DEL:KIRC": "0"}], ("CNV_AMP:KIRC", "CNV_DEL:KIRC"))


class IntegrityRegressionTests(unittest.TestCase):
    def setUp(self):
        from data_processing.d1 import pipeline
        self.pipeline = pipeline
        self.out = pipeline.OUT

    def _snapshot(self, root):
        root = Path(root)
        return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()}

    def _copy_bundle(self, base):
        base = Path(base)
        for name in ("cnv_features.tsv", "feature_observed.tsv", "cnv_patient_counts.tsv",
                     "cnv_event_counts.tsv", "missing_reasons.tsv", "sample_metadata.tsv",
                     "gene_map.tsv", "feature_schema.tsv", "quality_report.json",
                     "manifest.json", "source_manifest.json", "README.md",
                     "cnv_gene_links.tsv", "mapping_issues.tsv"):
            shutil.copyfile(self.out / name, base / name)
        for subdir in ("sources", "audit"):
            shutil.copytree(self.out / subdir, base / subdir)
        return base

    def _tamper(self, base, name, append=b"\n"):
        path = Path(base) / name
        path.write_bytes(path.read_bytes() + append)

    def test_real_copy_passes_and_each_tamper_has_target_error(self):
        cases = [
            ("sources/kirc_tcga_pan_can_atlas_2018/data_cna.txt", "Source integrity failed"),
            ("cnv_features.tsv", "Registered output hash mismatch"),
            ("feature_observed.tsv", "Registered output hash mismatch"),
            ("cnv_patient_counts.tsv", "Registered output hash mismatch"),
            ("cnv_event_counts.tsv", "Registered output hash mismatch"),
            ("feature_schema.tsv", "Registered output hash mismatch"),
        ]
        official_before = self._snapshot(self.out)
        for target, error in cases:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as folder:
                base = self._copy_bundle(folder)
                before = self._snapshot(base)
                result = self.pipeline.verify(base, protected=False)
                self.assertEqual(result["status"], "data_prepared_not_trainable")
                self.assertEqual(self._snapshot(base), before)
                self._tamper(base, target)
                with self.assertRaisesRegex(ValueError, error):
                    self.pipeline.verify(base, protected=False)
        self.assertEqual(self._snapshot(self.out), official_before)

    def test_code_hash_tampering_is_detected(self):
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        manifest["code_sha256"]["data_processing/d1/tables.py"] = "0" * 64
        self.assertEqual(self.pipeline._verify_code(manifest), ["data_processing/d1/tables.py"])

    def test_verify_success_and_failure_do_not_write_logs(self):
        official_before = self._snapshot(self.out)
        with tempfile.TemporaryDirectory() as folder:
            base = self._copy_bundle(folder)
            before = self._snapshot(base)
            self.pipeline.verify(base, protected=False)
            self.assertEqual(self._snapshot(base), before)
            self._tamper(base, "cnv_features.tsv")
            with self.assertRaises(ValueError):
                self.pipeline.verify(base, protected=False)
            after = self._snapshot(base)
            self.assertEqual(set(after), set(before))
            changed = [name for name in before if after[name] != before[name]]
            self.assertEqual(changed, ["cnv_features.tsv"])
        self.assertEqual(self._snapshot(self.out), official_before)


class FirstDownloadFlowTests(unittest.TestCase):
    def test_fresh_download_is_staged_and_mocked(self):
        from data_processing.d1 import pipeline
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            source = {"commit": "860b63a7f2d23540c94e25dd35302e133c29f48b", "files": []}
            with self.assertRaisesRegex(ValueError, "expected 120"):
                pipeline.verify_sources(out, source)
            def fake_download(source_root, downloaded_at, progress=None):
                source_root.mkdir(parents=True, exist_ok=True)
                return ({
                    "lfs_endpoint": "https://example.invalid/",
                    "lfsconfig_text": "[lfs]\nurl = https://example.invalid/\n",
                    "gitattributes_text": "",
                    "readme_sha256": "0" * 64,
                    "readme_excerpt": "",
                }, [], [])
            original = pipeline.OUT
            pipeline.OUT = out
            try:
                from unittest.mock import patch
                with patch("data_processing.d1.pipeline.download_studies", fake_download):
                    info = pipeline.download(progress=None)
                self.assertTrue((out / "source_manifest.json").is_file())
                self.assertEqual(info["files"], [])
            finally:
                pipeline.OUT = original


class RealInterfacePreflightTests(unittest.TestCase):
    def test_baseline_layout_split_and_real_identity_preflight(self):
        from data_processing.d1 import build_graph
        from data_processing.d1.pipeline import OUT
        frozen = build_graph._baseline_frozen(build_graph.ROOT)
        self.assertIn("data/network/hetero/new_9nodes_graph.bin", frozen)
        _mapping, gene_rows = build_graph._load_identity(build_graph.ROOT)
        features = read_d1_tsv(OUT / "cnv_features.tsv")
        aligned = build_graph.align_feature_rows(list(reversed(features[:20])) + features[20:], gene_rows)
        self.assertEqual([row["node_id"] for row in aligned], [str(i) for i in range(len(gene_rows))])
        self.assertEqual(aligned[0]["source_id"], gene_rows[0]["source_id"])
        report = build_graph.assess(aligned)
        self.assertEqual(report["missing"], 1860)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "finite 32-column"):
                build_graph.build_variant(features, variant_root=Path(folder) / "variant",
                                          missing_policy="test policy that must not be reached")
            self.assertFalse((Path(folder) / "variant").exists())

    def test_shuffle_restores_distinct_values_by_identity(self):
        from data_processing.d1.build_graph import align_feature_rows
        rows = [gene(0, "A", "HGNC:1", "A", "1"), gene(1, "B", "HGNC:2", "B", "2"), gene(2, "C", "HGNC:3", "C", "3")]
        table = []
        for node_id, source in ((0, "A"), (1, "B"), (2, "C")):
            row = {"node_id": str(node_id), "source_id": source,
                   "hgnc_id": f"HGNC:{node_id + 1}", "symbol": source, "entrez_id": str(node_id + 1)}
            for idx, column in enumerate(feature_columns()):
                row[column] = f"{(node_id + 1) / 10 + idx / 1000:.3f}"
            table.append(row)
        aligned = align_feature_rows(list(reversed(table)), rows)
        for node_id, row in enumerate(aligned):
            expected = table[node_id]
            self.assertEqual([row[column] for column in feature_columns()],
                             [expected[column] for column in feature_columns()])


class GraphPackagingTests(unittest.TestCase):
    def test_finite_input_still_requires_policy(self):
        from data_processing.d1.build_graph import build_variant
        rows, (features, observed, patients, events), _reasons, _store, _samples = TableTests().make_rows_and_tables()
        finite = []
        for row in features:
            clone = dict(row)
            for column in feature_columns():
                clone[column] = "0"
            finite.append(clone)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "missing-value policy"):
                build_variant(finite, root=Path(folder), variant_root=Path(folder) / "variant")

    @unittest.skipUnless(importlib.util.find_spec("dgl") and importlib.util.find_spec("torch"),
                         "DGL/Torch are not installed in this environment")
    def test_synthetic_variant_build_and_reload(self):
        import torch
        import dgl
        from data_processing.d1.build_graph import build_variant
        from data_processing.p0.common import sha256, write_json, write_tsv

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data/step1").mkdir(parents=True)
            (root / "data/network/hetero").mkdir(parents=True)
            (root / "data/labels").mkdir(parents=True)
            (root / "Intermediate/label_set").mkdir(parents=True)
            (root / "Intermediate/blocks").mkdir(parents=True)
            (root / "data_versions/p0_v1/splits").mkdir(parents=True)
            (root / "data_versions/p0_v1/sources/hgnc").mkdir(parents=True)
            mapping = {"B": 1, "A": 0, "C": 2}
            with (root / "data/step1/gene_toids.pkl").open("wb") as handle:
                pickle.dump(mapping, handle)
            hetero = dgl.heterograph({
                ("Gene", "gg", "Gene"): (torch.tensor([0, 1]), torch.tensor([1, 2])),
                ("Gene", "ga", "A"): (torch.tensor([0, 1]), torch.tensor([0, 0])),
            })
            hetero.nodes["Gene"].data["feature"] = torch.arange(3 * 48, dtype=torch.float32).reshape(3, 48) / 100
            dgl.save_graphs(str(root / "data/network/hetero/new_9nodes_graph.bin"), [hetero])
            with (root / "Intermediate/label_set/train_batch.pkl").open("wb") as handle:
                pickle.dump([], handle)
            for index in range(9):
                with (root / f"Intermediate/blocks/{index}gs.bin").open("wb") as handle:
                    pickle.dump({"index": index}, handle)
            write_json(root / "data_versions/p0_v1/metapaths.json", {"paths": ["gg"]})
            write_json(root / "data_versions/p0_v1/split_manifest.json", {"splits": []})
            write_tsv(root / "data_versions/p0_v1/gene_map.tsv", [gene(0, "A", "HGNC:1", "A", "1"), gene(1, "B", "HGNC:2", "B", "2"), gene(2, "C", "HGNC:3", "C", "3")], ["node_id", "source_id", "hgnc_id", "symbol", "entrez_id", "mapping_status", "ambiguity_reason"])
            write_tsv(root / "data_versions/p0_v1/feature_schema.tsv", [
                {"position": i, "column_name": f"OLD{i}", "modality": "M", "cancer_type": "X", "source": "old", "source_release": "old", "missing_rule": "none", "unit": "u"}
                for i in range(48)
            ], ["position", "column_name", "modality", "cancer_type", "source", "source_release", "missing_rule", "unit"])
            data_files = [
                "data/network/hetero/new_9nodes_graph.bin",
                "data/step1/gene_toids.pkl",
                "Intermediate/label_set/train_batch.pkl",
            ] + [f"Intermediate/blocks/{index}gs.bin" for index in range(9)]
            write_json(root / "data_versions/p0_v1/baseline_manifest.json", {"files": [{"path": path, "kind": "data", "sha256": sha256(root / path)} for path in data_files]})
            write_json(root / "data_versions/p0_v1/layout_manifest.json", {"artifacts": [
                {"path": "gene_map.tsv", "sha256": sha256(root / "data_versions/p0_v1/gene_map.tsv")},
                {"path": "feature_schema.tsv", "sha256": sha256(root / "data_versions/p0_v1/feature_schema.tsv")},
                {"path": "metapaths.json", "sha256": sha256(root / "data_versions/p0_v1/metapaths.json")},
                {"path": "split_manifest.json", "sha256": sha256(root / "data_versions/p0_v1/split_manifest.json")},
            ]})
            d1 = root / "data_versions/d1_cnv_v1"
            d1.mkdir(parents=True)
            write_json(d1 / "source_manifest.json", {"synthetic": True, "files": []})
            original_rows = []
            mask_rows = []
            rows = []
            for node_id, source in ((0, "A"), (1, "B"), (2, "C")):
                identity = {"node_id": str(node_id), "source_id": source,
                            "hgnc_id": f"HGNC:{node_id + 1}", "symbol": source, "entrez_id": str(node_id + 1)}
                original = dict(identity)
                mask = dict(identity)
                finite = dict(identity)
                for idx, column in enumerate(feature_columns()):
                    observed = not (node_id == 2 and idx < 2)
                    if observed:
                        value = f"{0.05 + (idx // 2) * 0.01:.3f}"
                        original[column] = value
                        mask[column] = "1"
                        finite[column] = value
                    else:
                        original[column] = ""
                        mask[column] = "0"
                        finite[column] = "0.400"
                original_rows.append(original)
                mask_rows.append(mask)
                rows.append(finite)
            fields = ["node_id", "source_id", "hgnc_id", "symbol", "entrez_id"] + list(feature_columns())
            write_tsv(d1 / "cnv_features.tsv", original_rows, fields)
            write_tsv(d1 / "feature_observed.tsv", mask_rows, fields)
            write_json(d1 / "manifest.json", {
                "status": "synthetic", "datahub_commit": "synthetic", "outputs": [
                    {"path": "cnv_features.tsv", "sha256": sha256(d1 / "cnv_features.tsv"), "size_bytes": (d1 / "cnv_features.tsv").stat().st_size},
                    {"path": "feature_observed.tsv", "sha256": sha256(d1 / "feature_observed.tsv"), "size_bytes": (d1 / "feature_observed.tsv").stat().st_size},
                    {"path": "source_manifest.json", "sha256": sha256(d1 / "source_manifest.json"), "size_bytes": (d1 / "source_manifest.json").stat().st_size},
                ]})
            variant = root / "new_parent" / "variant"
            policy = "Synthetic acceptance test: blanks were already replaced by an explicitly supplied finite policy."
            result = build_variant(list(reversed(rows)), root=root, variant_root=variant, missing_policy=policy)
            self.assertEqual(result["feature_shape"], [3, 80])
            loaded, _ = dgl.load_graphs(str(variant / "data/network/hetero/new_9nodes_graph.bin"))
            combined = loaded[0].nodes["Gene"].data["feature"].float()
            self.assertEqual(tuple(combined.shape), (3, 80))
            self.assertTrue(torch.equal(combined[:, :48], torch.arange(3 * 48, dtype=torch.float32).reshape(3, 48) / 100))
            expected = [[float(table_row[column]) for column in feature_columns()] for table_row in rows]
            self.assertTrue(torch.allclose(combined[:, 48:], torch.tensor(expected, dtype=torch.float32)))
            self.assertAlmostEqual(combined[2, 48].item(), float(rows[2]["CNV_AMP:KIRC"]), places=6)
            self.assertAlmostEqual(combined[2, 49].item(), float(rows[2]["CNV_DEL:KIRC"]), places=6)
            schema = read_d1_tsv(variant / "feature_schema.tsv")
            self.assertEqual(len(schema), 80)
            self.assertEqual(schema[48]["column_name"], "CNV_AMP:KIRC")
            self.assertEqual(schema[48]["missing_rule"], policy)
            self.assertEqual(sha256(variant / "gene_map.tsv"), sha256(root / "data_versions/p0_v1/gene_map.tsv"))
            self.assertEqual(sha256(variant / "data/step1/gene_toids.pkl"), sha256(root / "data/step1/gene_toids.pkl"))
            manifest = json.loads((variant / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "finite_80d_graph_packaged")
            self.assertEqual(len(manifest["outputs"]), 22)
            self.assertEqual(manifest["finite_input_path"], None)
            with self.assertRaisesRegex(ValueError, "already exists"):
                build_variant(rows, root=root, variant_root=variant, missing_policy=policy)

    @unittest.skipUnless(importlib.util.find_spec("dgl") and importlib.util.find_spec("torch"),
                         "DGL/Torch are not installed in this environment")
    def test_variant_rejects_changed_observed_and_bad_masks(self):
        import torch
        import dgl
        from data_processing.d1.build_graph import build_variant
        from data_processing.p0.common import sha256, write_json, write_tsv

        def make_root():
            folder = tempfile.TemporaryDirectory()
            root = Path(folder.name)
            (root / "data/step1").mkdir(parents=True)
            (root / "data/network/hetero").mkdir(parents=True)
            (root / "Intermediate/label_set").mkdir(parents=True)
            (root / "Intermediate/blocks").mkdir(parents=True)
            (root / "data_versions/p0_v1").mkdir(parents=True)
            mapping = {"A": 0, "B": 1}
            with (root / "data/step1/gene_toids.pkl").open("wb") as handle:
                pickle.dump(mapping, handle)
            hetero = dgl.heterograph({("Gene", "gg", "Gene"): (torch.tensor([0]), torch.tensor([1]))})
            hetero.nodes["Gene"].data["feature"] = torch.arange(2 * 48, dtype=torch.float32).reshape(2, 48) / 100
            dgl.save_graphs(str(root / "data/network/hetero/new_9nodes_graph.bin"), [hetero])
            with (root / "Intermediate/label_set/train_batch.pkl").open("wb") as handle:
                pickle.dump([], handle)
            for index in range(9):
                with (root / f"Intermediate/blocks/{index}gs.bin").open("wb") as handle:
                    pickle.dump({"index": index}, handle)
            write_json(root / "data_versions/p0_v1/metapaths.json", {"paths": ["gg"]})
            write_json(root / "data_versions/p0_v1/split_manifest.json", {"splits": []})
            write_tsv(root / "data_versions/p0_v1/gene_map.tsv", [gene(0, "A", "HGNC:1", "A", "1"), gene(1, "B", "HGNC:2", "B", "2")], ["node_id", "source_id", "hgnc_id", "symbol", "entrez_id", "mapping_status", "ambiguity_reason"])
            write_tsv(root / "data_versions/p0_v1/feature_schema.tsv", [
                {"position": i, "column_name": f"OLD{i}", "modality": "M", "cancer_type": "X", "source": "old", "source_release": "old", "missing_rule": "none", "unit": "u"}
                for i in range(48)
            ], ["position", "column_name", "modality", "cancer_type", "source", "source_release", "missing_rule", "unit"])
            data_files = ["data/network/hetero/new_9nodes_graph.bin", "data/step1/gene_toids.pkl", "Intermediate/label_set/train_batch.pkl"] + [f"Intermediate/blocks/{index}gs.bin" for index in range(9)]
            write_json(root / "data_versions/p0_v1/baseline_manifest.json", {"files": [{"path": path, "kind": "data", "sha256": sha256(root / path)} for path in data_files]})
            write_json(root / "data_versions/p0_v1/layout_manifest.json", {"artifacts": [
                {"path": "gene_map.tsv", "sha256": sha256(root / "data_versions/p0_v1/gene_map.tsv")},
                {"path": "feature_schema.tsv", "sha256": sha256(root / "data_versions/p0_v1/feature_schema.tsv")},
                {"path": "metapaths.json", "sha256": sha256(root / "data_versions/p0_v1/metapaths.json")},
                {"path": "split_manifest.json", "sha256": sha256(root / "data_versions/p0_v1/split_manifest.json")},
            ]})
            d1 = root / "data_versions/d1_cnv_v1"
            d1.mkdir(parents=True)
            write_json(d1 / "source_manifest.json", {"synthetic": True, "files": []})
            original_rows, mask_rows, finite_rows = [], [], []
            for node_id, source in ((0, "A"), (1, "B")):
                identity = {"node_id": str(node_id), "source_id": source, "hgnc_id": f"HGNC:{node_id + 1}", "symbol": source, "entrez_id": str(node_id + 1)}
                original, mask, finite = dict(identity), dict(identity), dict(identity)
                for idx, column in enumerate(feature_columns()):
                    observed = not (node_id == 1 and idx == 0)
                    base_value = 0.05 + (idx // 2) * 0.01
                    original[column] = f"{base_value:.3f}" if observed else ""
                    mask[column] = "1" if observed else "0"
                    finite[column] = original[column] if observed else "0.500"
                original_rows.append(original); mask_rows.append(mask); finite_rows.append(finite)
            fields = ["node_id", "source_id", "hgnc_id", "symbol", "entrez_id"] + list(feature_columns())
            write_tsv(d1 / "cnv_features.tsv", original_rows, fields)
            write_tsv(d1 / "feature_observed.tsv", mask_rows, fields)
            write_json(d1 / "manifest.json", {"status": "synthetic", "outputs": [
                {"path": "cnv_features.tsv", "sha256": sha256(d1 / "cnv_features.tsv"), "size_bytes": (d1 / "cnv_features.tsv").stat().st_size},
                {"path": "feature_observed.tsv", "sha256": sha256(d1 / "feature_observed.tsv"), "size_bytes": (d1 / "feature_observed.tsv").stat().st_size},
                {"path": "source_manifest.json", "sha256": sha256(d1 / "source_manifest.json"), "size_bytes": (d1 / "source_manifest.json").stat().st_size},
            ]})
            return folder, root, finite_rows

        policy = "Synthetic acceptance test: only missing cells receive 0.5."
        folder, root, finite_rows = make_root()
        self.addCleanup(folder.cleanup)
        bad = [dict(row) for row in finite_rows]
        bad[0]["CNV_AMP:KIRC"] = "0.080"
        with self.assertRaisesRegex(ValueError, "changed an observed"):
            build_variant(bad, root=root, variant_root=root / "bad1", missing_policy=policy)
        self.assertFalse((root / "bad1").exists())

        folder, root, finite_rows = make_root()
        self.addCleanup(folder.cleanup)
        mask_path = root / "data_versions/d1_cnv_v1/feature_observed.tsv"
        lines = mask_path.read_text(encoding="utf-8").splitlines()
        mask_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cover every frozen node|row count|identity"):
            build_variant(finite_rows, root=root, variant_root=root / "bad2", missing_policy=policy)
        self.assertFalse((root / "bad2").exists())

        folder, root, finite_rows = make_root()
        self.addCleanup(folder.cleanup)
        source_path = root / "data_versions/d1_cnv_v1/source_manifest.json"
        source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Source integrity failed|source_manifest"):
            build_variant(finite_rows, root=root, variant_root=root / "bad3", missing_policy=policy)
        self.assertFalse((root / "bad3").exists())

    def test_variant_rejects_duplicate_out_of_range_and_identity_mismatch(self):
        from data_processing.d1.build_graph import align_feature_rows
        rows = [gene(0, "A", "HGNC:1", "A", "1"), gene(1, "B", "HGNC:2", "B", "2")]
        base = []
        for node_id, source in ((0, "A"), (1, "B")):
            row = {"node_id": str(node_id), "source_id": source,
                   "hgnc_id": f"HGNC:{node_id + 1}", "symbol": source, "entrez_id": str(node_id + 1)}
            row.update({column: "0" for column in feature_columns()})
            base.append(row)
        with self.assertRaisesRegex(ValueError, "repeats"):
            align_feature_rows(base + [dict(base[0])], rows)
        bad = [dict(row) for row in base]
        bad[1]["node_id"] = "2"
        with self.assertRaisesRegex(ValueError, "outside"):
            align_feature_rows(bad, rows)
        bad = [dict(row) for row in base]
        bad[1]["source_id"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "source_id"):
            align_feature_rows(bad, rows)


if __name__ == "__main__":
    unittest.main()
