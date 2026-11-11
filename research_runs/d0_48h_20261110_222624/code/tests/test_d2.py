"""D2 mutation-type rules: classification, dedup, denominators, recurrence, integrity."""
import tempfile
import unittest
from pathlib import Path

from data_processing.d1.genes import map_source_row, nodes_by_hgnc
from data_processing.p0.audit import HgncIndex
from data_processing.d2.maf import RecurrenceAccumulator, accumulate_mutations, parse_protein_change
from data_processing.d2.select import select_samples, trim_barcode
from data_processing.d2.spec import EVENT_CLASSES, feature_columns, map_classification
from data_processing.d2.tables import build_tables, check_tables, dominant_site_fraction


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
        "sample_id": sample_id, "patient_id": patient, "study": "thca_tcga_pan_can_atlas_2018",
        "study_cancer": "THCA", "is_coadread": False, "oncotree_code": "THPA",
        "tumor_tissue_site": "Thyroid", "tumor_type": "Thyroid Adenocarcinoma",
        "cancer_type_detailed": "Papillary Thyroid Cancer", "sample_type": "Primary",
        "has_mutation": True, "in_clinical": True,
    }
    row.update(extra)
    return row


class ClassificationTests(unittest.TestCase):
    def test_three_classes_are_disjoint(self):
        mapped = {map_classification(k) for k in (
            "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
            "Nonstop_Mutation", "Splice_Site",
        )}
        self.assertEqual(mapped, set(EVENT_CLASSES))
        self.assertEqual(map_classification("Splice_Region"), "OTHER")
        self.assertEqual(map_classification("Silent"), "OTHER")
        self.assertEqual(map_classification("In_Frame_Del"), "OTHER")
        self.assertEqual(map_classification("Weird_New_Class"), "UNKNOWN")

    def test_feature_column_order(self):
        cols = feature_columns()
        self.assertEqual(len(cols), 48)
        self.assertEqual(cols[0], "MUT_MISSENSE:KIRC")
        self.assertEqual(cols[2], "MUT_SPLICE:KIRC")
        self.assertEqual(cols[-1], "MUT_SPLICE:KIRP")


class SelectionTests(unittest.TestCase):
    def test_ultramutated_excluded(self):
        rows = select_samples([
            sample("TCGA-A1-0001-01A", "TCGA-A1-0001"),
        ], ultramutated_ids={"TCGA-A1-0001-01A"})
        self.assertFalse(rows[0]["selected"])
        self.assertEqual(rows[0]["selection_reason"], "excluded_ultramutated_legacy_list")

    def test_lexicographic_primary(self):
        rows = select_samples([
            sample("TCGA-A1-0001-01B", "TCGA-A1-0001"),
            sample("TCGA-A1-0001-01A", "TCGA-A1-0001"),
        ])
        chosen = [row["sample_id"] for row in rows if row["selected"]]
        self.assertEqual(chosen, ["TCGA-A1-0001-01A"])


class SyntheticMafTests(unittest.TestCase):
    def setUp(self):
        self.genes = [
            gene(0, "BRAF", "HGNC:1097", "BRAF", "673"),
            gene(1, "TP53", "HGNC:11998", "TP53", "7157"),
            gene(2, "TAZ", "", "TAZ", "", status="ambiguous"),
        ]
        self.index = HgncIndex([
            hgnc("HGNC:1097", "BRAF", "673"),
            hgnc("HGNC:11998", "TP53", "7157"),
            hgnc("HGNC:11577", "TAFAZZIN", "6901", alias="TAZ"),
            hgnc("HGNC:24042", "WWTR1", "25937", alias="TAZ"),
        ])
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_maf(self, name, rows):
        header = [
            "Hugo_Symbol", "Entrez_Gene_Id", "Chromosome", "Start_Position", "End_Position",
            "Reference_Allele", "Tumor_Seq_Allele1", "Tumor_Seq_Allele2", "Variant_Classification",
            "Tumor_Sample_Barcode", "HGVSp_Short", "Protein_position", "Transcript_ID",
        ]
        path = self.root / name
        lines = ["#" + "comment", "\t".join(header)]
        for row in rows:
            lines.append("\t".join(str(row.get(col, "")) for col in header))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_patient_gene_class_dedup_and_denominator(self):
        # Patient A: two missense rows same allele/transcript variants → once
        # Patient B: sequenced, no mutation → denominator only
        # Patient C: nonsense → TRUNC
        maf = self._write_maf("toy.maf", [
            {"Hugo_Symbol": "BRAF", "Entrez_Gene_Id": "673", "Chromosome": "7",
             "Start_Position": "140453136", "End_Position": "140453136",
             "Reference_Allele": "A", "Tumor_Seq_Allele1": "A", "Tumor_Seq_Allele2": "T",
             "Variant_Classification": "Missense_Mutation",
             "Tumor_Sample_Barcode": "TCGA-A1-0001-01A", "HGVSp_Short": "p.V600E",
             "Protein_position": "600/766", "Transcript_ID": "ENST00000288602"},
            {"Hugo_Symbol": "BRAF", "Entrez_Gene_Id": "673", "Chromosome": "7",
             "Start_Position": "140453136", "End_Position": "140453136",
             "Reference_Allele": "A", "Tumor_Seq_Allele1": "A", "Tumor_Seq_Allele2": "T",
             "Variant_Classification": "Missense_Mutation",
             "Tumor_Sample_Barcode": "TCGA-A1-0001-01A", "HGVSp_Short": "p.V600E",
             "Protein_position": "600/766", "Transcript_ID": "ENST999"},
            {"Hugo_Symbol": "BRAF", "Entrez_Gene_Id": "673", "Chromosome": "7",
             "Start_Position": "1", "End_Position": "1",
             "Reference_Allele": "C", "Tumor_Seq_Allele1": "C", "Tumor_Seq_Allele2": "T",
             "Variant_Classification": "Nonsense_Mutation",
             "Tumor_Sample_Barcode": "TCGA-A1-0003-01A", "HGVSp_Short": "p.R100*",
             "Protein_position": "100", "Transcript_ID": "ENST00000288602"},
        ])
        sample_map = {
            "TCGA-A1-0001-01A": ("TCGA-A1-0001", "THCA"),
            "TCGA-A1-0002-01A": ("TCGA-A1-0002", "THCA"),  # no rows → denom only
            "TCGA-A1-0003-01A": ("TCGA-A1-0003", "THCA"),
        }
        issues = []
        rec = RecurrenceAccumulator()
        events, stats, _ = accumulate_mutations(maf, sample_map, self.index, self.genes, issues, rec)
        self.assertEqual(len(events[("0", "THCA", "MISSENSE")]), 1)
        self.assertEqual(len(events[("0", "THCA", "TRUNC")]), 1)
        self.assertEqual(stats["rows_duplicate_allele"], 1)
        callable_patients = {"THCA": {"TCGA-A1-0001", "TCGA-A1-0002", "TCGA-A1-0003"}}
        blocked = {"2"}
        tables = build_tables(self.genes, events, callable_patients, blocked)
        # BRAF missense = 1/3
        self.assertEqual(tables["features"][0]["MUT_MISSENSE:THCA"], "0.3333333333")
        self.assertEqual(tables["observed"][0]["MUT_MISSENSE:THCA"], "1")
        # zero trunc for TP53 with denom → 0 observed
        self.assertEqual(tables["features"][1]["MUT_TRUNC:THCA"], "0")
        self.assertEqual(tables["observed"][1]["MUT_TRUNC:THCA"], "1")
        # ambiguous TAZ blocked → empty
        self.assertEqual(tables["features"][2]["MUT_MISSENSE:THCA"], "")
        self.assertEqual(tables["observed"][2]["MUT_MISSENSE:THCA"], "0")
        errors = check_tables(tables["features"], tables["observed"], tables["patient_counts"], callable_patients)
        self.assertEqual(errors, [])
        # recurrence position 600
        pos_rows = [row for row in rec.export_rows() if row["protein_position"] == "600"]
        self.assertEqual(len(pos_rows), 1)
        self.assertEqual(pos_rows[0]["n_patients_same_position"], "1")

    def test_unsequenced_not_in_denominator_keeps_missing_without_fake_zero_matrix(self):
        callable_patients = {"THCA": set()}  # no callable
        tables = build_tables(self.genes, {}, callable_patients, set())
        self.assertEqual(tables["features"][0]["MUT_MISSENSE:THCA"], "")
        self.assertEqual(tables["observed"][0]["MUT_MISSENSE:THCA"], "0")

    def test_site_concentration_missing_rules(self):
        self.assertEqual(dominant_site_fraction(1, 1, min_parseable=2), "")
        self.assertEqual(dominant_site_fraction(2, 2, min_parseable=2), "1")
        self.assertEqual(dominant_site_fraction(1, 2, min_parseable=2), "0.5")

    def test_protein_parse(self):
        parsed = parse_protein_change("p.V600E", "600/766")
        self.assertEqual(parsed["protein_position"], 600)
        self.assertEqual(parsed["change_kind"], "missense_substitution")
        missing = parse_protein_change("", "")
        self.assertEqual(missing["parse_status"], "missing_or_unparsed")


class IntegritySmokeTests(unittest.TestCase):
    def test_trim_barcode(self):
        self.assertEqual(trim_barcode("TCGA-A1-0001-01A-11D-A13W-08", 4), "TCGA-A1-0001-01A")
        self.assertEqual(trim_barcode("TCGA-A1-0001-01A-11D-A13W-08", 3), "TCGA-A1-0001")


if __name__ == "__main__":
    unittest.main()
