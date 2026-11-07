"""Stream-parse cBioPortal MUTATION_EXTENDED tables and accumulate patient sets."""
import csv
import gzip
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path

from data_processing.d1.genes import map_source_row, nodes_by_hgnc
from data_processing.d1.select import clean_token

from .select import trim_barcode
from .spec import EVENT_CLASSES, map_classification

HGVSP_RE = re.compile(
    r"^p\.([A-Z*\d]+?)(\d+)([A-Z*\d]+|=|fs(?:\*\d+)?|del(?:ins[A-Z]+)?|ins[A-Z]+|dup[A-Z]*)$",
    re.IGNORECASE,
)
HGVSP_MISSENSE_RE = re.compile(r"^p\.([A-Z])(\d+)([A-Z])$", re.IGNORECASE)
PROTEIN_POS_RE = re.compile(r"^(\d+)(?:/\d+)?$")


def open_text(path):
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def parse_protein_change(hgvsp_short, protein_position):
    """Return structured protein-site fields; missing parses stay missing (not 0)."""
    hgvsp = clean_token(hgvsp_short)
    pos_raw = clean_token(protein_position)
    position = None
    aa_ref = aa_alt = ""
    change_kind = "unparsed"
    if hgvsp and hgvsp.lower() not in {"", ".", "na", "n/a"}:
        miss = HGVSP_MISSENSE_RE.fullmatch(hgvsp)
        if miss:
            aa_ref, position, aa_alt = miss.group(1).upper(), int(miss.group(2)), miss.group(3).upper()
            change_kind = "missense_substitution"
        else:
            match = HGVSP_RE.fullmatch(hgvsp)
            if match:
                aa_ref = match.group(1)
                position = int(match.group(2))
                aa_alt = match.group(3)
                change_kind = "other_protein_change"
    if position is None and pos_raw:
        match = PROTEIN_POS_RE.fullmatch(pos_raw.split("-")[0])
        if match:
            position = int(match.group(1))
            if change_kind == "unparsed":
                change_kind = "position_only"
    return {
        "hgvsp_short": hgvsp,
        "protein_position": position,
        "aa_ref": aa_ref,
        "aa_alt": aa_alt,
        "aa_change": (aa_ref + str(position) + aa_alt) if position is not None and aa_ref and aa_alt else hgvsp,
        "parse_status": "ok" if position is not None else "missing_or_unparsed",
        "change_kind": change_kind if position is not None else "unparsed",
    }


def event_key(gene_node, cancer, event_class):
    return gene_node, cancer, event_class


def allele_key(row):
    return (
        clean_token(row.get("Chromosome", "")),
        clean_token(row.get("Start_Position", "")),
        clean_token(row.get("End_Position", "")),
        clean_token(row.get("Reference_Allele", "")),
        clean_token(row.get("Tumor_Seq_Allele2", "") or row.get("Tumor_Seq_Allele1", "")),
        clean_token(row.get("Variant_Classification", "")),
    )


def accumulate_mutations(
    maf_path,
    sample_to_patient_cancer,
    index,
    gene_rows,
    issues,
    recurrence_acc,
    class_counter=None,
):
    """Stream one mutation table.

    sample_to_patient_cancer maps sample barcode (full or 4-part) →
    (patient_id, cancer_type) for selected denominator samples only.

    Returns:
      events: dict[(node_id, cancer, event_class)] -> set(patient_id)
      stats: counters
    """
    grouped = nodes_by_hgnc(gene_rows)
    events = defaultdict(set)
    seen_patient_gene_class = set()
    seen_patient_allele = set()
    stats = Counter()
    class_counter = class_counter if class_counter is not None else Counter()

    # Build lookup for both full barcode and 4-part trim.
    lookup = {}
    for sample_id, value in sample_to_patient_cancer.items():
        lookup[sample_id] = value
        lookup[trim_barcode(sample_id, 4)] = value

    def row_iter():
        with open_text(maf_path) as fh:
            first = None
            for line in fh:
                if line.startswith("#"):
                    continue
                first = line
                break
            if first is None:
                return
            yield from csv.DictReader(itertools.chain([first], fh), delimiter="\t")

    for row in row_iter():
        stats["rows_total"] += 1
        sample = clean_token(row.get("Tumor_Sample_Barcode", ""))
        if not sample:
            stats["rows_missing_sample"] += 1
            continue
        mapped_sample = lookup.get(sample) or lookup.get(trim_barcode(sample, 4))
        if mapped_sample is None:
            stats["rows_sample_not_selected"] += 1
            continue
        patient_id, cancer = mapped_sample
        variant_class = clean_token(row.get("Variant_Classification", ""))
        event_class = map_classification(variant_class)
        class_counter[(cancer, variant_class or "EMPTY", event_class)] += 1
        stats["rows_in_selected_samples"] += 1

        symbol = row.get("Hugo_Symbol", "")
        entrez = row.get("Entrez_Gene_Id", "")
        mapped = map_source_row(symbol, entrez, index, grouped)
        if mapped["status"] != "accepted":
            stats["rows_mapping_rejected"] += 1
            if len(issues) < 5000:
                issues.append({
                    "issue_type": "mutation_gene_unmapped",
                    "severity": "row",
                    "study": "",
                    "line_number": "",
                    "source_symbol": clean_token(symbol),
                    "source_entrez": clean_token(str(entrez)),
                    "node_id": mapped.get("node_id", ""),
                    "hgnc_id": mapped.get("hgnc_id", ""),
                    "mapping_method": mapped.get("method", ""),
                    "detail": mapped.get("detail", ""),
                    "cancer_type": cancer,
                    "patient_id": patient_id,
                    "variant_classification": variant_class,
                    "event_class": event_class,
                })
            continue

        node_id = str(mapped["node_id"])
        stats["rows_mapped"] += 1

        # Dedup identical allele within patient (multi-transcript / duplicate rows).
        akey = (patient_id, cancer, node_id) + allele_key(row)
        if akey in seen_patient_allele:
            stats["rows_duplicate_allele"] += 1
            continue
        seen_patient_allele.add(akey)

        if event_class in EVENT_CLASSES:
            gkey = (patient_id, cancer, node_id, event_class)
            if gkey not in seen_patient_gene_class:
                seen_patient_gene_class.add(gkey)
                events[(node_id, cancer, event_class)].add(patient_id)
                stats["patient_gene_class_events"] += 1
            else:
                stats["rows_duplicate_patient_gene_class"] += 1
        else:
            stats["rows_other_or_unknown_class"] += 1

        # Recurrence accumulation (independent of the 48-dim classes).
        transcript = clean_token(row.get("Transcript_ID", "") or row.get("Annotation_Transcript", ""))
        protein = parse_protein_change(row.get("HGVSp_Short", ""), row.get("Protein_position", ""))
        recurrence_acc.add(
            node_id=node_id,
            cancer=cancer,
            patient_id=patient_id,
            transcript=transcript,
            protein=protein,
            variant_class=variant_class,
            event_class=event_class,
            hgvsp=protein["hgvsp_short"],
        )

    return events, stats, class_counter


class RecurrenceAccumulator:
    """Collect protein-site recurrence without mixing unresolved transcripts."""

    def __init__(self):
        # (node, cancer, transcript, position) -> patients with any parseable pos
        self.by_position = defaultdict(set)
        # (node, cancer, transcript, position, aa_change) -> patients
        self.by_aa_change = defaultdict(set)
        self.missense_by_position = defaultdict(set)
        self.patients_with_parseable = defaultdict(set)  # (node, cancer)
        self.patients_unparsed = defaultdict(set)
        self.patients_any = defaultdict(set)

    def add(self, node_id, cancer, patient_id, transcript, protein, variant_class, event_class, hgvsp):
        key_gene = (node_id, cancer)
        self.patients_any[key_gene].add(patient_id)
        if protein["parse_status"] != "ok":
            self.patients_unparsed[key_gene].add(patient_id)
            return
        # Require transcript identity for position pooling; empty transcript kept separate.
        tx = transcript or "UNKNOWN_TRANSCRIPT"
        pos = protein["protein_position"]
        pos_key = (node_id, cancer, tx, pos)
        self.by_position[pos_key].add(patient_id)
        self.patients_with_parseable[key_gene].add(patient_id)
        aa = protein["aa_change"] or hgvsp
        self.by_aa_change[(node_id, cancer, tx, pos, aa)].add(patient_id)
        if event_class == "MISSENSE" or protein["change_kind"] == "missense_substitution":
            self.missense_by_position[pos_key].add(patient_id)

    def export_rows(self):
        rows = []
        for (node_id, cancer, tx, pos), patients in sorted(
            self.by_position.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            missense_n = len(self.missense_by_position.get((node_id, cancer, tx, pos), ()))
            parseable_n = len(self.patients_with_parseable.get((node_id, cancer), ()))
            rows.append({
                "node_id": node_id,
                "cancer_type": cancer,
                "transcript_id": tx,
                "protein_position": str(pos),
                "n_patients_same_position": str(len(patients)),
                "n_patients_parseable_protein_pos": str(parseable_n),
                "n_patients_missense_at_position": str(missense_n),
                "n_patients_unparsed_or_missing_pos": str(
                    len(self.patients_unparsed.get((node_id, cancer), ()))
                ),
            })
        return rows

    def export_aa_rows(self):
        rows = []
        for (node_id, cancer, tx, pos, aa), patients in sorted(
            self.by_aa_change.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            rows.append({
                "node_id": node_id,
                "cancer_type": cancer,
                "transcript_id": tx,
                "protein_position": str(pos),
                "amino_acid_change": aa,
                "n_patients_same_aa_change": str(len(patients)),
                "n_patients_same_position": str(
                    len(self.by_position.get((node_id, cancer, tx, pos), ()))
                ),
                "n_patients_parseable_protein_pos": str(
                    len(self.patients_with_parseable.get((node_id, cancer), ()))
                ),
            })
        return rows
