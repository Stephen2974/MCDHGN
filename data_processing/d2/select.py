"""Sample eligibility for mutation denominators.

Mirrors D1 primary-tumor / COAD-READ rules, but keys off sequenced mutation
coverage (`has_mutation`) instead of CNA, and applies the legacy ultramutated
exclusion list without silently changing its membership.
"""
from collections import defaultdict
from pathlib import Path

from data_processing.d1.select import (
    BARCODE_CLASS,
    classify_coadread,
    clean_token,
    clinical_tumor_class,
    parse_tcga_sample,
)


def trim_barcode(barcode, parts=4):
    text = clean_token(barcode)
    if not text:
        return ""
    return "-".join(text.split("-")[:parts])


def load_ultramutated_ids(path):
    """Load HotNet2/EMOGI ultramutated aliquot IDs (trimmed to 4 barcode parts)."""
    path = Path(path)
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = clean_token(line)
        if not text or text.startswith("#"):
            continue
        ids.add(trim_barcode(text, 4))
    return ids


def select_samples(records, ultramutated_ids=None):
    """Annotate sample records. Exactly one primary tumor sample per patient×cancer.

    Ultramutated samples from the legacy list are excluded after primary-tumor
    eligibility is established. The dynamic 1000-mutation threshold from the old
    script is recorded as unused; only the explicit file list is applied.
    """
    ultra = ultramutated_ids or set()
    prepared = []
    for source in records:
        rec = dict(source)
        rec["sample_id"] = clean_token(rec.get("sample_id", ""))
        rec["patient_id"] = clean_token(rec.get("patient_id", ""))
        rec["selected"] = False
        rec.setdefault("cancer_type", "")
        rec.setdefault("classification_basis", "")
        rec.setdefault("selection_reason", "")
        rec.setdefault("barcode_sample_type_code", "")
        rec.setdefault("clinical_sample_class", "")
        rec.setdefault("ultramutated", False)
        if not rec.get("in_clinical"):
            rec.update(cancer_type="", classification_basis="not_classified",
                       selection_reason="missing_clinical_record")
            prepared.append(rec)
            continue
        if rec.get("is_coadread"):
            cancer, basis = classify_coadread(
                rec.get("oncotree_code", ""), rec.get("tumor_tissue_site", ""),
                rec.get("tumor_type", ""), rec.get("cancer_type_detailed", ""))
            rec["cancer_type"] = cancer
            rec["classification_basis"] = basis
            if not cancer:
                rec["selection_reason"] = basis
                prepared.append(rec)
                continue
        else:
            rec["cancer_type"] = rec.get("study_cancer", "")
            rec["classification_basis"] = "single_cancer_study:" + rec.get("study", "")
            if not rec["cancer_type"]:
                rec["selection_reason"] = "study_cancer_missing"
                prepared.append(rec)
                continue
        if not rec.get("has_mutation"):
            rec["selection_reason"] = "not_in_cases_sequenced"
            prepared.append(rec)
            continue
        parsed = parse_tcga_sample(rec["sample_id"])
        if parsed is None:
            rec["selection_reason"] = "barcode_not_tcga_or_sample_type_unverified"
            prepared.append(rec)
            continue
        patient_from_barcode, code = parsed
        rec["barcode_sample_type_code"] = code
        if rec["patient_id"] and rec["patient_id"] != patient_from_barcode:
            rec["selection_reason"] = "patient_id_barcode_contradiction"
            prepared.append(rec)
            continue
        if not rec["patient_id"]:
            rec["selection_reason"] = "clinical_patient_id_missing"
            prepared.append(rec)
            continue
        clinical_class = clinical_tumor_class(rec.get("sample_type", ""))
        barcode_class = BARCODE_CLASS.get(code, "unknown")
        rec["clinical_sample_class"] = clinical_class
        if clinical_class == "missing":
            rec["selection_reason"] = "clinical_sample_type_missing"
            prepared.append(rec)
            continue
        if barcode_class == "unknown":
            rec["selection_reason"] = "barcode_sample_type_unrecognized:" + code
            prepared.append(rec)
            continue
        if clinical_class != barcode_class:
            rec["selection_reason"] = f"clinical_barcode_contradiction:{clinical_class}_vs_{barcode_class}"
            prepared.append(rec)
            continue
        if clinical_class != "primary":
            rec["selection_reason"] = "not_primary_tumor:" + clinical_class
            prepared.append(rec)
            continue
        sample_key = trim_barcode(rec["sample_id"], 4)
        if sample_key in ultra:
            rec["ultramutated"] = True
            rec["selection_reason"] = "excluded_ultramutated_legacy_list"
            prepared.append(rec)
            continue
        rec["selected"] = True
        rec["selection_reason"] = "eligible_pending_patient_dedup"
        prepared.append(rec)

    by_patient = defaultdict(set)
    groups = defaultdict(list)
    for rec in prepared:
        if rec["selected"]:
            groups[(rec["cancer_type"], rec["patient_id"])].append(rec)
            by_patient[rec["patient_id"]].add(rec["cancer_type"])
    split_patients = {patient for patient, cancers in by_patient.items() if {"COAD", "READ"} <= cancers}
    chosen = {}
    for key, items in groups.items():
        cancer, patient = key
        if patient in split_patients and cancer in {"COAD", "READ"}:
            continue
        chosen[key] = min(items, key=lambda item: item["sample_id"])["sample_id"]
    for rec in prepared:
        if rec["selection_reason"] != "eligible_pending_patient_dedup":
            continue
        if rec["patient_id"] in split_patients and rec["cancer_type"] in {"COAD", "READ"}:
            rec["selected"] = False
            rec["selection_reason"] = "patient_assigned_to_both_coad_and_read"
            continue
        key = (rec["cancer_type"], rec["patient_id"])
        if chosen.get(key) == rec["sample_id"]:
            rec["selection_reason"] = "selected_lexicographic_primary_sequenced"
        else:
            rec["selected"] = False
            rec["selection_reason"] = "duplicate_patient_not_lexicographic_minimum"
    return prepared
