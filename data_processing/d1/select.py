"""Sample eligibility and COAD/READ assignment.

Selection is fixed before any amplification or deletion count is computed.
"""
import re
from collections import defaultdict

from .spec import MISSING_TOKENS

TCGA_SAMPLE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})[A-Z0-9]*$")
BARCODE_CLASS = {
    "01": "primary",
    "02": "recurrent",
    "03": "primary_blood",
    "04": "recurrent_blood",
    "05": "additional_primary",
    "06": "metastasis",
    "07": "additional_metastasis",
    "08": "other",
    "09": "other",
    "10": "normal",
    "11": "normal",
    "12": "normal",
    "13": "normal",
    "14": "normal",
}


def clean_token(value):
    return (value or "").strip()


def is_missing_token(value):
    return clean_token(value).lower() in MISSING_TOKENS


def parse_tcga_sample(sample_id):
    match = TCGA_SAMPLE.fullmatch(clean_token(sample_id))
    if not match:
        return None
    return match.group(1), match.group(2)


def clinical_tumor_class(sample_type):
    text = " ".join(clean_token(sample_type).lower().replace("_", " ").replace("-", " ").split())
    if text in MISSING_TOKENS:
        return "missing"
    if "normal" in text:
        return "normal"
    if "metast" in text and "additional" in text:
        return "additional_metastasis"
    if "metast" in text:
        return "metastasis"
    if "recurren" in text:
        return "recurrent"
    if "primary" in text and "blood" in text:
        return "primary_blood"
    if "additional" in text and "primary" in text or "new primary" in text:
        return "additional_primary"
    if text == "primary" or text.startswith("primary "):
        return "primary"
    return "other"


def _organ_hint(text):
    lowered = clean_token(text).lower()
    if is_missing_token(lowered):
        return ""
    colon = "colon" in lowered
    rectum = "rectum" in lowered or "rectal" in lowered
    if colon and rectum:
        return "both"
    if colon:
        return "colon"
    if rectum:
        return "rectum"
    return "other"


def _site_class(site):
    if is_missing_token(site):
        return "missing"
    lowered = clean_token(site).lower()
    if lowered == "colon":
        return "colon"
    if lowered == "rectum":
        return "rectum"
    return "other"


def classify_coadread(oncotree, site, tumor_type="", cancer_type_detailed=""):
    """Return (cancer_type or '', basis).

    ONCOTREE COAD/READ assigns the cancer unless site or tumor-type text names
    only the other organ. MACR is assigned only from an explicit Colon or Rectum
    site, and is dropped when another field names only the opposite organ.
    """
    code = clean_token(oncotree).upper()
    site_class = _site_class(site)
    tumor_hint = _organ_hint(tumor_type)
    detailed_hint = _organ_hint(cancer_type_detailed)
    hints = {hint for hint in (tumor_hint, detailed_hint) if hint and hint != "both"}

    def contradicting(expected):
        opposite = "rectum" if expected == "colon" else "colon"
        if site_class == opposite:
            return True
        if opposite in hints and expected not in hints:
            return True
        return False

    if code == "COAD":
        if contradicting("colon") or site_class == "other":
            return "", f"oncotree_site_contradiction:COAD:site={clean_token(site) or 'missing'}"
        site_note = "missing" if site_class == "missing" else "Colon"
        return "COAD", f"oncotree:COAD;site:{site_note}"
    if code == "READ":
        if contradicting("rectum") or site_class == "other":
            return "", f"oncotree_site_contradiction:READ:site={clean_token(site) or 'missing'}"
        site_note = "missing" if site_class == "missing" else "Rectum"
        return "READ", f"oncotree:READ;site:{site_note}"
    if code == "MACR":
        if site_class == "colon" and "rectum" not in hints:
            return "COAD", "oncotree:MACR;site:Colon"
        if site_class == "rectum" and "colon" not in hints:
            return "READ", "oncotree:MACR;site:Rectum"
        if site_class == "missing":
            return "", "macr_site_missing"
        if site_class in {"colon", "rectum"}:
            return "", f"macr_field_contradiction:site={site_class}"
        return "", "macr_site_not_colon_or_rectum:" + (clean_token(site) or "missing")
    if not code or is_missing_token(code):
        return "", "oncotree_missing"
    return "", "oncotree_not_coad_read_or_macr:" + code


def select_samples(records):
    """Annotate sample records in place and return them.

    `selected` is true only for one lexicographically smallest primary-tumor
    sample per patient and cancer, among samples whose clinical class and TCGA
    barcode code both say primary solid tumor and that have CNA data.
    """
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
        if not rec.get("has_cna"):
            rec["selection_reason"] = "no_cna_data"
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
            rec["selection_reason"] = "selected_lexicographic_primary_with_cna"
        else:
            rec["selected"] = False
            rec["selection_reason"] = "duplicate_patient_not_lexicographic_minimum"
    return prepared
