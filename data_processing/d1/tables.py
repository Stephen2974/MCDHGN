"""Assemble and strictly validate node-aligned CNV tables.

Missing measurements stay blank here. A shorter auxiliary table is an error,
never a reason to return a partial summary.
"""
import csv
from collections import Counter
from pathlib import Path

from .cna import ABSENT, CnaFormatError, format_rate
from .spec import CANCER_TYPES, feature_columns

IDENTITY = ("node_id", "source_id", "hgnc_id", "symbol", "entrez_id")
REASON_COLUMNS = ("node_id", "source_id") + CANCER_TYPES
CNV_FEATURE_COLUMNS = IDENTITY + tuple(feature_columns())
SAMPLE_CONTRACT = ("sample_id", "patient_id", "study", "cancer_type", "sample_type", "has_cna", "selected",
                   "selection_reason", "classification_basis", "oncotree_code", "tumor_tissue_site", "tumor_type",
                   "barcode_sample_type_code", "clinical_sample_class")
EVENT_COLUMNS = tuple(column for cancer in CANCER_TYPES for column in (f"AMP_COUNT:{cancer}", f"DEL_COUNT:{cancer}"))


def read_d1_tsv(path):
    """Read a D1 table and reject duplicate or empty headers before DictReader hides them."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(path.name + " has no header") from None
        if any(not column for column in header):
            raise ValueError(path.name + " has an empty column name")
        if len(set(header)) != len(header):
            raise ValueError(path.name + " has duplicate columns")
        rows = []
        for line_number, parts in enumerate(reader, start=2):
            if len(parts) != len(header):
                raise ValueError(path.name + f" has {len(parts)} columns at line {line_number}; expected {len(header)}")
            rows.append(dict(zip(header, parts)))
    return rows


def empty_store(n_nodes):
    width = len(CANCER_TYPES)
    return {
        "den": [[ABSENT] * width for _ in range(n_nodes)],
        "amp": [[0] * width for _ in range(n_nodes)],
        "dele": [[0] * width for _ in range(n_nodes)],
        "reason": [[""] * width for _ in range(n_nodes)],
    }


def apply_study_counts(store, study_counts):
    for node_id, per_cancer in study_counts.items():
        for cancer, payload in per_cancer.items():
            column = CANCER_TYPES.index(cancer)
            if payload == "conflict":
                store["den"][node_id][column] = ABSENT
                store["amp"][node_id][column] = 0
                store["dele"][node_id][column] = 0
                store["reason"][node_id][column] = "conflicting_source_rows"
                continue
            denominator, amp, deletion = payload
            if denominator < 0 or amp < 0 or deletion < 0 or amp + deletion > denominator:
                raise CnaFormatError("Event counts are inconsistent with the denominator")
            store["den"][node_id][column] = denominator
            store["amp"][node_id][column] = amp
            store["dele"][node_id][column] = deletion
            store["reason"][node_id][column] = "" if denominator else "no_valid_gistic_in_selected_patients"


def fill_absent_reasons(store, gene_rows):
    for index, row in enumerate(gene_rows):
        if int(row["node_id"]) != index:
            raise ValueError("gene_map is not in frozen node_id order")
        for column, denominator in enumerate(store["den"][index]):
            if denominator == ABSENT and not store["reason"][index][column]:
                status = row.get("mapping_status", "")
                if status != "resolved":
                    store["reason"][index][column] = "node_mapping_status:" + status
                else:
                    store["reason"][index][column] = "no_mapped_cna_row"


def _identity(row):
    return {
        "node_id": str(int(row["node_id"])),
        "source_id": row.get("source_id", ""),
        "hgnc_id": row.get("hgnc_id", ""),
        "symbol": row.get("symbol", ""),
        "entrez_id": row.get("entrez_id", ""),
    }


def build_tables(gene_rows, store):
    features, observed, patients, events = [], [], [], []
    for index, row in enumerate(gene_rows):
        if int(row["node_id"]) != index:
            raise ValueError("Refusing to emit features in anything other than node_id order")
        feature = _identity(row)
        mask = _identity(row)
        patient = _identity(row)
        event = _identity(row)
        for column, cancer in enumerate(CANCER_TYPES):
            denominator = store["den"][index][column]
            amp_name = f"CNV_AMP:{cancer}"
            del_name = f"CNV_DEL:{cancer}"
            if denominator <= 0:
                feature[amp_name] = ""
                feature[del_name] = ""
                mask[amp_name] = "0"
                mask[del_name] = "0"
                patient[cancer] = "" if denominator < 0 else "0"
                event[f"AMP_COUNT:{cancer}"] = ""
                event[f"DEL_COUNT:{cancer}"] = ""
            else:
                feature[amp_name] = format_rate(store["amp"][index][column], denominator)
                feature[del_name] = format_rate(store["dele"][index][column], denominator)
                mask[amp_name] = "1"
                mask[del_name] = "1"
                patient[cancer] = str(denominator)
                event[f"AMP_COUNT:{cancer}"] = str(store["amp"][index][column])
                event[f"DEL_COUNT:{cancer}"] = str(store["dele"][index][column])
        features.append(feature)
        observed.append(mask)
        patients.append(patient)
        events.append(event)
    return features, observed, patients, events


def _expected_table_fields(gene_rows, name, columns):
    if name == "features":
        return list(IDENTITY + tuple(columns))
    if name == "observed":
        return list(IDENTITY + tuple(columns))
    if name == "patients":
        return list(IDENTITY + CANCER_TYPES)
    if name == "events":
        return list(IDENTITY + EVENT_COLUMNS)
    if name == "reasons":
        return list(REASON_COLUMNS)
    if name == "samples":
        return list(SAMPLE_CONTRACT)
    raise ValueError("Unknown table contract: " + name)


def _fieldnames(row, name):
    keys = list(row.keys())
    if len(set(keys)) != len(keys):
        raise ValueError(name + " has duplicate columns")
    return keys


def _check_contract(rows, name, gene_rows, columns):
    if rows is None:
        raise ValueError(name + " table is missing")
    n_nodes = len(gene_rows)
    if len(rows) != n_nodes:
        raise ValueError(f"{name} row count is {len(rows)}; expected exactly {n_nodes}")
    expected = _expected_table_fields(gene_rows, name, columns)
    identity_fields = {
        "features": IDENTITY,
        "observed": IDENTITY,
        "patients": IDENTITY,
        "events": IDENTITY,
        "reasons": ("node_id", "source_id"),
        "samples": (),
    }[name]
    node_ids = []
    for index, row in enumerate(rows):
        actual = _fieldnames(row, name)
        if actual != expected:
            raise ValueError(name + f" column contract mismatch at node_id {index}")
        for key, value in row.items():
            if value is None:
                raise ValueError(name + f" contains None at node_id {index} column {key}")
        if "node_id" in row:
            try:
                node = int(row["node_id"])
            except (TypeError, ValueError):
                raise ValueError(name + " has a non-integer node_id") from None
            if str(node) != str(row["node_id"]).strip():
                raise ValueError(name + " has a non-canonical node_id")
            node_ids.append(node)
        for field in identity_fields:
            expected_value = _identity(gene_rows[index])[field] if field in IDENTITY else ""
            if row.get(field) != expected_value:
                raise ValueError(f"{name} identity mismatch at node_id {index} field {field}")
    if name != "samples":
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(name + " repeats a node_id")
        if node_ids != list(range(n_nodes)):
            raise ValueError(name + " node_id sequence is not the frozen contiguous order")


def selected_patient_limits(sample_rows):
    """Return deduplicated selected patient counts per cancer."""
    samples = list(sample_rows)
    expected = list(SAMPLE_CONTRACT)
    sample_keys = set()
    for index, row in enumerate(samples):
        actual = list(row.keys())
        if len(set(actual)) != len(actual):
            raise ValueError("sample_metadata has duplicate columns")
        if actual != expected:
            raise ValueError(f"sample_metadata column contract mismatch at row {index}")
        if any(value is None for value in row.values()):
            raise ValueError("sample_metadata contains None")
        key = (row["study"], row["sample_id"])
        if not row["study"] or not row["sample_id"] or key in sample_keys:
            raise ValueError("sample_metadata has a missing or repeated study/sample_id")
        sample_keys.add(key)
        if row["selected"] not in {"true", "false"} or row["has_cna"] not in {"true", "false"}:
            raise ValueError("sample_metadata boolean flags are not true/false")
    limits = Counter()
    seen = set()
    for row in samples:
        if row["selected"] != "true":
            continue
        key = (row["cancer_type"], row["patient_id"])
        if not row["patient_id"] or key in seen:
            continue
        seen.add(key)
        limits[row["cancer_type"]] += 1
    return limits


def check_tables(features, observed, patients, events, gene_rows, reasons=None, sample_rows=None):
    """Validate every published CNV table against the frozen identities.

    The check is intentionally structural: a loop that visits no row can never
    make the result pass. Optional sample rows bound every denominator by the
    number of selected deduplicated patients for that cancer.
    """
    from .spec import feature_columns
    columns = feature_columns()
    if len(CANCER_TYPES) != 16 or len(columns) != 32:
        raise ValueError("CNV contract is not 16 cancers and 32 features")
    if not gene_rows:
        raise ValueError("Frozen gene map is empty")
    table_rows = {"features": features, "observed": observed, "patients": patients, "events": events}
    if reasons is not None:
        table_rows["reasons"] = reasons
    for name, rows in table_rows.items():
        _check_contract(rows, name, gene_rows, columns)
    limits = selected_patient_limits(sample_rows) if sample_rows is not None else None

    missing_cells = 0
    observed_cells = 0
    per_cancer_observed = Counter()
    for index in range(len(gene_rows)):
        feature = features[index]
        mask = observed[index]
        patient = patients[index]
        event = events[index]
        reason = reasons[index] if reasons is not None else None
        for cancer in CANCER_TYPES:
            amp_name = f"CNV_AMP:{cancer}"
            del_name = f"CNV_DEL:{cancer}"
            amp_text, del_text = feature[amp_name], feature[del_name]
            amp_seen, del_seen = mask[amp_name], mask[del_name]
            if amp_seen not in {"0", "1"} or del_seen not in {"0", "1"} or amp_seen != del_seen:
                raise ValueError("Observed mask is inconsistent for " + cancer)
            seen = amp_seen == "1"
            if seen != bool(amp_text) or seen != bool(del_text):
                raise ValueError("Observed mask does not match missing feature cells")
            count_text = patient[cancer]
            amp_count = event[f"AMP_COUNT:{cancer}"]
            del_count = event[f"DEL_COUNT:{cancer}"]
            if not seen:
                missing_cells += 2
                if amp_count or del_count:
                    raise ValueError("Missing measurements were given event counts")
                if count_text not in {"", "0"}:
                    raise ValueError("Unobserved cancer has an unexpected patient count")
                if reason is not None:
                    marker = reason[cancer]
                    if count_text == "" and not marker:
                        raise ValueError("Missing measurement has no reason")
                    if count_text == "0" and marker:
                        raise ValueError("Zero-denominator measurement must not be given another missing reason")
                continue
            observed_cells += 2
            per_cancer_observed[cancer] += 1
            if reason is not None and reason[cancer]:
                raise ValueError("Observed measurement still has a missing reason")
            denominator = int(count_text)
            amp = int(amp_count)
            deletion = int(del_count)
            if denominator <= 0 or amp < 0 or deletion < 0 or amp + deletion > denominator:
                raise ValueError("Event counts exceed the effective patient count")
            if limits is not None and denominator > limits[cancer]:
                raise ValueError(f"Denominator exceeds selected patients for {cancer}")
            if format_rate(amp, denominator) != amp_text or format_rate(deletion, denominator) != del_text:
                raise ValueError("Stored frequency does not match the event counts")
            if float(amp_text) < 0 or float(amp_text) > 1 or float(del_text) < 0 or float(del_text) > 1:
                raise ValueError("Frequency is outside [0, 1]")
            if float(amp_text) + float(del_text) > 1 + 1e-12:
                raise ValueError("AMP plus DEL exceeds 1")
    checked_pairs = len(gene_rows) * len(CANCER_TYPES)
    if missing_cells + observed_cells != 2 * checked_pairs:
        raise ValueError("Table visit count does not cover every gene-cancer pair")
    return {
        "nodes": len(gene_rows),
        "cancers": list(CANCER_TYPES),
        "features": list(columns),
        "missing_cells": missing_cells,
        "observed_cells": observed_cells,
        "genes_observed_by_cancer": dict(per_cancer_observed),
        "finite_complete": missing_cells == 0,
        "checked_pairs": checked_pairs,
    }
