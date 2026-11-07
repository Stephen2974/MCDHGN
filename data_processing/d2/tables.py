"""Assemble D2 feature / mask / count tables aligned to frozen node_id order."""
from .spec import CANCER_TYPES, EVENT_CLASSES, feature_columns


IDENTITY = (
    "node_id", "source_id", "hgnc_id", "symbol", "entrez_id", "mapping_status", "ambiguity_reason",
)


def empty_feature_row(gene):
    row = {key: gene.get(key, "") for key in IDENTITY}
    for column in feature_columns():
        row[column] = ""
    return row


def empty_observed_row(gene):
    row = {key: gene.get(key, "") for key in IDENTITY}
    for column in feature_columns():
        row[column] = "0"
    return row


def empty_count_row(gene, prefix):
    row = {key: gene.get(key, "") for key in IDENTITY}
    for cancer in CANCER_TYPES:
        for event in EVENT_CLASSES:
            row[f"{prefix}_{event}:{cancer}"] = ""
    return row


def rate_text(numerator, denominator):
    if denominator <= 0:
        return ""
    return f"{numerator / denominator:.10f}".rstrip("0").rstrip(".") if numerator else "0"


def build_tables(gene_rows, events, callable_patients, mapping_blocklist):
    """Build mutation_type_features and companion tables.

    callable_patients: cancer -> set(patient_id)
    events: (node_id, cancer, event_class) -> set(patient_id)
    mapping_blocklist: node_ids that must remain missing (unmapped/ambiguous/duplicate).
    """
    columns = feature_columns()
    features = []
    observed = []
    patient_counts = []
    callable_rows = []
    missing_reasons = []

    callable_by_cancer = {cancer: set(callable_patients.get(cancer, ())) for cancer in CANCER_TYPES}

    for gene in gene_rows:
        node_id = str(gene["node_id"])
        blocked = node_id in mapping_blocklist or gene.get("mapping_status") in {
            "unmapped", "ambiguous", "duplicate_identity",
        }
        feat = empty_feature_row(gene)
        obs = empty_observed_row(gene)
        counts = {key: gene.get(key, "") for key in IDENTITY}
        calls = {key: gene.get(key, "") for key in IDENTITY}
        for cancer in CANCER_TYPES:
            denom = callable_by_cancer[cancer]
            calls[f"CALLABLE:{cancer}"] = str(len(denom)) if denom else "0"
            for event in EVENT_CLASSES:
                col = f"MUT_{event}:{cancer}"
                count_col = f"N_{event}:{cancer}"
                n_patients = len(events.get((node_id, cancer, event), ()))
                counts[count_col] = str(n_patients)
                if blocked:
                    feat[col] = ""
                    obs[col] = "0"
                    missing_reasons.append({
                        "node_id": node_id,
                        "column": col,
                        "reason": "mapping_blocked:" + (gene.get("mapping_status") or "unknown"),
                        "n_event_patients": str(n_patients),
                        "n_callable_patients": str(len(denom)),
                    })
                    continue
                if not denom:
                    feat[col] = ""
                    obs[col] = "0"
                    missing_reasons.append({
                        "node_id": node_id,
                        "column": col,
                        "reason": "no_callable_denominator",
                        "n_event_patients": str(n_patients),
                        "n_callable_patients": "0",
                    })
                    continue
                if n_patients > len(denom):
                    raise ValueError(
                        f"Event patients exceed callable denominator for {col} node {node_id}: "
                        f"{n_patients}>{len(denom)}"
                    )
                feat[col] = rate_text(n_patients, len(denom))
                obs[col] = "1"
        features.append(feat)
        observed.append(obs)
        patient_counts.append(counts)
        callable_rows.append(calls)

    schema = []
    for cancer in CANCER_TYPES:
        for event in EVENT_CLASSES:
            schema.append({
                "column_name": f"MUT_{event}:{cancer}",
                "modality": "mutation_type_frequency",
                "cancer_type": cancer,
                "event_class": event,
                "unit": "patient_fraction",
                "statistic": "n_patients_with_ge1_event / n_callable_patients_cohort_level",
                "missing_rule": "empty_string_when_no_denominator_or_mapping_blocked; zero_is_valid_observed",
                "source_release": "tcga_pan_can_atlas_2018@datahub:" + "860b63a7f2d23540c94e25dd35302e133c29f48b"[:12],
                "denominator_note": "cohort_level_cases_sequenced_approximation_not_per_gene_callable",
            })
    return {
        "features": features,
        "observed": observed,
        "patient_counts": patient_counts,
        "callable_rows": callable_rows,
        "missing_reasons": missing_reasons,
        "schema": schema,
        "feature_fields": list(IDENTITY) + list(columns),
        "observed_fields": list(IDENTITY) + list(columns),
        "patient_count_fields": list(IDENTITY) + [
            f"N_{event}:{cancer}" for cancer in CANCER_TYPES for event in EVENT_CLASSES
        ],
        "callable_fields": list(IDENTITY) + [f"CALLABLE:{cancer}" for cancer in CANCER_TYPES],
    }


def check_tables(features, observed, patient_counts, callable_patients):
    """Validate rates in [0,1] and event counts ≤ denominators."""
    errors = []
    columns = feature_columns()
    if len(features) != len(observed):
        errors.append("features/observed length mismatch")
    for feat, obs, counts in zip(features, observed, patient_counts):
        node_id = feat["node_id"]
        for col in columns:
            event, cancer = col.split(":")[0].replace("MUT_", ""), col.split(":")[1]
            value = feat[col]
            flag = obs[col]
            n = int(counts.get(f"N_{event}:{cancer}", "0") or 0)
            denom = len(callable_patients.get(cancer, ()))
            if flag not in {"0", "1"}:
                errors.append(f"bad observed flag {node_id} {col}")
            if flag == "0":
                if value != "":
                    errors.append(f"unobserved cell must be empty {node_id} {col}")
                continue
            if value == "":
                errors.append(f"observed cell empty {node_id} {col}")
                continue
            try:
                rate = float(value)
            except ValueError:
                errors.append(f"non-float rate {node_id} {col}={value}")
                continue
            if rate < 0 or rate > 1:
                errors.append(f"rate outside [0,1] {node_id} {col}={value}")
            if n > denom:
                errors.append(f"n>denom {node_id} {col} {n}>{denom}")
            expected = 0.0 if n == 0 else n / denom
            if abs(rate - expected) > 1e-9:
                errors.append(f"rate mismatch {node_id} {col}")
    return errors


def dominant_site_fraction(n_at_site, n_parseable, min_parseable=2):
    """Candidate concentration feature; missing when parseable support is insufficient."""
    if n_parseable is None or n_parseable < min_parseable:
        return ""
    if n_at_site is None:
        return ""
    if n_at_site > n_parseable:
        raise ValueError("site patients exceed parseable denominator")
    return rate_text(n_at_site, n_parseable)
