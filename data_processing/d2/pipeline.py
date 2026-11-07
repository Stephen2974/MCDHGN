"""Download, align and audit D2 mutation-type features. Does not train or edit D0/D1/P0."""
import argparse
import json
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from data_processing.d1.cna import parse_case_list, parse_meta, read_clinical_table, validate_study_meta
from data_processing.d1.select import clean_token, is_missing_token
from data_processing.p0.common import ROOT, now, read_tsv, sha256, write_json, write_tsv

from .download import download_studies, redact, sha256_file
from .lengths import exon_lengths_from_gff3_gz
from .maf import RecurrenceAccumulator, accumulate_mutations
from .select import load_ultramutated_ids, select_samples, trim_barcode
from .spec import (
    CANCER_TYPES, CLASSIFICATION_MAP, COMMIT, EVENT_CLASSES, REPOSITORY, STUDY_BATCH, STUDY_BATCH_ID,
    VERSION_ID, cancers_for_study, feature_columns, study_id, unique_studies,
)
from .tables import (
    IDENTITY, build_tables, check_tables, dominant_site_fraction,
)

OUT = ROOT / "data_versions" / VERSION_ID
CODE_FILES = (
    "data_processing/d2/__init__.py",
    "data_processing/d2/spec.py",
    "data_processing/d2/download.py",
    "data_processing/d2/select.py",
    "data_processing/d2/maf.py",
    "data_processing/d2/tables.py",
    "data_processing/d2/lengths.py",
    "data_processing/d2/pipeline.py",
    "data_processing/d2/build_graph.py",
    "data_processing/d2/README.md",
    "tests/test_d2.py",
    "tests/test_d2_build_graph.py",
)
CLINICAL_COLUMNS = (
    "PATIENT_ID", "SAMPLE_ID", "ONCOTREE_CODE", "TUMOR_TISSUE_SITE", "SAMPLE_TYPE",
    "TUMOR_TYPE", "CANCER_TYPE_DETAILED",
)
SAMPLE_FIELDS = (
    "sample_id", "patient_id", "study", "cancer_type", "sample_type", "has_mutation", "selected",
    "selection_reason", "classification_basis", "oncotree_code", "tumor_tissue_site", "tumor_type",
    "barcode_sample_type_code", "clinical_sample_class", "ultramutated",
)
ISSUE_FIELDS = (
    "issue_type", "severity", "study", "line_number", "source_symbol", "source_entrez",
    "node_id", "hgnc_id", "mapping_method", "detail", "cancer_type", "patient_id",
    "variant_classification", "event_class",
)


def log_event(step, status, details):
    stamp = now()
    text = "\n## " + stamp + " · " + step + " · " + status + "\n\n" + details.strip() + "\n"
    for path in (OUT / "EXECUTION_LOG.md", ROOT / "data_processing/d2/EXECUTION_LOG.md"):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# D2 mutation 执行日志\n\n时间使用 Asia/Shanghai。只记录已经执行的步骤。\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
    with (OUT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": stamp, "step": step, "status": status, "details": details.strip()}, ensure_ascii=False) + "\n")


def emit_tsv(path, rows, fields):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite existing release file: " + str(path))
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_tsv(temporary, rows, fields)
    temporary.replace(path)


def emit_json(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite existing release file: " + str(path))
    write_json(path, data)


HARD_PROTECT_PREFIXES = ("d0_input:", "train_code:", "p0_artifact:", "legacy_readonly:")


def protected_rows(hard_only=False):
    before = json.loads((OUT / "protected_files_before.json").read_text(encoding="utf-8"))
    rows = []
    for row in before["files"]:
        if hard_only and not row["label"].startswith(HARD_PROTECT_PREFIXES):
            continue
        rows.append((row["label"], ROOT / row["path"], row["expected_sha256"] or row["actual_sha256"], row.get("role", "")))
    return rows


def verify_protected():
    hard_mismatches = []
    soft_mismatches = []
    for label, path, digest, role in protected_rows(hard_only=False):
        ok = path.is_file() and sha256(path) == digest
        if ok:
            continue
        if label.startswith(HARD_PROTECT_PREFIXES):
            hard_mismatches.append(label)
        else:
            soft_mismatches.append(label)
    if hard_mismatches:
        raise RuntimeError("Protected training inputs/code changed: " + ", ".join(hard_mismatches[:30]))
    return {
        "checked_hard": len(protected_rows(hard_only=True)),
        "hard_mismatches": 0,
        "soft_readonly_mismatches": soft_mismatches,
        "soft_note": "d1_readonly hashes may change under concurrent D1 work; D2 must not revert them",
    }


def load_gene_rows():
    import pickle
    with (ROOT / "data/step1/gene_toids.pkl").open("rb") as handle:
        mapping = pickle.load(handle)
    source = ROOT / "data_versions/p0_v1/gene_map.tsv"
    rows = read_tsv(source)
    genes = sorted(mapping, key=mapping.get)
    if [mapping[gene] for gene in genes] != list(range(len(genes))) or len(rows) != len(genes):
        raise ValueError("Frozen gene map and gene_toids.pkl disagree")
    for index, (gene, row) in enumerate(zip(genes, rows)):
        if int(row["node_id"]) != index or row["source_id"] != gene:
            raise ValueError("gene_map.tsv is not in frozen node order at row " + str(index))
    hgnc_manifest = json.loads((ROOT / "data_versions/p0_v1/sources/hgnc/source_manifest.json").read_text(encoding="utf-8"))
    hgnc_path = ROOT / "data_versions/p0_v1/sources/hgnc/hgnc_complete_set.tsv"
    if sha256(hgnc_path) != hgnc_manifest["sha256"]:
        raise ValueError("Pinned HGNC snapshot hash mismatch")
    from data_processing.p0.audit import HgncIndex
    index = HgncIndex(read_tsv(hgnc_path))
    return rows, index, {
        "gene_toids_sha256": sha256(ROOT / "data/step1/gene_toids.pkl"),
        "gene_map_sha256": sha256(source),
        "hgnc_sha256": hgnc_manifest["sha256"],
        "hgnc_release": hgnc_manifest["release"],
    }


def mapping_blocklist(gene_rows):
    blocked = set()
    for row in gene_rows:
        if row.get("mapping_status") in {"unmapped", "ambiguous", "duplicate_identity"}:
            blocked.add(str(row["node_id"]))
    return blocked


def validate_mutations_meta(fields, study):
    errors = []
    if fields.get("cancer_study_identifier") != study:
        errors.append("cancer_study_identifier mismatch")
    if fields.get("genetic_alteration_type") != "MUTATION_EXTENDED":
        errors.append("genetic_alteration_type is not MUTATION_EXTENDED")
    if fields.get("datatype") != "MAF":
        errors.append("datatype is not MAF")
    if fields.get("data_filename") not in {None, "", "data_mutations.txt"} and fields.get("data_filename") != "data_mutations.txt":
        # some metas omit data_filename
        if fields.get("data_filename") and fields.get("data_filename") != "data_mutations.txt":
            errors.append("unexpected data_filename=" + fields.get("data_filename", ""))
    if errors:
        raise ValueError(study + " mutation meta invalid: " + "; ".join(errors))
    return {
        "study": study,
        "datatype": fields.get("datatype", ""),
        "stable_id": fields.get("stable_id", ""),
        "profile_description": fields.get("profile_description", ""),
        "genome_build": fields.get("reference_genome_id") or fields.get("ncbi_build") or "see_maf_NCBI_Build_column",
    }


def build_records(study, clinical_rows, sequenced_samples):
    sequenced = set(sequenced_samples)
    is_coadread = study == "coadread_tcga_pan_can_atlas_2018"
    study_cancers = cancers_for_study(study)
    study_cancer = "" if is_coadread else study_cancers[0]
    records = []
    for row in clinical_rows:
        sample_id = clean_token(row.get("SAMPLE_ID", ""))
        patient_id = clean_token(row.get("PATIENT_ID", ""))
        records.append({
            "sample_id": sample_id,
            "patient_id": patient_id,
            "study": study,
            "study_cancer": study_cancer,
            "is_coadread": is_coadread,
            "oncotree_code": clean_token(row.get("ONCOTREE_CODE", "")),
            "tumor_tissue_site": clean_token(row.get("TUMOR_TISSUE_SITE", "")),
            "tumor_type": clean_token(row.get("TUMOR_TYPE", "")),
            "cancer_type_detailed": clean_token(row.get("CANCER_TYPE_DETAILED", "")),
            "sample_type": clean_token(row.get("SAMPLE_TYPE", "")),
            "has_mutation": sample_id in sequenced,
            "in_clinical": True,
        })
    # sequenced samples missing from clinical
    clinical_ids = {rec["sample_id"] for rec in records}
    for sample_id in sorted(sequenced - clinical_ids):
        records.append({
            "sample_id": sample_id,
            "patient_id": trim_barcode(sample_id, 3),
            "study": study,
            "study_cancer": study_cancer,
            "is_coadread": is_coadread,
            "oncotree_code": "",
            "tumor_tissue_site": "",
            "tumor_type": "",
            "cancer_type_detailed": "",
            "sample_type": "",
            "has_mutation": True,
            "in_clinical": False,
        })
    return records


def write_rules():
    class_rows = []
    for variant, event in sorted(CLASSIFICATION_MAP.items()):
        class_rows.append({
            "Variant_Classification": variant,
            "event_class": event,
            "in_d2_48dim": "1" if event in EVENT_CLASSES else "0",
            "notes": "" if event in EVENT_CLASSES else "audited_only_not_in_MUT_*_features",
        })
    class_rows.append({
        "Variant_Classification": "<any_unlisted>",
        "event_class": "UNKNOWN",
        "in_d2_48dim": "0",
        "notes": "retained_in_class_histogram_not_assigned_to_MISSENSE_TRUNC_SPLICE",
    })
    emit_tsv(OUT / "classification_map.tsv", class_rows,
             ["Variant_Classification", "event_class", "in_d2_48dim", "notes"])
    rules = {
        "patient_id": "TCGA barcode parts 1-3 (TCGA-XX-XXXX)",
        "sample_selection": [
            "Require membership in cases_sequenced for the study",
            "Require clinical primary solid tumor agreeing with barcode sample type 01",
            "One sample per patient×cancer: lexicographically smallest SAMPLE_ID",
            "COAD/READ split uses the same ONCOTREE/site rules as D1; patients assigned to both are dropped",
            "Exclude samples present in legacy ultramutated_tumor_ids.txt (4-part barcode match)",
            "Do NOT apply a recomputed 1000-mutation threshold; the old script only used the file list",
        ],
        "dedup": [
            "Identical genomic allele + Variant_Classification within patient×gene counted once",
            "Same patient×gene×event_class counted once for D2 frequencies",
            "Multi-transcript rows for the same allele do not increase counts",
        ],
        "denominator": {
            "definition": "Deduplicated selected patients per cancer from cases_sequenced",
            "per_gene_callable": False,
            "limitation": "Cohort-level WXS/mutation-profile completion only; not locus-specific callable intervals",
            "gene_length_normalization": False,
        },
        "ultramutated_policy": {
            "source": "preprocess_data/gene_mutation/ultramutated_tumor_ids.txt",
            "dynamic_threshold_1000": "defined_in_legacy_script_but_not_used_there_or_here",
            "applied": True,
        },
    }
    emit_json(OUT / "sample_selection_and_dedup_rules.json", rules)


def write_gene_map_copy(gene_rows, gene_meta):
    emit_tsv(OUT / "gene_map.tsv", gene_rows, list(gene_rows[0].keys()))
    issues = []
    for row in gene_rows:
        if row.get("mapping_status") in {"unmapped", "ambiguous", "duplicate_identity"}:
            issues.append({
                "node_id": row["node_id"],
                "source_id": row["source_id"],
                "mapping_status": row["mapping_status"],
                "hgnc_id": row.get("hgnc_id", ""),
                "ambiguity_reason": row.get("ambiguity_reason", ""),
                "d2_policy": "no_automatic_feature_fill",
            })
    emit_tsv(OUT / "mapping_issues.tsv", issues,
             ["node_id", "source_id", "mapping_status", "hgnc_id", "ambiguity_reason", "d2_policy"])
    return gene_meta


def copy_ultramutated():
    src = ROOT / "preprocess_data/gene_mutation/ultramutated_tumor_ids.txt"
    dest = OUT / "sources" / "ultramutated_tumor_ids.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
    return load_ultramutated_ids(dest), {
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(dest),
        "n_ids": len(load_ultramutated_ids(dest)),
    }


def build_braf_trace(gene_rows, events, callable_patients, recurrence_acc, sample_rows):
    braf = next((row for row in gene_rows if row.get("symbol") == "BRAF" or row.get("source_id") == "BRAF"), None)
    if braf is None:
        return {"status": "BRAF node not found"}
    node_id = str(braf["node_id"])
    cancer = "THCA" if callable_patients.get("THCA") else next(
        (c for c in CANCER_TYPES if callable_patients.get(c)), "")
    if not cancer:
        return {"status": "no callable patients"}
    denom = sorted(callable_patients[cancer])
    trace = {
        "gene": "BRAF",
        "node_id": node_id,
        "hgnc_id": braf.get("hgnc_id", ""),
        "mapping_status": braf.get("mapping_status", ""),
        "cancer_type": cancer,
        "callable_patients_n": len(denom),
        "callable_policy": "cohort_level_cases_sequenced_selected_primary",
        "event_counts": {},
        "feature_values": {},
        "selected_samples_n": sum(1 for row in sample_rows if row.get("selected") and row.get("cancer_type") == cancer),
        "protein_position_600": [],
    }
    for event in EVENT_CLASSES:
        patients = sorted(events.get((node_id, cancer, event), ()))
        rate = "" if not denom or braf.get("mapping_status") != "resolved" else (
            "0" if not patients else f"{len(patients) / len(denom):.10f}".rstrip("0").rstrip(".")
        )
        trace["event_counts"][event] = {"n_patients": len(patients), "patient_ids_head": patients[:20]}
        trace["feature_values"][f"MUT_{event}:{cancer}"] = rate
    # position 600
    for (nid, can, tx, pos), patients in recurrence_acc.by_position.items():
        if nid == node_id and can == cancer and int(pos) == 600:
            aa_rows = []
            for (n2, c2, t2, p2, aa), pats in recurrence_acc.by_aa_change.items():
                if (n2, c2, t2, p2) == (nid, can, tx, pos):
                    aa_rows.append({"amino_acid_change": aa, "n_patients": len(pats), "patient_ids_head": sorted(pats)[:20]})
            trace["protein_position_600"].append({
                "transcript_id": tx,
                "n_patients_same_position": len(patients),
                "n_patients_missense": len(recurrence_acc.missense_by_position.get((nid, can, tx, pos), ())),
                "n_patients_parseable_in_gene_cancer": len(recurrence_acc.patients_with_parseable.get((nid, can), ())),
                "aa_changes": aa_rows,
            })
    return trace


def cmd_download(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources").mkdir(exist_ok=True)
    verify_protected()
    stamp = now()
    config, files = download_studies(
        OUT / "sources",
        stamp,
        progress=lambda msg: print(msg, flush=True),
        studies=args.studies.split(",") if args.studies else None,
    )
    manifest = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "commit_role": "Pinned DataHub file snapshot. Not the biological date of the 2018 cohort.",
        "study_batch": STUDY_BATCH,
        "study_batch_id": STUDY_BATCH_ID,
        "downloaded_at": stamp,
        "lfs_endpoint": config["lfs_endpoint"],
        "legacy_gdc_manifest": {
            "path": "preprocess_data/gene_mutation/mut_data/gdc_manifest.mutations.mutect2.2018-11-23.txt",
            "status": "uuids_not_found_on_gdc_files_api_2026-11-06",
            "decision": "reuse_blocked; used_cbioportal_pancan_2018_mutations_instead",
        },
        "denominator_source": "case_lists/cases_sequenced.txt",
        "denominator_limitation": "cohort_level_not_per_gene_callable",
        "files": [{k: v for k, v in row.items() if k != "lfsconfig_text"} for row in files],
        "readme_sha256": config["readme_sha256"],
        "usage": "Public cBioPortal DataHub MUTATION_EXTENDED tables for TCGA PanCancer Atlas 2018. Written only under data_versions/d2_mutation_v1/sources/.",
    }
    write_json(OUT / "source_manifest.json", manifest)
    log_event("download", "ok", f"Downloaded/reused {len(files)} files into {OUT / 'sources'}")
    print(json.dumps({"files": len(files), "out": str(OUT / "sources")}, ensure_ascii=False))


def cmd_build(args):
    OUT.mkdir(parents=True, exist_ok=True)
    verify_protected()
    if (OUT / "mutation_type_features.tsv").exists() and not args.force:
        raise FileExistsError("mutation_type_features.tsv exists; pass --force to rebuild into a new version dir only")
    # --force still refuses overwrite of release products; require explicit new version
    if (OUT / "mutation_type_features.tsv").exists():
        raise FileExistsError("Refusing overwrite of d2_mutation_v1; create d2_mutation_v2 if needed")

    gene_rows, hgnc_index, gene_meta = load_gene_rows()
    blocked = mapping_blocklist(gene_rows)
    ultra_ids, ultra_meta = copy_ultramutated()
    write_rules()
    write_gene_map_copy(gene_rows, gene_meta)

    sample_rows = []
    events = defaultdict(set)
    recurrence_acc = RecurrenceAccumulator()
    issues = []
    class_counter = Counter()
    study_reports = []
    all_stats = Counter()

    for study in unique_studies():
        study_dir = OUT / "sources" / study
        meta_mut = parse_meta((study_dir / "meta_mutations.txt").read_text(encoding="utf-8"))
        mut_info = validate_mutations_meta(meta_mut, study)
        study_meta = validate_study_meta(parse_meta((study_dir / "meta_study.txt").read_text(encoding="utf-8")), study)
        case_meta, sequenced = parse_case_list(study_dir / "case_lists" / "cases_sequenced.txt")
        header, clinical_rows = read_clinical_table(study_dir / "data_clinical_sample.txt")
        missing_cols = [col for col in CLINICAL_COLUMNS if col not in header]
        if missing_cols:
            raise ValueError(study + " missing clinical columns: " + ", ".join(missing_cols))
        records = build_records(study, clinical_rows, sequenced)
        selected = select_samples(records, ultramutated_ids=ultra_ids)
        sample_rows.extend(selected)
        sample_map = {}
        for rec in selected:
            if rec["selected"]:
                sample_map[rec["sample_id"]] = (rec["patient_id"], rec["cancer_type"])
        study_events, stats, class_counter = accumulate_mutations(
            study_dir / "data_mutations.txt",
            sample_map,
            hgnc_index,
            gene_rows,
            issues,
            recurrence_acc,
            class_counter=class_counter,
        )
        for key, patients in study_events.items():
            events[key].update(patients)
        all_stats.update(stats)
        study_reports.append({
            "study": study,
            "n_sequenced_case_list": len(sequenced),
            "n_clinical_samples": len(clinical_rows),
            "n_selected": sum(1 for rec in selected if rec["selected"]),
            "n_ultramutated_excluded": sum(1 for rec in selected if rec.get("ultramutated")),
            "mutation_meta": mut_info,
            "study_meta": study_meta,
            "case_list_name": case_meta.get("case_list_name", ""),
            "maf_stats": dict(stats),
        })
        print(f"processed {study}: selected={sum(1 for rec in selected if rec['selected'])} rows_mapped={stats.get('rows_mapped', 0)}", flush=True)

    callable_patients = defaultdict(set)
    for rec in sample_rows:
        if rec.get("selected"):
            callable_patients[rec["cancer_type"]].add(rec["patient_id"])

    tables = build_tables(gene_rows, events, callable_patients, blocked)
    errors = check_tables(tables["features"], tables["observed"], tables["patient_counts"], callable_patients)
    if errors:
        raise ValueError("Table validation failed: " + "; ".join(errors[:20]))

    emit_tsv(OUT / "mutation_type_features.tsv", tables["features"], tables["feature_fields"])
    emit_tsv(OUT / "feature_observed.tsv", tables["observed"], tables["observed_fields"])
    emit_tsv(OUT / "mutation_patient_counts.tsv", tables["patient_counts"], tables["patient_count_fields"])
    emit_tsv(OUT / "callable_patient_counts.tsv", tables["callable_rows"], tables["callable_fields"])
    emit_tsv(OUT / "feature_schema.tsv", tables["schema"], list(tables["schema"][0].keys()))
    emit_tsv(OUT / "sample_metadata.tsv", [
        {field: rec.get(field, "") for field in SAMPLE_FIELDS} for rec in sample_rows
    ], list(SAMPLE_FIELDS))
    emit_tsv(OUT / "missing_reasons.tsv", tables["missing_reasons"],
             ["node_id", "column", "reason", "n_event_patients", "n_callable_patients"])

    # classification histogram
    class_rows = []
    for (cancer, variant, event), count in sorted(class_counter.items()):
        class_rows.append({
            "cancer_type": cancer,
            "Variant_Classification": variant,
            "event_class": event,
            "n_rows_in_selected_samples": str(count),
        })
    emit_tsv(OUT / "classification_histogram.tsv", class_rows,
             ["cancer_type", "Variant_Classification", "event_class", "n_rows_in_selected_samples"])

    # mapping issues already written; append runtime gene issues summary
    if issues:
        # keep a capped audit of unmapped mutation rows
        emit_tsv(OUT / "mutation_row_mapping_issues_sample.tsv", issues[:5000], list(ISSUE_FIELDS))

    # recurrence outputs (separate from 48-dim)
    rec_dir = OUT / "recurrence"
    rec_dir.mkdir(exist_ok=True)
    pos_rows = recurrence_acc.export_rows()
    aa_rows = recurrence_acc.export_aa_rows()
    emit_tsv(rec_dir / "protein_site_recurrence.tsv", pos_rows,
             ["node_id", "cancer_type", "transcript_id", "protein_position",
              "n_patients_same_position", "n_patients_parseable_protein_pos",
              "n_patients_missense_at_position", "n_patients_unparsed_or_missing_pos"])
    emit_tsv(rec_dir / "protein_aa_change_recurrence.tsv", aa_rows,
             ["node_id", "cancer_type", "transcript_id", "protein_position", "amino_acid_change",
              "n_patients_same_aa_change", "n_patients_same_position", "n_patients_parseable_protein_pos"])

    # candidate concentration feature (NOT merged into D2 48 columns)
    cand_rows = []
    cand_schema = [{
        "column_name": "MUT_SITE_CONC:<CANCER>",
        "formula": "max_patients_at_any_transcript_position / n_patients_with_parseable_protein_position",
        "scope": "per node_id × cancer_type; transcripts not pooled",
        "min_parseable_patients": 2,
        "missing_rule": "empty when parseable<2 or no parseable sites; never coerce to 0",
        "singleton_note": "n_parseable==1 yielding 100% is suppressed by min_parseable=2",
        "in_d2_mutation_type_features": False,
    }]
    # build max site per gene×cancer
    max_site = {}
    for row in pos_rows:
        key = (row["node_id"], row["cancer_type"])
        n = int(row["n_patients_same_position"])
        parseable = int(row["n_patients_parseable_protein_pos"])
        prev = max_site.get(key)
        if prev is None or n > prev["n"]:
            max_site[key] = {"n": n, "parseable": parseable, "transcript_id": row["transcript_id"],
                             "protein_position": row["protein_position"]}
    for gene in gene_rows:
        node_id = str(gene["node_id"])
        out = {key: gene.get(key, "") for key in IDENTITY}
        for cancer in CANCER_TYPES:
            col = f"MUT_SITE_CONC:{cancer}"
            info = max_site.get((node_id, cancer))
            if gene.get("mapping_status") != "resolved" or info is None:
                out[col] = ""
            else:
                out[col] = dominant_site_fraction(info["n"], info["parseable"], min_parseable=2)
        cand_rows.append(out)
    emit_tsv(rec_dir / "candidate_site_concentration.tsv", cand_rows,
             list(IDENTITY) + [f"MUT_SITE_CONC:{c}" for c in CANCER_TYPES])
    emit_tsv(rec_dir / "feature_schema_recurrence.tsv", cand_schema, list(cand_schema[0].keys()))

    braf_trace = build_braf_trace(gene_rows, events, callable_patients, recurrence_acc, sample_rows)
    emit_json(OUT / "braf_trace_example.json", braf_trace)

    # gene length audit only (never used to rescale D2 frequencies)
    gencode_gz = ROOT / "preprocess_data/gene_mutation/gencode.v28.basic.annotation.gff3.gz"
    length_map = exon_lengths_from_gff3_gz(gencode_gz)
    length_rows = []
    for gene in gene_rows:
        symbol = gene.get("symbol") or gene.get("source_id") or ""
        bp = length_map.get(symbol)
        length_rows.append({
            "node_id": gene["node_id"],
            "source_id": gene.get("source_id", ""),
            "symbol": symbol,
            "exonic_bp_gencode_v28_basic": "" if bp is None else str(bp),
            "used_in_d2_frequency": "0",
        })
    (OUT / "audit").mkdir(exist_ok=True)
    emit_tsv(OUT / "audit" / "gene_exon_lengths_gencode_v28.tsv", length_rows,
             ["node_id", "source_id", "symbol", "exonic_bp_gencode_v28_basic", "used_in_d2_frequency"])

    # coverage / quality
    n_genes = len(gene_rows)
    n_blocked = len(blocked)
    observed_cells = sum(1 for row in tables["observed"] for col in feature_columns() if row[col] == "1")
    total_cells = n_genes * len(feature_columns())
    quality = {
        "version": VERSION_ID,
        "generated_at": now(),
        "n_genes": n_genes,
        "n_feature_columns": len(feature_columns()),
        "feature_columns": list(feature_columns()),
        "n_mapping_blocked_nodes": n_blocked,
        "callable_patients_per_cancer": {c: len(callable_patients.get(c, ())) for c in CANCER_TYPES},
        "observed_cell_fraction": observed_cells / total_cells if total_cells else 0,
        "observed_cells": observed_cells,
        "total_feature_cells": total_cells,
        "ultramutated": ultra_meta,
        "gene_meta": gene_meta,
        "maf_stats_total": dict(all_stats),
        "studies": study_reports,
        "denominator_is_per_gene_callable": False,
        "gene_length_normalization_applied": False,
        "gene_length_audit_path": "audit/gene_exon_lengths_gencode_v28.tsv",
        "ready_for_training_graph": False,
        "training_admission_note": "Missing cells remain; do not claim direct trainability. Later admission needs an explicit missingness protocol and a new data_variants graph; not done here.",
        "validation_errors": errors,
        "protected_verify": verify_protected(),
    }
    emit_json(OUT / "quality_report.json", quality)

    # REVIEW checklist
    (OUT / "REVIEW_CHECKLIST.md").write_text(
        "\n".join([
            "# D2 人工审核清单",
            "",
            "- [ ] 确认旧 GDC Mutect2 UUID 不可用，改用 DataHub 提交 " + COMMIT,
            "- [ ] 确认分母来自 cases_sequenced，且文档写明非逐基因 callable",
            "- [ ] 确认未做基因长度校正、未中位数插补、未写入旧 mut_data",
            "- [ ] 确认 U2AF1L5/CBSL/TAZ/MYO18A/TIAF1 无自动填值",
            "- [ ] 确认 recurrence/ 未拼入 mutation_type_features 的 48 列",
            "- [ ] 核对 BRAF 追溯案例 braf_trace_example.json",
            "- [ ] 核对 protected_files_before/after 哈希一致",
            "- [ ] 未启动训练、未改默认 data-root",
            "",
        ]),
        encoding="utf-8",
    )

    log_event("build", "ok", json.dumps({
        "n_genes": n_genes,
        "observed_cells": observed_cells,
        "callable": quality["callable_patients_per_cancer"],
    }, ensure_ascii=False))
    print(json.dumps({"status": "ok", "observed_cells": observed_cells, "callable": quality["callable_patients_per_cancer"]}, ensure_ascii=False))


def cmd_verify(args):
    verify_protected()
    required = [
        "mutation_type_features.tsv", "feature_schema.tsv", "feature_observed.tsv",
        "mutation_patient_counts.tsv", "callable_patient_counts.tsv", "sample_metadata.tsv",
        "gene_map.tsv", "mapping_issues.tsv", "classification_map.tsv",
        "sample_selection_and_dedup_rules.json", "source_manifest.json", "quality_report.json",
        "protected_files_before.json", "braf_trace_example.json", "REVIEW_CHECKLIST.md",
        "recurrence/protein_site_recurrence.tsv", "recurrence/candidate_site_concentration.tsv",
        "recurrence/feature_schema_recurrence.tsv",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    features = read_tsv(OUT / "mutation_type_features.tsv")
    observed = read_tsv(OUT / "feature_observed.tsv")
    counts = read_tsv(OUT / "mutation_patient_counts.tsv")
    sample_rows = read_tsv(OUT / "sample_metadata.tsv")
    callable_patients = defaultdict(set)
    for row in sample_rows:
        if row.get("selected") in {True, "True", "true", "1"}:
            callable_patients[row["cancer_type"]].add(row["patient_id"])
    # selected field is string in tsv
    callable_patients = defaultdict(set)
    for row in sample_rows:
        if str(row.get("selected", "")).lower() in {"true", "1"} or row.get("selection_reason", "").startswith("selected_"):
            if row.get("selection_reason", "").startswith("selected_"):
                callable_patients[row["cancer_type"]].add(row["patient_id"])
    errors = check_tables(features, observed, counts, callable_patients)
    cols = feature_columns()
    if list(features[0].keys())[len(IDENTITY):] != list(cols):
        # DictReader preserves header order
        header = list(features[0].keys())
        if header[len(IDENTITY):] != list(cols):
            errors.append("feature column order mismatch")
    if len(features) != 12944:
        errors.append(f"expected 12944 genes, got {len(features)}")
    if len(cols) != 48:
        errors.append(f"expected 48 feature columns, got {len(cols)}")
    report = {
        "missing_files": missing,
        "n_genes": len(features),
        "n_columns": len(cols),
        "table_errors": errors[:50],
        "protected": verify_protected(),
        "ok": not missing and not errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def write_protected_after():
    before = json.loads((OUT / "protected_files_before.json").read_text(encoding="utf-8"))
    rows = []
    for row in before["files"]:
        path = ROOT / row["path"]
        expected = row.get("expected_sha256") or row.get("actual_sha256")
        actual = sha256(path) if path.is_file() else None
        hard = row["label"].startswith(HARD_PROTECT_PREFIXES)
        rows.append({
            **{k: row[k] for k in ("label", "path", "role", "expected_sha256", "size") if k in row},
            "actual_sha256": actual,
            "exists": path.is_file(),
            "hard_protect": hard,
            "match": bool(actual and actual == expected),
            "size_after": path.stat().st_size if path.is_file() else None,
        })
    hard_mismatches = [row["label"] for row in rows if row["hard_protect"] and not row["match"]]
    soft_mismatches = [row["label"] for row in rows if (not row["hard_protect"]) and not row["match"]]
    payload = {
        "phase": "after",
        "generated_at": now(),
        "n_files": len(rows),
        "n_hard_mismatches": len(hard_mismatches),
        "n_soft_mismatches": len(soft_mismatches),
        "hard_mismatches": hard_mismatches,
        "soft_mismatches": soft_mismatches,
        "n_mismatches": len(hard_mismatches),
        "mismatches": hard_mismatches,
        "files": rows,
    }
    write_json(OUT / "protected_files_after.json", payload)
    return payload


def cmd_seal(args):
    cmd_verify(args)
    after = write_protected_after()
    if after["n_hard_mismatches"]:
        raise RuntimeError("Protected training inputs/code changed after D2: " + ", ".join(after["hard_mismatches"][:20]))
    log_event("seal", "ok", json.dumps({
        "hard_mismatches": after["n_hard_mismatches"],
        "soft_mismatches": after["soft_mismatches"],
    }, ensure_ascii=False))
    print(json.dumps({
        "protected_after_hard_mismatches": after["n_hard_mismatches"],
        "soft_readonly_mismatches": after["soft_mismatches"],
    }, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m data_processing.d2.pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_down = sub.add_parser("download", help="Download pinned mutation sources")
    p_down.add_argument("--studies", default="", help="Optional comma-separated study IDs")
    p_down.set_defaults(func=cmd_download)
    p_build = sub.add_parser("build", help="Build D2 tables from downloaded sources")
    p_build.add_argument("--force", action="store_true", help="Reserved; still refuses overwrite of v1")
    p_build.set_defaults(func=cmd_build)
    p_ver = sub.add_parser("verify", help="Read-only verification of outputs and protected hashes")
    p_ver.set_defaults(func=cmd_verify)
    p_seal = sub.add_parser("seal", help="Verify and write protected_files_after.json")
    p_seal.set_defaults(func=cmd_seal)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        log_event(args.cmd, "error", redact(str(exc)) + "\n" + traceback.format_exc()[-2000:])
        raise


if __name__ == "__main__":
    main()
