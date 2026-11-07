"""Package D0+D2 as a 96-dim training variant under an explicit zero-fill policy.

Only mapping-blocked identity gaps confirmed jointly by gene_map,
feature_observed and missing_reasons may be filled with 0. Observed D2 values
are never changed. Does not train, edit D0/P0/D1, or overwrite d2_mutation_v1.
"""
import argparse
import json
import math
import os
import pickle
import shutil
import tempfile
from pathlib import Path

from data_processing.p0.common import ROOT, now, read_tsv, sha256, write_json, write_tsv

from .spec import COMMIT, VERSION_ID, feature_columns
from .tables import IDENTITY

VARIANT_NAME = "d2_mutation_zero_v1"
VARIANT = ROOT / "data_variants" / VARIANT_NAME
D2_OUT = ROOT / "data_versions" / VERSION_ID
ALLOWED_BLOCK_STATUSES = frozenset({"unmapped", "ambiguous", "duplicate_identity"})
DATA_INPUTS = (
    "data/step1/gene_toids.pkl",
    "Intermediate/label_set/train_batch.pkl",
    *(f"Intermediate/blocks/{index}gs.bin" for index in range(9)),
)
BASELINE_INPUTS = (
    "data/network/hetero/new_9nodes_graph.bin",
    *DATA_INPUTS,
)
P0_CONTRACTS = (
    "gene_map.tsv",
    "feature_schema.tsv",
    "metapaths.json",
    "split_manifest.json",
)
D2_AUDIT_COPIES = (
    "mutation_type_features.tsv",
    "feature_observed.tsv",
    "feature_schema.tsv",
    "missing_reasons.tsv",
    "gene_map.tsv",
    "mapping_issues.tsv",
    "source_manifest.json",
    "quality_report.json",
)
CODE_FILES = (
    "data_processing/d2/__init__.py",
    "data_processing/d2/spec.py",
    "data_processing/d2/tables.py",
    "data_processing/d2/build_graph.py",
    "data_processing/d2/README.md",
    "tests/test_d2_build_graph.py",
)


class MissingPatternError(ValueError):
    """Raised when the blank-cell pattern is not the approved mapping-block set."""


def _same_bytes(left, right):
    return Path(left).is_file() and Path(right).is_file() and sha256(left) == sha256(right)


def _link_or_copy(source, target):
    """Prefer a same-volume hard link; fall back to copy. Never opens source for write."""
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("Refusing to replace existing path: " + str(target))
    try:
        os.link(source, target)
        mode = "hardlink"
    except OSError:
        shutil.copyfile(source, target)
        mode = "copy"
    if not _same_bytes(source, target):
        raise ValueError("Linked/copied file hash differs from source: " + str(source))
    return {
        "path": None,
        "source": str(source),
        "mode": mode,
        "sha256": sha256(source),
        "size_bytes": source.stat().st_size,
        "samefile": os.path.samefile(source, target),
    }


def _baseline_frozen(root):
    root = Path(root)
    baseline = json.loads((root / "data_versions/p0_v1/baseline_manifest.json").read_text(encoding="utf-8"))
    frozen = {row["path"]: row["sha256"] for row in baseline["files"] if row["kind"] == "data"}
    missing = [path for path in BASELINE_INPUTS if path not in frozen]
    if missing:
        raise ValueError("D0 baseline manifest does not pin raw inputs: " + ", ".join(missing))
    mismatches = [path for path in BASELINE_INPUTS
                  if not (root / path).is_file() or sha256(root / path) != frozen[path]]
    if mismatches:
        raise ValueError("D0 raw input hash mismatch: " + ", ".join(mismatches))
    layout = json.loads((root / "data_versions/p0_v1/layout_manifest.json").read_text(encoding="utf-8"))
    artifacts = {row["path"]: row["sha256"] for row in layout["artifacts"]}
    for name in P0_CONTRACTS:
        if name not in artifacts:
            raise ValueError("P0 layout manifest does not pin " + name)
        target = root / "data_versions/p0_v1" / name
        if not target.is_file() or sha256(target) != artifacts[name]:
            raise ValueError("P0 sealed artifact hash mismatch: " + name)
    return frozen, artifacts


def load_identity(root):
    _baseline_frozen(root)
    with (Path(root) / "data/step1/gene_toids.pkl").open("rb") as handle:
        mapping = pickle.load(handle)
    if len(mapping) != len(set(mapping.values())):
        raise ValueError("gene_toids.pkl node ids are not unique")
    if any(type(node) is not int for node in mapping.values()):
        raise ValueError("gene_toids.pkl node ids are not integers")
    genes = sorted(mapping, key=mapping.get)
    if [mapping[name] for name in genes] != list(range(len(genes))):
        raise ValueError("gene_toids.pkl is not contiguous from 0")
    rows = read_tsv(Path(root) / "data_versions/p0_v1/gene_map.tsv")
    if len(rows) != len(genes):
        raise ValueError("P0 gene_map.tsv and gene_toids.pkl disagree")
    for index, (name, row) in enumerate(zip(genes, rows)):
        if int(row["node_id"]) != index or row["source_id"] != name:
            raise ValueError("P0 gene_map.tsv is not in frozen node order")
    return mapping, rows


def align_feature_rows(feature_rows, gene_rows, columns=None):
    """Reorder by node_id after validating identity fields. Never trust TSV row order."""
    columns = tuple(columns or feature_columns())
    expected = list(IDENTITY) + list(columns)
    seen = set()
    by_node = {}
    for position, row in enumerate(feature_rows):
        actual = list(row.keys())
        if actual != expected:
            missing = [name for name in expected if name not in actual]
            extra = [name for name in actual if name not in expected]
            raise ValueError(
                f"D2 feature column contract mismatch at row {position}; missing={missing} extra={extra}"
            )
        try:
            node = int(row["node_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("D2 feature has a non-integer node_id") from exc
        if str(node) != str(row["node_id"]).strip():
            raise ValueError("D2 feature has a non-canonical node_id")
        if node < 0 or node >= len(gene_rows):
            raise ValueError("D2 feature node_id is outside the frozen graph")
        if node in seen:
            raise ValueError("D2 feature repeats node_id " + str(node))
        seen.add(node)
        gene = gene_rows[node]
        for field in ("node_id", "source_id", "hgnc_id", "symbol", "entrez_id", "mapping_status", "ambiguity_reason"):
            expected_value = str(gene.get(field, ""))
            if field == "node_id":
                expected_value = str(int(gene["node_id"]))
            if row.get(field, "") != expected_value:
                raise ValueError(f"D2 feature identity mismatch at node_id {node} field {field}")
        by_node[node] = row
    if len(by_node) != len(gene_rows):
        raise ValueError("D2 features do not cover every frozen node exactly once")
    return [by_node[node] for node in range(len(gene_rows))]


def analyze_missing_pattern(features, observed, missing_reasons, gene_rows, columns=None):
    """Return the confirmed fillable cell set or raise MissingPatternError."""
    columns = tuple(columns or feature_columns())
    features = align_feature_rows(features, gene_rows, columns)
    observed = align_feature_rows(observed, gene_rows, columns)

    blank_cells = []
    invalid = []
    for index, (feat, mask) in enumerate(zip(features, observed)):
        for column in columns:
            flag = mask[column]
            text = (feat.get(column) or "").strip()
            if flag not in {"0", "1"}:
                invalid.append((index, column, "bad_observed_flag", flag))
                continue
            if flag == "1":
                if not text:
                    invalid.append((index, column, "observed_but_blank", text))
                    continue
                try:
                    value = float(text)
                except ValueError:
                    invalid.append((index, column, "non_numeric", text))
                    continue
                if not math.isfinite(value) or value < 0 or value > 1:
                    invalid.append((index, column, "out_of_range", text))
                continue
            if text:
                invalid.append((index, column, "unobserved_but_nonempty", text))
                continue
            blank_cells.append((index, column))

    if invalid:
        sample = ", ".join(f"node={n}:{c}:{why}" for n, c, why, _ in invalid[:8])
        raise MissingPatternError("Invalid D2 values/mask: " + sample)

    reason_keys = set()
    for row in missing_reasons:
        try:
            node = int(row["node_id"])
        except (TypeError, ValueError) as exc:
            raise MissingPatternError("missing_reasons has non-integer node_id") from exc
        column = row["column"]
        reason = row.get("reason", "")
        if not reason.startswith("mapping_blocked:"):
            raise MissingPatternError("Unexpected missing reason (not mapping_blocked): " + reason)
        status = reason.split(":", 1)[1]
        if status not in ALLOWED_BLOCK_STATUSES:
            raise MissingPatternError("Unexpected mapping_blocked status: " + status)
        if gene_rows[node].get("mapping_status") != status:
            raise MissingPatternError(
                f"missing_reasons status {status} disagrees with gene_map at node_id {node}"
            )
        if column not in columns:
            raise MissingPatternError("missing_reasons column not in D2 schema: " + column)
        reason_keys.add((node, column))

    blank_set = set(blank_cells)
    if blank_set != reason_keys:
        only_blank = sorted(blank_set - reason_keys)[:5]
        only_reason = sorted(reason_keys - blank_set)[:5]
        raise MissingPatternError(
            "feature blanks and missing_reasons disagree; "
            f"only_blank={only_blank} only_reason={only_reason}"
        )

    blocked_nodes = sorted({node for node, _column in blank_cells})
    for node in blocked_nodes:
        status = gene_rows[node].get("mapping_status", "")
        if status not in ALLOWED_BLOCK_STATUSES:
            raise MissingPatternError(f"Blank cells on non-blocked node_id {node} status={status}")
        for column in columns:
            if (node, column) not in blank_set:
                raise MissingPatternError(
                    f"Blocked node_id {node} is missing only a subset of D2 columns"
                )
        # observed must be 0 across all D2 columns for blocked nodes
        for column in columns:
            if observed[node][column] != "0":
                raise MissingPatternError(f"Blocked node_id {node} has observed=1 on {column}")

    expected_cells = len(blocked_nodes) * len(columns)
    if len(blank_cells) != expected_cells:
        raise MissingPatternError(
            f"Expected {expected_cells} blank cells for full blocked-node rows, found {len(blank_cells)}"
        )

    # No blanks outside blocked nodes (already enforced by equality with reasons + status checks)
    return {
        "features": features,
        "observed": observed,
        "blank_cells": blank_cells,
        "blocked_nodes": [
            {
                "node_id": node,
                "source_id": gene_rows[node]["source_id"],
                "mapping_status": gene_rows[node]["mapping_status"],
                "ambiguity_reason": gene_rows[node].get("ambiguity_reason", ""),
                "n_cells": len(columns),
            }
            for node in blocked_nodes
        ],
        "n_blank_cells": len(blank_cells),
        "n_blocked_nodes": len(blocked_nodes),
        "columns": list(columns),
    }


def apply_zero_fill(analysis):
    """Return finite feature rows; only approved blanks become 0."""
    columns = analysis["columns"]
    fillable = set(analysis["blank_cells"])
    filled = []
    n_filled = 0
    for index, row in enumerate(analysis["features"]):
        out = dict(row)
        for column in columns:
            text = (out.get(column) or "").strip()
            if (index, column) in fillable:
                if text:
                    raise MissingPatternError("Refusing to overwrite a non-blank fill target")
                out[column] = "0"
                n_filled += 1
            else:
                if not text:
                    raise MissingPatternError("Unexpected blank outside approved fill set")
                value = float(text)
                if not math.isfinite(value) or value < 0 or value > 1:
                    raise MissingPatternError(f"Illegal rate at node_id {index} {column}")
        filled.append(out)
    if n_filled != analysis["n_blank_cells"]:
        raise MissingPatternError("Filled cell count mismatch")
    return filled, n_filled


def _edge_fingerprint(graph):
    import torch
    fingerprint = {}
    for canonical in graph.canonical_etypes:
        source, target = graph.edges(etype=canonical)
        if torch.is_tensor(source):
            source = source.cpu().tolist()
        if torch.is_tensor(target):
            target = target.cpu().tolist()
        fingerprint["|".join(canonical)] = [(int(a), int(b)) for a, b in zip(source, target)]
    return fingerprint


def _code_hashes(code_root=ROOT):
    code_root = Path(code_root)
    missing = [name for name in CODE_FILES if not (code_root / name).is_file()]
    if missing:
        raise ValueError("Cannot hash D2 build code: " + ", ".join(missing))
    return {name: sha256(code_root / name) for name in CODE_FILES}


def protected_snapshot(root, phase):
    """Hash D0/P0/D2 sealed inputs and key training code without mutating them."""
    root = Path(root)
    rows = []
    frozen, artifacts = _baseline_frozen(root)
    for path, digest in frozen.items():
        if path not in BASELINE_INPUTS:
            continue
        actual = sha256(root / path)
        rows.append({
            "label": "d0_input:" + path,
            "path": path,
            "expected_sha256": digest,
            "actual_sha256": actual,
            "match": actual == digest,
            "role": "hard",
        })
    for name in P0_CONTRACTS:
        path = "data_versions/p0_v1/" + name
        expected = artifacts[name]
        actual = sha256(root / path)
        rows.append({
            "label": "p0_artifact:" + name,
            "path": path,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
            "role": "hard",
        })
    for name in D2_AUDIT_COPIES:
        path = f"data_versions/{VERSION_ID}/{name}"
        target = root / path
        if not target.is_file():
            raise ValueError("D2 audit file missing: " + path)
        digest = sha256(target)
        rows.append({
            "label": "d2_readonly:" + name,
            "path": path,
            "expected_sha256": digest,
            "actual_sha256": digest,
            "match": True,
            "role": "d2_source",
        })
    for rel in ("run_ablation.py", "model.py", "fusion.py", "ablation_config.py"):
        digest = sha256(root / rel)
        rows.append({
            "label": "train_code:" + rel,
            "path": rel,
            "expected_sha256": digest,
            "actual_sha256": digest,
            "match": True,
            "role": "train_code",
        })
    hard_mismatches = [row["label"] for row in rows if row["role"] == "hard" and not row["match"]]
    return {
        "phase": phase,
        "generated_at": now(),
        "n_files": len(rows),
        "hard_mismatches": hard_mismatches,
        "files": rows,
    }


def build_variant(root=ROOT, variant_root=VARIANT, d2_out=D2_OUT, code_root=ROOT):
    root = Path(root).resolve()
    variant_root = Path(variant_root).resolve()
    d2_out = Path(d2_out).resolve()
    code_root = Path(code_root).resolve()

    protected = [root / "data", root / "Intermediate", root / "data_versions" / "p0_v1",
                 root / "data_versions" / VERSION_ID, root / "results"]
    if any(variant_root == path or path in variant_root.parents for path in protected):
        raise ValueError("Refusing to write the D2 variant over protected roots")
    if variant_root.exists():
        raise ValueError("Variant root already exists; refusing to overwrite: " + str(variant_root))

    before = protected_snapshot(root, "before")
    if before["hard_mismatches"]:
        raise ValueError("Hard-protected inputs changed before build: " + ", ".join(before["hard_mismatches"]))

    mapping, gene_rows = load_identity(root)
    features = read_tsv(d2_out / "mutation_type_features.tsv")
    observed = read_tsv(d2_out / "feature_observed.tsv")
    missing_reasons = read_tsv(d2_out / "missing_reasons.tsv")
    analysis = analyze_missing_pattern(features, observed, missing_reasons, gene_rows)
    filled_rows, n_filled = apply_zero_fill(analysis)
    columns = analysis["columns"]

    import torch
    import dgl

    source_graph_path = root / "data/network/hetero/new_9nodes_graph.bin"
    source_graphs, labels = dgl.load_graphs(str(source_graph_path))
    if len(source_graphs) != 1:
        raise ValueError("D0 graph file does not contain exactly one graph")
    source_graph = source_graphs[0]
    if source_graph.num_nodes("Gene") != len(gene_rows):
        raise ValueError("D0 Gene node count differs from gene_map")
    old = source_graph.nodes["Gene"].data["feature"].float().cpu()
    if old.ndim != 2 or old.shape != (len(gene_rows), 48) or not torch.isfinite(old).all():
        raise ValueError("D0 Gene.feature is not a finite 12944×48 matrix")
    schema_old = read_tsv(root / "data_versions/p0_v1/feature_schema.tsv")
    if len(schema_old) != 48 or [int(row["position"]) for row in schema_old] != list(range(48)):
        raise ValueError("P0 feature_schema does not match the D0 feature width")

    extra = torch.tensor([[float(row[column]) for column in columns] for row in filled_rows], dtype=torch.float32)
    if tuple(extra.shape) != (len(gene_rows), 48) or not torch.isfinite(extra).all():
        raise ValueError("Filled D2 block is not finite 12944×48")
    if ((extra < 0) | (extra > 1)).any():
        raise ValueError("Filled D2 block has values outside [0,1]")

    # Observed cells must equal original numeric strings converted to float32 policy
    for index, column_i in enumerate(columns):
        mask_col = torch.tensor([analysis["observed"][r][column_i] == "1" for r in range(len(gene_rows))])
        original_vals = []
        for r in range(len(gene_rows)):
            text = (analysis["features"][r][column_i] or "").strip()
            original_vals.append(float(text) if text else float("nan"))
        original = torch.tensor(original_vals, dtype=torch.float32)
        if not torch.equal(extra[mask_col, index], original[mask_col]):
            raise ValueError(f"Zero-fill changed an observed D2 value in column {column_i}")
        fill_mask = ~mask_col
        if fill_mask.any() and not torch.all(extra[fill_mask, index] == 0):
            raise ValueError(f"Fill targets are not all zero in column {column_i}")

    before_edges = _edge_fingerprint(source_graph)
    before_nodes = {ntype: source_graph.num_nodes(ntype) for ntype in source_graph.ntypes}
    graph = source_graph.clone()
    graph.nodes["Gene"].data["feature"] = torch.cat([old, extra], dim=1)

    variant_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d2_mutation_zero_v1_build_", dir=str(variant_root.parent)) as temporary:
        temp_root = Path(temporary)
        out_graph = temp_root / "data/network/hetero/new_9nodes_graph.bin"
        out_graph.parent.mkdir(parents=True, exist_ok=True)
        dgl.save_graphs(str(out_graph), [graph], labels)

        linked = []
        for relative in DATA_INPUTS:
            record = _link_or_copy(root / relative, temp_root / relative)
            record["path"] = relative
            linked.append(record)

        for name in ("gene_map.tsv", "metapaths.json", "split_manifest.json"):
            record = _link_or_copy(root / "data_versions/p0_v1" / name, temp_root / name)
            record["path"] = name
            linked.append(record)

        # Processed finite D2 features + original observed mask (unchanged)
        write_tsv(temp_root / "mutation_type_features_zero_filled.tsv", filled_rows, list(IDENTITY) + list(columns))
        write_tsv(temp_root / "feature_observed.tsv", analysis["observed"], list(IDENTITY) + list(columns))
        write_tsv(temp_root / "missing_reasons.tsv", missing_reasons,
                  ["node_id", "column", "reason", "n_event_patients", "n_callable_patients"])

        schema = [dict(row) for row in schema_old]
        d2_schema_src = read_tsv(d2_out / "feature_schema.tsv")
        by_name = {row["column_name"]: row for row in d2_schema_src}
        for offset, column in enumerate(columns):
            src = by_name[column]
            schema.append({
                "position": 48 + offset,
                "column_name": column,
                "modality": "mutation_type_frequency",
                "cancer_type": src.get("cancer_type", ""),
                "event_class": src.get("event_class", ""),
                "source": f"data_versions/{VERSION_ID}/mutation_type_features.tsv",
                "source_release": f"datahub_commit_{COMMIT}",
                "unit": "patient_fraction",
                "missing_rule": "mapping_blocked_identity_zero_placeholder_v1",
                "missing_placeholder_meaning": "zero_is_missing_placeholder_not_confirmed_zero_mutation",
            })
        if len(schema) != 96 or [int(row["position"]) for row in schema] != list(range(96)):
            raise ValueError("Combined feature schema is not contiguous 0..95")
        schema_fields = [
            "position", "column_name", "modality", "cancer_type", "event_class",
            "source", "source_release", "unit", "missing_rule", "missing_placeholder_meaning",
        ]
        # D0 schema rows may lack event_class / missing_placeholder_meaning
        for row in schema:
            for field in schema_fields:
                row.setdefault(field, "")
        write_tsv(temp_root / "feature_schema.tsv", schema, schema_fields)

        protocol = {
            "version": "d2_mapping_blocked_zero_v1",
            "created_at": now(),
            "policy": (
                "Fill D2 MUT_* blanks with 0 only for nodes whose gene_map mapping_status is "
                "unmapped, ambiguous, or duplicate_identity, and only when feature_observed=0 "
                "and missing_reasons reason starts with mapping_blocked. Do not impute other "
                "missingness. Zero is a missingness placeholder, not evidence of zero mutation "
                "rate. Keep the original feature_observed mask unchanged (still 0 on filled cells). "
                "Do not concatenate the mask into model inputs. Final Gene.feature is "
                "[D0 48 dims | D2 48 dims] = 96."
            ),
            "fill_value": 0.0,
            "fill_semantics": "missing_placeholder_not_biological_zero",
            "mask_concatenated_into_model_input": False,
            "d0_features_modified": False,
            "observed_d2_values_modified": False,
            "blocked_nodes": analysis["blocked_nodes"],
            "n_cells_filled": n_filled,
            "expected_cells_filled": 240,
            "approved_only_if_n_cells_filled_equals_240_and_pattern_matches": True,
        }
        if n_filled != 240:
            raise MissingPatternError(f"Refusing release: filled {n_filled} cells, expected exactly 240")
        write_json(temp_root / "missingness_protocol.json", protocol)

        # Verify reloaded graph before publish
        loaded, _ = dgl.load_graphs(str(out_graph))
        if len(loaded) != 1:
            raise ValueError("Saved variant does not contain exactly one graph")
        new_graph = loaded[0]
        combined = new_graph.nodes["Gene"].data["feature"].float().cpu()
        if tuple(combined.shape) != (len(gene_rows), 96) or not torch.isfinite(combined).all():
            raise ValueError("Saved variant feature shape is invalid")
        if not torch.equal(combined[:, :48], old):
            raise ValueError("Saved variant changed the original D0 features")
        if not torch.equal(combined[:, 48:], extra):
            raise ValueError("Saved variant changed the D2 block")
        if ((combined[:, 48:] < 0) | (combined[:, 48:] > 1)).any():
            raise ValueError("Saved D2 block outside [0,1]")
        if _edge_fingerprint(new_graph) != before_edges:
            raise ValueError("Saved variant changed edge topology")
        if {ntype: new_graph.num_nodes(ntype) for ntype in new_graph.ntypes} != before_nodes:
            raise ValueError("Saved variant changed node counts")

        d2_files = []
        for name in D2_AUDIT_COPIES:
            source = d2_out / name
            target = temp_root / "d2_source" / name
            record = _link_or_copy(source, target)
            record["path"] = "d2_source/" + name
            d2_files.append(record)

        outputs = [{
            "path": "data/network/hetero/new_9nodes_graph.bin",
            "sha256": sha256(out_graph),
            "size_bytes": out_graph.stat().st_size,
            "mode": "written",
        }]
        for record in linked + d2_files:
            outputs.append({
                "path": record["path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "mode": record["mode"],
                "samefile": record["samefile"],
                "source": record["source"],
            })
        for name in (
            "feature_schema.tsv",
            "mutation_type_features_zero_filled.tsv",
            "feature_observed.tsv",
            "missing_reasons.tsv",
            "missingness_protocol.json",
        ):
            path = temp_root / name
            outputs.append({"path": name, "sha256": sha256(path), "size_bytes": path.stat().st_size, "mode": "written"})

        quality = {
            "variant": VARIANT_NAME,
            "created_at": now(),
            "feature_shape": [len(gene_rows), 96],
            "d0_width": 48,
            "d2_width": 48,
            "n_blank_cells_before_fill": analysis["n_blank_cells"],
            "n_cells_filled": n_filled,
            "blocked_nodes": analysis["blocked_nodes"],
            "ready_for_training_loader": True,
            "ready_for_training_run": False,
            "note": "Loader-ready 96-dim graph packaged; training not started by this module.",
            "d2_source_version": VERSION_ID,
            "datahub_commit": COMMIT,
        }
        write_json(temp_root / "quality_report.json", quality)
        outputs.append({
            "path": "quality_report.json",
            "sha256": sha256(temp_root / "quality_report.json"),
            "size_bytes": (temp_root / "quality_report.json").stat().st_size,
            "mode": "written",
        })

        write_json(temp_root / "protected_files_before.json", before)

        manifest = {
            "schema_version": 1,
            "created_at": now(),
            "variant": VARIANT_NAME,
            "status": "finite_96d_graph_packaged",
            "missingness_protocol": protocol,
            "code_sha256": _code_hashes(code_root),
            "d0_baseline_inputs": [
                {"path": row["path"], "sha256": row["actual_sha256"]}
                for row in before["files"] if row["label"].startswith("d0_input:")
            ],
            "d0_inputs": [row for row in before["files"] if row["label"].startswith("d0_input:")],
            "d2_audit_files": d2_files,
            "input_output_relation": (
                "48 D0 columns preserved byte-for-byte in Gene.feature[:, :48]; "
                "observed D2 cells preserved; only mapping_blocked blanks filled with 0; "
                "feature_observed unchanged; mask not concatenated."
            ),
            "outputs": outputs,
            "quality": quality,
        }
        write_json(temp_root / "manifest.json", manifest)
        write_json(temp_root / "release_notes.json", {
            "variant": VARIANT_NAME,
            "how_to_load": "python run_ablation.py --data-root data_variants/d2_mutation_zero_v1 ...",
            "do_not_train_from_this_script": True,
        })

        (temp_root / "README.md").write_text(
            "\n".join([
                "# d2_mutation_zero_v1",
                "",
                "D0 48 + D2 48 = 96 dim Gene.feature. Mapping-blocked D2 blanks filled with 0 as missing placeholders.",
                "Original `feature_observed.tsv` keeps those cells as unobserved (0).",
                "Reusable blocks/splits/gene_toids are hard-linked or copied from D0; do not write through them.",
                "",
                "Load-only check:",
                "",
                "```powershell",
                "python -c \"from run_ablation import load_data; from pathlib import Path; x,g,s,f=load_data(Path('data_variants/d2_mutation_zero_v1')); print(tuple(x.shape), len(g), len(s))\"",
                "```",
                "",
            ]),
            encoding="utf-8",
        )

        if variant_root.exists():
            raise ValueError("Variant root appeared during build; refusing to replace it")
        # Move contents: TemporaryDirectory will delete temp_root; move the directory itself.
        shutil.move(str(temp_root), str(variant_root))
        # TemporaryDirectory cleanup: suppress by replacing with empty? After move, temp path is gone.
        # Create a decoy so cleanup succeeds
        Path(temporary).mkdir(parents=True, exist_ok=True)

    after = protected_snapshot(root, "after")
    # Compare hard + d2_source hashes against before
    before_by_label = {row["label"]: row for row in before["files"]}
    soft = []
    hard = []
    for row in after["files"]:
        prev = before_by_label.get(row["label"])
        if not prev:
            continue
        if row["actual_sha256"] != prev["actual_sha256"]:
            if row["role"] == "hard" or row["label"].startswith("d2_readonly:") or row["label"].startswith("train_code:"):
                hard.append(row["label"])
            else:
                soft.append(row["label"])
    after["compared_to_before_hard_or_source_mismatches"] = hard
    after["compared_to_before_soft_mismatches"] = soft
    write_json(variant_root / "protected_files_after.json", after)
    if hard or after["hard_mismatches"]:
        raise ValueError("Protected inputs changed during/after build: " + ", ".join(hard or after["hard_mismatches"]))

    # Ensure before snapshot is present in published root
    if not (variant_root / "protected_files_before.json").is_file():
        write_json(variant_root / "protected_files_before.json", before)

    return {
        "variant_root": str(variant_root),
        "feature_shape": [len(gene_rows), 96],
        "n_cells_filled": n_filled,
        "blocked_nodes": analysis["blocked_nodes"],
        "protected_after_mismatches": hard,
    }


def verify_loader(variant_root=VARIANT):
    """Load via run_ablation.load_data without training or GPU allocation for the model."""
    import torch
    import run_ablation

    # run_ablation lazily binds torch/dgl inside dependencies(); call it before load_data.
    run_ablation.dependencies()
    variant_root = Path(variant_root).resolve()
    x, graphs, splits, fingerprint = run_ablation.load_data(variant_root, smoke=False)
    if tuple(x.shape) != (12944, 96):
        raise ValueError(f"load_data returned shape {tuple(x.shape)}, expected (12944, 96)")
    if not torch.isfinite(x).all():
        raise ValueError("load_data features are non-finite")
    if ((x[:, 48:] < 0) | (x[:, 48:] > 1)).any():
        raise ValueError("load_data D2 block outside [0,1]")
    if len(graphs) != 9:
        raise ValueError("Expected 9 metapath graphs")
    if len(splits) != 5:
        raise ValueError("Expected 5 folds")
    return {
        "feature_shape": list(x.shape),
        "n_graphs": len(graphs),
        "n_folds": len(splits),
        "fingerprint_files": sorted(fingerprint),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m data_processing.d2.build_graph")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_assess = sub.add_parser("assess", help="Read-only check of the D2 missing pattern")
    p_assess.set_defaults(func="assess")
    p_build = sub.add_parser("build", help="Build data_variants/d2_mutation_zero_v1")
    p_build.add_argument("--variant-root", type=Path, default=VARIANT)
    p_build.set_defaults(func="build")
    p_verify = sub.add_parser("verify-loader", help="Load the variant with run_ablation.load_data (no training)")
    p_verify.add_argument("--variant-root", type=Path, default=VARIANT)
    p_verify.set_defaults(func="verify")
    args = parser.parse_args(argv)

    if args.func == "assess":
        mapping, gene_rows = load_identity(ROOT)
        analysis = analyze_missing_pattern(
            read_tsv(D2_OUT / "mutation_type_features.tsv"),
            read_tsv(D2_OUT / "feature_observed.tsv"),
            read_tsv(D2_OUT / "missing_reasons.tsv"),
            gene_rows,
        )
        print(json.dumps({
            "n_blank_cells": analysis["n_blank_cells"],
            "n_blocked_nodes": analysis["n_blocked_nodes"],
            "blocked_nodes": analysis["blocked_nodes"],
            "pattern_ok_for_zero_fill": analysis["n_blank_cells"] == 240,
        }, ensure_ascii=False, indent=2))
        return
    if args.func == "build":
        result = build_variant(variant_root=args.variant_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.func == "verify":
        result = verify_loader(args.variant_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
