"""Build a CNV graph variant only for an explicit, finite missing policy.

The current D1 table has 1,860 blank cells and no selected policy, so this
module must refuse it. It never fills blanks, never trains and never edits D0.
"""
import argparse
import json
import math
import pickle
import shutil
import tempfile
from pathlib import Path

from data_processing.p0.common import ROOT, now, read_tsv, sha256, write_json, write_tsv

from .spec import COMMIT, feature_columns
from .tables import CNV_FEATURE_COLUMNS, IDENTITY, read_d1_tsv

VARIANT = ROOT / "data_variants" / "cnv_v1"
DATA_INPUTS = (
    "data/step1/gene_toids.pkl",
    "Intermediate/label_set/train_batch.pkl",
    *(f"Intermediate/blocks/{index}gs.bin" for index in range(9)),
)
BASELINE_INPUTS = (
    "data/network/hetero/new_9nodes_graph.bin",
    *DATA_INPUTS,
)
P0_ARTIFACTS = (
    "gene_map.tsv",
    "feature_schema.tsv",
    "metapaths.json",
    "split_manifest.json",
)


def assess(feature_rows, columns=None):
    columns = tuple(columns or feature_columns())
    missing = invalid = 0
    for row in feature_rows:
        for column in columns:
            text = (row.get(column) or "").strip()
            if not text:
                missing += 1
                continue
            try:
                value = float(text)
            except ValueError:
                invalid += 1
                continue
            if not math.isfinite(value) or value < 0 or value > 1:
                invalid += 1
    return {"rows": len(feature_rows), "columns": len(columns), "missing": missing, "invalid": invalid}


def assert_ready(feature_rows, columns=None):
    report = assess(feature_rows, columns)
    if report["missing"] or report["invalid"] or report["columns"] != 32:
        raise ValueError(
            "Refusing to build a training graph: CNV input is not a finite 32-column matrix "
            f"(missing={report['missing']}, invalid={report['invalid']}, columns={report['columns']}). "
            "Choose an explicit missing policy before packaging a graph variant."
        )
    return report


def _baseline_frozen(root):
    """Check raw D0 inputs from baseline and P0 audit artifacts from layout."""
    root = Path(root)
    baseline_path = root / "data_versions/p0_v1/baseline_manifest.json"
    layout_path = root / "data_versions/p0_v1/layout_manifest.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    frozen = {row["path"]: row["sha256"] for row in baseline["files"] if row["kind"] == "data"}
    missing = [path for path in BASELINE_INPUTS if path not in frozen]
    if missing:
        raise ValueError("D0 baseline manifest does not pin raw inputs: " + ", ".join(missing))
    mismatches = [path for path in BASELINE_INPUTS
                  if not (root / path).is_file() or sha256(root / path) != frozen[path]]
    if mismatches:
        raise ValueError("D0 raw input hash mismatch: " + ", ".join(mismatches))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    artifacts = {row["path"]: row["sha256"] for row in layout["artifacts"]}
    missing_artifacts = [name for name in P0_ARTIFACTS if name not in artifacts]
    if missing_artifacts:
        raise ValueError("P0 layout manifest does not pin audit artifacts: " + ", ".join(missing_artifacts))
    artifact_mismatches = []
    for name in P0_ARTIFACTS:
        target = root / "data_versions/p0_v1" / name
        if not target.is_file() or sha256(target) != artifacts[name]:
            artifact_mismatches.append(name)
    if artifact_mismatches:
        raise ValueError("P0 sealed artifact hash mismatch: " + ", ".join(artifact_mismatches))
    return frozen


def _load_identity(root):
    _baseline_frozen(root)
    with (Path(root) / "data/step1/gene_toids.pkl").open("rb") as handle:
        mapping = pickle.load(handle)
    if len(mapping) != len(set(mapping)) or not all(isinstance(name, str) and name for name in mapping):
        raise ValueError("gene_toids.pkl has invalid identifiers")
    if any(type(node) is not int for node in mapping.values()):
        raise ValueError("gene_toids.pkl node ids are not integers")
    genes = sorted(mapping, key=mapping.get)
    if [mapping[name] for name in genes] != list(range(len(genes))):
        raise ValueError("gene_toids.pkl is not unique and contiguous")
    rows = read_tsv(Path(root) / "data_versions/p0_v1/gene_map.tsv")
    if len(rows) != len(genes):
        raise ValueError("P0 gene_map.tsv and gene_toids.pkl disagree")
    for index, (name, row) in enumerate(zip(genes, rows)):
        if int(row["node_id"]) != index or row["source_id"] != name:
            raise ValueError("P0 gene_map.tsv is not in frozen node order")
    return mapping, rows


def align_feature_rows(feature_rows, gene_rows, columns=None):
    """Validate all identity fields and reorder by node_id. Never trust row order."""
    columns = tuple(columns or feature_columns())
    expected = list(IDENTITY + columns)
    seen = set()
    by_node = {}
    for position, row in enumerate(feature_rows):
        if len(set(row.keys())) != len(row.keys()):
            raise ValueError("CNV input repeats a column")
        actual = list(row.keys())
        if actual != expected:
            missing = [name for name in expected if name not in actual]
            extra = [name for name in actual if name not in expected]
            raise ValueError(f"CNV input column contract mismatch at row {position}; missing={missing} extra={extra}")
        try:
            node = int(row["node_id"])
        except (TypeError, ValueError):
            raise ValueError("CNV input has a non-integer node_id") from None
        if str(node) != str(row["node_id"]).strip():
            raise ValueError("CNV input has a non-canonical node_id")
        if node < 0 or node >= len(gene_rows):
            raise ValueError("CNV input node_id is outside the frozen graph")
        if node in seen:
            raise ValueError("CNV input repeats node_id " + str(node))
        seen.add(node)
        expected_identity = {
            "node_id": str(int(gene_rows[node]["node_id"])),
            "source_id": gene_rows[node].get("source_id", ""),
            "hgnc_id": gene_rows[node].get("hgnc_id", ""),
            "symbol": gene_rows[node].get("symbol", ""),
            "entrez_id": gene_rows[node].get("entrez_id", ""),
        }
        for field, expected_value in expected_identity.items():
            if row[field] != expected_value:
                raise ValueError(f"CNV input identity mismatch at node_id {node} field {field}")
        by_node[node] = row
    if len(by_node) != len(gene_rows):
        raise ValueError("CNV input does not cover every frozen node exactly once")
    return [by_node[node] for node in range(len(gene_rows))]


def _parse_policy(policy, policy_path):
    if policy is not None and policy_path is not None:
        raise ValueError("Use either --missing-policy or --missing-policy-file, not both")
    if policy_path is not None:
        policy = Path(policy_path).read_text(encoding="utf-8")
    if policy is None or not policy.strip():
        raise ValueError("A finite matrix is not enough: provide a traceable missing-value policy")
    policy = policy.strip()
    if len(policy) < 12 or len(policy) > 100_000:
        raise ValueError("Missing-value policy is missing or implausibly sized")
    return policy


def _validate_values(aligned, columns):
    for index, row in enumerate(aligned):
        for column in columns:
            text = (row.get(column) or "").strip()
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"CNV input is non-numeric at node_id {index} column {column}") from None
            if not math.isfinite(value) or value < 0 or value > 1:
                raise ValueError(f"CNV input is outside [0,1] at node_id {index} column {column}")
        for column in columns:
            if column.startswith("CNV_AMP:"):
                partner = column.replace("CNV_AMP:", "CNV_DEL:")
                if float(row[column]) + float(row[partner]) > 1 + 1e-12:
                    raise ValueError("AMP plus DEL exceeds 1 at node_id " + str(index))


def _same_bytes(left, right):
    return Path(left).is_file() and Path(right).is_file() and sha256(left) == sha256(right)


def _code_hashes(code_root=ROOT):
    from .pipeline import CODE_FILES
    code_root = Path(code_root)
    missing = [name for name in CODE_FILES if not (code_root / name).is_file()]
    if missing:
        raise ValueError("Cannot hash the D1 processing code under " + str(code_root) + ": " + ", ".join(missing))
    return {name: sha256(code_root / name) for name in CODE_FILES}


def _verify_d1_source_bundle(d1_out, gene_rows):
    """Verify D1 manifests and original tables before any graph output is staged."""
    from .pipeline import _load_manifest_records, verify_sources
    d1_out = Path(d1_out)
    manifest, source = _load_manifest_records(d1_out)
    synthetic = bool(source.get("synthetic"))
    if not synthetic and manifest.get("datahub_commit") != COMMIT:
        raise ValueError("D1 manifest does not match the pinned source commit")
    verify_sources(d1_out, source, allow_synthetic=synthetic)
    original = read_d1_tsv(d1_out / "cnv_features.tsv")
    observed = read_d1_tsv(d1_out / "feature_observed.tsv")
    aligned_original = align_feature_rows(original, gene_rows)
    aligned_observed = align_feature_rows(observed, gene_rows)
    if len(aligned_observed) != len(gene_rows):
        raise ValueError("D1 observed mask does not cover every frozen node")
    columns = feature_columns()
    for index, (original_row, mask_row) in enumerate(zip(aligned_original, aligned_observed)):
        for field in IDENTITY:
            if original_row[field] != mask_row[field]:
                raise ValueError(f"D1 observed identity mismatch at node_id {index} field {field}")
        for column in columns:
            flag = mask_row[column]
            if flag not in {"0", "1"}:
                raise ValueError(f"D1 observed mask is not 0/1 at node_id {index} column {column}")
            if (flag == "1") != bool(original_row[column].strip()):
                raise ValueError(f"D1 observed mask conflicts with original CNV at node_id {index} column {column}")
    required_outputs = {row["path"] for row in manifest.get("outputs", [])}
    for name in ("cnv_features.tsv", "feature_observed.tsv", "source_manifest.json"):
        if name not in required_outputs:
            raise ValueError("D1 manifest does not register " + name)
    registered = {row["path"]: row for row in manifest.get("outputs", [])}
    for name in ("cnv_features.tsv", "feature_observed.tsv", "source_manifest.json"):
        path = d1_out / name
        row = registered[name]
        if path.stat().st_size != row.get("size_bytes") or sha256(path) != row.get("sha256"):
            raise ValueError("D1 registered audit file hash mismatch: " + name)
    return {"manifest": manifest, "source": source, "original": aligned_original, "observed": aligned_observed}


def _same_numeric(left, right, tolerance=1e-12):
    try:
        left_value = float(left)
        right_value = float(right)
    except ValueError:
        return False
    return math.isfinite(left_value) and math.isfinite(right_value) and abs(left_value - right_value) <= tolerance


def _check_finite_against_d1(aligned, bundle):
    """Observed cells must keep original values; missing cells are the only policy targets."""
    columns = feature_columns()
    changed_observed = []
    filled_missing = []
    for index, row in enumerate(aligned):
        original_row = bundle["original"][index]
        mask_row = bundle["observed"][index]
        for column in columns:
            if mask_row[column] == "1":
                if not _same_numeric(row[column], original_row[column]):
                    changed_observed.append((index, column))
            else:
                if not row[column].strip():
                    raise ValueError(f"Finite input still blank at missing node_id {index} column {column}")
                filled_missing.append((index, column))
    if changed_observed:
        index, column = changed_observed[0]
        raise ValueError(f"Finite input changed an observed CNV value at node_id {index} column {column}")
    return {"observed_cells_checked": sum(1 for row in bundle["observed"] for column in columns if row[column] == "1"),
            "missing_cells_filled": len(filled_missing)}


def _canonical_edge_tuple(value):
    try:
        source, target = int(value[0]), int(value[1])
    except Exception:
        return repr(value)
    return (source, target)


def _edge_fingerprint(graph):
    import torch
    fingerprint = {}
    for canonical in graph.canonical_etypes:
        source, target = graph.edges(etype=canonical)
        if torch.is_tensor(source):
            source = source.cpu().tolist()
        if torch.is_tensor(target):
            target = target.cpu().tolist()
        fingerprint["|".join(canonical)] = [_canonical_edge_tuple(edge) for edge in zip(source, target)]
    return fingerprint


def build_variant(feature_rows, root=ROOT, variant_root=VARIANT, missing_policy=None, missing_policy_path=None,
                  code_root=ROOT, d1_out=None, finite_input_path=None):
    """Package an isolated 80-column graph in a temporary directory first."""
    policy = _parse_policy(missing_policy, missing_policy_path)
    root = Path(root).resolve()
    variant_root = Path(variant_root).resolve()
    d1_out = Path(d1_out).resolve() if d1_out is not None else root / "data_versions" / "d1_cnv_v1"
    protected = [root / "data", root / "Intermediate", root / "data_versions" / "p0_v1", root / "results"]
    if any(variant_root == path or path in variant_root.parents for path in protected):
        raise ValueError("Refusing to write the CNV variant over D0, P0 or results")
    if variant_root.exists():
        raise ValueError("Variant root already exists; refusing to overwrite it")
    variant_root.parent.mkdir(parents=True, exist_ok=True)
    columns = feature_columns()
    _mapping, gene_rows = _load_identity(root)
    bundle = _verify_d1_source_bundle(d1_out, gene_rows)
    assert_ready(feature_rows, columns)
    aligned = align_feature_rows(feature_rows, gene_rows, columns)
    _validate_values(aligned, columns)
    relation_check = _check_finite_against_d1(aligned, bundle)

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
    if old.ndim != 2 or old.shape[0] != len(gene_rows) or not torch.isfinite(old).all():
        raise ValueError("D0 Gene.feature is not a finite matrix with one row per frozen node")
    schema_old = read_tsv(root / "data_versions/p0_v1/feature_schema.tsv")
    old_width = old.shape[1]
    if old_width != 48:
        raise ValueError(f"D1 requires the original 48-column D0 feature, found {old_width}")
    if len(schema_old) != old_width or [int(row["position"]) for row in schema_old] != list(range(old_width)):
        raise ValueError("P0 feature_schema does not match the D0 feature width")
    extra = torch.tensor([[float(row[column]) for column in columns] for row in aligned], dtype=torch.float32)
    if tuple(extra.shape) != (len(gene_rows), 32) or not torch.isfinite(extra).all():
        raise ValueError("CNV block is not finite")
    before_edges = _edge_fingerprint(source_graph)
    before_nodes = {ntype: source_graph.num_nodes(ntype) for ntype in source_graph.ntypes}
    graph = source_graph.clone()
    graph.nodes["Gene"].data["feature"] = torch.cat([old, extra], dim=1)

    with tempfile.TemporaryDirectory(prefix="cnv_v1_build_", dir=str(variant_root.parent)) as temporary:
        temp_root = Path(temporary)
        out_graph = temp_root / "data/network/hetero/new_9nodes_graph.bin"
        out_graph.parent.mkdir(parents=True, exist_ok=True)
        dgl.save_graphs(str(out_graph), [graph], labels)
        copied = []
        for relative in DATA_INPUTS:
            source = root / relative
            target = temp_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if not _same_bytes(source, target):
                raise ValueError("Copied D0 file hash changed: " + relative)
            copied.append({"path": relative, "sha256": sha256(source), "size_bytes": source.stat().st_size})
        for name in ("gene_map.tsv", "metapaths.json", "split_manifest.json"):
            source = root / "data_versions/p0_v1" / name
            target = temp_root / name
            shutil.copyfile(source, target)
            if not _same_bytes(source, target):
                raise ValueError("Copied P0 contract hash changed: " + name)
            copied.append({"path": name, "sha256": sha256(source), "size_bytes": source.stat().st_size})

        schema = []
        for row in schema_old:
            schema.append(dict(row))
        for offset, column in enumerate(columns):
            kind, cancer = column.split(":")
            schema.append({
                "position": old_width + offset,
                "column_name": column,
                "modality": kind,
                "cancer_type": cancer,
                "source": "data_versions/d1_cnv_v1/cnv_features.tsv",
                "source_release": f"datahub_commit_{COMMIT}",
                "missing_rule": policy,
                "unit": "fraction_of_selected_patients",
            })
        if len(schema) != old_width + 32 or [int(row["position"]) for row in schema] != list(range(old_width + 32)):
            raise ValueError("Combined feature schema does not match D0 plus 32 CNV columns")
        schema_fields = ["position", "column_name", "modality", "cancer_type", "source", "source_release", "missing_rule", "unit"]
        write_tsv(temp_root / "feature_schema.tsv", schema, schema_fields)

        policy_payload = {
            "version": "cnv_missing_policy_v1",
            "created_at": now(),
            "policy": policy,
            "source": "caller_supplied_via_cli",
            "input_columns": list(columns),
        }
        write_json(temp_root / "missing_policy.json", policy_payload)

        loaded, _loaded_labels = dgl.load_graphs(str(out_graph))
        if len(loaded) != 1:
            raise ValueError("Saved variant does not contain exactly one graph")
        new_graph = loaded[0]
        combined = new_graph.nodes["Gene"].data["feature"].float().cpu()
        if tuple(combined.shape) != (len(gene_rows), old_width + 32) or not torch.isfinite(combined).all():
            raise ValueError("Saved variant feature shape is invalid")
        if not torch.equal(combined[:, :old_width], old):
            raise ValueError("Saved variant changed the original D0 features")
        if not torch.equal(combined[:, old_width:], extra):
            raise ValueError("Saved variant changed the CNV block")
        if _edge_fingerprint(new_graph) != before_edges:
            raise ValueError("Saved variant changed edge topology")
        if {ntype: new_graph.num_nodes(ntype) for ntype in new_graph.ntypes} != before_nodes:
            raise ValueError("Saved variant changed node counts")

        d1_files = []
        for name in ("source_manifest.json", "manifest.json", "feature_observed.tsv", "cnv_features.tsv"):
            source = d1_out / name
            if not source.is_file():
                raise ValueError("D1 audit input missing: " + name)
            target = temp_root / "d1" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if not _same_bytes(source, target):
                raise ValueError("Copied D1 audit file hash changed: " + name)
            d1_files.append({"path": "d1/" + name, "sha256": sha256(source), "size_bytes": source.stat().st_size})
        finite_payload = {
            "schema_version": 1,
            "rows": len(aligned),
            "columns": list(columns),
            "values": [[row[column] for column in columns] for row in aligned],
            "relation_check": relation_check,
        }
        write_json(temp_root / "finite_cnv_input.json", finite_payload)
        outputs = [{"path": "data/network/hetero/new_9nodes_graph.bin", "sha256": sha256(out_graph), "size_bytes": out_graph.stat().st_size}]
        outputs += copied
        outputs += d1_files
        for name in ("feature_schema.tsv", "missing_policy.json", "finite_cnv_input.json"):
            path = temp_root / name
            outputs.append({"path": name, "sha256": sha256(path), "size_bytes": path.stat().st_size})
        manifest = {
            "schema_version": 1,
            "created_at": now(),
            "variant": "cnv_v1",
            "status": "finite_80d_graph_packaged",
            "datahub_commit": COMMIT,
            "missing_policy": policy_payload,
            "code_sha256": _code_hashes(code_root),
            "d0_baseline_inputs": [{"path": path, "sha256": digest} for path, digest in _baseline_frozen(root).items() if path in BASELINE_INPUTS],
            "d1_audit_files": d1_files,
            "finite_input_path": str(finite_input_path) if finite_input_path else None,
            "finite_input_relation": relation_check,
            "input_output_relation": "48 D0 columns preserved; observed CNV cells preserved; only originally missing cells were supplied by the explicit policy",
            "outputs": outputs,
        }
        write_json(temp_root / "manifest.json", manifest)
        if variant_root.exists():
            raise ValueError("Variant root appeared during build; refusing to replace it")
        shutil.move(str(temp_root), str(variant_root))
    return {"variant_root": str(variant_root), "feature_shape": [len(gene_rows), old_width + 32], "policy_sha256": sha256(variant_root / "missing_policy.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data_versions/d1_cnv_v1/cnv_features.tsv")
    parser.add_argument("--write-variant", action="store_true", help="Package a variant only with --missing-policy(-file)")
    parser.add_argument("--missing-policy", type=str, default=None)
    parser.add_argument("--missing-policy-file", type=Path, default=None)
    parser.add_argument("--variant-root", type=Path, default=VARIANT)
    args = parser.parse_args()
    rows = read_d1_tsv(args.features)
    try:
        _mapping, gene_rows = _load_identity(ROOT)
        aligned = align_feature_rows(rows, gene_rows)
        report = assert_ready(aligned)
    except (ValueError, KeyError) as exc:
        print(exc)
        raise SystemExit(2) from None
    if not args.write_variant:
        print("Feature table identity and frozen mapping passed; no variant was written. Pass --write-variant and a missing policy to package it.")
        return
    result = build_variant(rows, variant_root=args.variant_root, missing_policy=args.missing_policy,
                           missing_policy_path=args.missing_policy_file, finite_input_path=args.features)
    print(result)
    print(report)


if __name__ == "__main__":
    main()
