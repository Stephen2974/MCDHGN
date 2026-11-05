"""Discrete GISTIC parsing, duplicate-row checks and frequency counts."""
import hashlib
from decimal import Decimal

from .select import is_missing_token
from .spec import GISTIC_VALUES

ABSENT = -1
ILLEGAL_LIMIT = 0.001


class CnaFormatError(ValueError):
    pass


def parse_gistic(token):
    text = (token or "").strip()
    if is_missing_token(text):
        return None
    if text.startswith("+"):
        text = text[1:]
    if text in {str(value) for value in GISTIC_VALUES}:
        return int(text)
    raise CnaFormatError("illegal GISTIC value: " + (token or "").strip())


def parse_meta(text):
    from .download import ensure_table_bytes
    raw = text.encode("utf-8") if isinstance(text, str) else text
    ensure_table_bytes(raw, "metadata")
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def validate_cna_meta(fields, study):
    expected = {
        "cancer_study_identifier": study,
        "genetic_alteration_type": "COPY_NUMBER_ALTERATION",
        "datatype": "DISCRETE",
        "stable_id": "gistic",
        "data_filename": "data_cna.txt",
    }
    errors = [f"{key}={fields.get(key, '')!r}" for key, value in expected.items() if fields.get(key) != value]
    description = fields.get("profile_description", "")
    if "GISTIC" not in description:
        errors.append("profile_description does not identify GISTIC")
    if errors:
        raise CnaFormatError(study + " CNA profile is not the discrete GISTIC table: " + "; ".join(errors))
    return {
        "study": study,
        "datatype": fields["datatype"],
        "stable_id": fields["stable_id"],
        "profile_description": description,
        "genome_build": "not_specified_in_meta_cna",
    }


def validate_study_meta(fields, study):
    if fields.get("cancer_study_identifier") != study:
        raise CnaFormatError(f"{study} meta_study identifier is {fields.get('cancer_study_identifier')}")
    return {
        "study": study,
        "type_of_cancer": fields.get("type_of_cancer", ""),
        "name": fields.get("name", ""),
        "citation": fields.get("citation", ""),
        "pmid": fields.get("pmid", ""),
        "groups": fields.get("groups", ""),
        "description": fields.get("description", ""),
    }


def read_clinical_table(path):
    from .download import ensure_table_bytes
    data = path.read_bytes()
    ensure_table_bytes(data, path.name)
    lines = data.decode("utf-8-sig").splitlines()
    header = None
    start = 0
    for index, line in enumerate(lines):
        if line.startswith("#") or not line.strip():
            continue
        header = line.split("\t")
        start = index + 1
        break
    if not header:
        raise CnaFormatError("Clinical file has no header: " + path.name)
    rows = []
    for line_number, line in enumerate(lines[start:], start + 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < len(header):
            parts.extend([""] * (len(header) - len(parts)))
        row = dict(zip(header, parts))
        row["_line_number"] = line_number
        rows.append(row)
    return header, rows


def parse_case_list(path):
    from .download import ensure_table_bytes
    text = ensure_table_bytes(path.read_bytes(), path.name).decode("utf-8")
    meta = {}
    identifiers = []
    for line in text.splitlines():
        if line.startswith("case_list_ids:"):
            identifiers = [item for item in line.split(":", 1)[1].split() if item]
        elif ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, identifiers


def _signature(tokens):
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def format_rate(numerator, denominator):
    if denominator <= 0:
        raise ValueError("A zero denominator is missing, not a rate")
    text = format(Decimal(numerator) / Decimal(denominator), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def frequency_from_calls(calls):
    """Count AMP/DEL rates. Missing calls stay out of the denominator; 0 stays in."""
    valid = []
    illegal = []
    for call in calls:
        try:
            value = parse_gistic(call)
        except CnaFormatError:
            illegal.append(call)
            continue
        if value is not None:
            valid.append(value)
    denominator = len(valid)
    amp = sum(value == 2 for value in valid)
    deletion = sum(value == -2 for value in valid)
    result = {
        "denominator": denominator,
        "amp_count": amp if denominator else None,
        "del_count": deletion if denominator else None,
        "amp_rate": format_rate(amp, denominator) if denominator else None,
        "del_rate": format_rate(deletion, denominator) if denominator else None,
        "illegal": illegal,
    }
    if amp + deletion > denominator:
        raise CnaFormatError("AMP and DEL counts exceed the denominator")
    return result


def accumulate_cna(path, selected_by_cancer, map_row, n_nodes, cancer_order):
    """Stream a discrete matrix into per-node counts.

    Identical duplicate source rows can be collapsed. Conflicting measurements
    quarantine that node for every cancer in this file. Illegal tokens are not
    entered into the denominator; a continuous-looking matrix is rejected.
    """
    from .download import ensure_table_bytes
    issues = []
    illegal = []
    stats = {
        "source_rows": 0,
        "accepted_rows": 0,
        "rejected_rows": 0,
        "outside_node_set": 0,
        "deduplicated_rows": 0,
        "conflicting_nodes": 0,
        "illegal_cells": 0,
        "parsed_cells": 0,
        "methods": {},
    }
    links = {}
    state = {}
    cancer_pos = {cancer: index for index, cancer in enumerate(cancer_order)}

    def blank_state():
        return {
            "signature": None,
            "rows": 0,
            "lines": [],
            "symbols": [],
            "entrez": [],
            "methods": [],
            "hgnc_id": "",
            "quarantined": False,
            "counts": {cancer: None for cancer in selected_by_cancer},
        }

    with path.open("rb") as handle:
        header_line = handle.readline()
        ensure_table_bytes(header_line, path.name)
        header = header_line.decode("utf-8-sig").rstrip("\r\n").split("\t")
        if len(header) < 3 or header[0] != "Hugo_Symbol" or header[1] != "Entrez_Gene_Id":
            raise CnaFormatError("CNA header is not Hugo_Symbol, Entrez_Gene_Id, samples")
        samples = header[2:]
        if len(samples) != len(set(samples)):
            raise CnaFormatError("CNA header repeats a sample identifier")
        sample_index = {sample: index for index, sample in enumerate(samples)}
        column_groups = {}
        for cancer, identifiers in selected_by_cancer.items():
            missing = [sample for sample in identifiers if sample not in sample_index]
            if missing:
                raise CnaFormatError("Selected sample missing from CNA header: " + ",".join(missing[:5]))
            column_groups[cancer] = [sample_index[sample] for sample in identifiers]
        for line_number, raw_line in enumerate(handle, start=2):
            if raw_line.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise CnaFormatError("CNA body contains an LFS pointer")
            parts = raw_line.decode("utf-8").rstrip("\r\n").split("\t")
            stats["source_rows"] += 1
            if len(parts) != len(header):
                issues.append({
                    "issue_type": "column_count_mismatch", "severity": "error", "study": path.parent.name,
                    "line_number": line_number, "source_symbol": parts[0] if parts else "",
                    "source_entrez": parts[1] if len(parts) > 1 else "", "node_id": "", "hgnc_id": "",
                    "mapping_method": "", "detail": f"columns={len(parts)} expected={len(header)}",
                })
                stats["rejected_rows"] += 1
                continue
            symbol, entrez = parts[0], parts[1]
            mapped = map_row(symbol, entrez)
            method = mapped["method"]
            if mapped["status"] != "accepted":
                stats["rejected_rows"] += 1
                detail = mapped["detail"]
                if "identity_not_assignable" in detail or "symbol_entrez_conflict" in detail or "ambiguous_entrez" in detail:
                    issues.append({
                        "issue_type": "mapping_rejected", "severity": "error", "study": path.parent.name,
                        "line_number": line_number, "source_symbol": symbol, "source_entrez": entrez,
                        "node_id": mapped["node_id"], "hgnc_id": mapped["hgnc_id"],
                        "mapping_method": method, "detail": detail,
                    })
                elif "outside_node_set" in detail:
                    stats["outside_node_set"] += 1
                elif "ambiguous_symbol" in detail:
                    stats["ambiguous_symbol"] = stats.get("ambiguous_symbol", 0) + 1
                else:
                    stats["unmapped_symbol"] = stats.get("unmapped_symbol", 0) + 1
                continue
            node_id = int(mapped["node_id"])
            if node_id < 0 or node_id >= n_nodes:
                raise CnaFormatError("Mapped node id is outside the frozen node list")
            stats["accepted_rows"] += 1
            stats["methods"][method] = stats["methods"].get(method, 0) + 1
            signature = _signature(parts[2:])
            slot = state.get(node_id)
            if slot is None:
                slot = blank_state()
                state[node_id] = slot
            slot["rows"] += 1
            slot["lines"].append(line_number)
            slot["symbols"].append(symbol)
            slot["entrez"].append(entrez)
            slot["methods"].append(method)
            slot["hgnc_id"] = mapped["hgnc_id"]
            if slot["signature"] is None:
                slot["signature"] = signature
                for cancer, columns in column_groups.items():
                    denominator = amp = deletion = 0
                    for column in columns:
                        stats["parsed_cells"] += 1
                        try:
                            value = parse_gistic(parts[2 + column])
                        except CnaFormatError:
                            stats["illegal_cells"] += 1
                            if len(illegal) < 50:
                                illegal.append({
                                    "study": path.parent.name, "line_number": line_number,
                                    "sample_id": samples[column], "source_symbol": symbol,
                                    "token": parts[2 + column],
                                })
                            continue
                        if value is None:
                            continue
                        denominator += 1
                        amp += value == 2
                        deletion += value == -2
                    slot["counts"][cancer] = (denominator, amp, deletion)
            elif slot["signature"] != signature:
                slot["quarantined"] = True
            else:
                stats["deduplicated_rows"] += 1
            if method.startswith("symbol_fallback") or mapped["detail"] in {"source_symbol_not_in_hgnc", "source_symbol_missing"}:
                issues.append({
                    "issue_type": "mapping_note", "severity": "info", "study": path.parent.name,
                    "line_number": line_number, "source_symbol": symbol, "source_entrez": entrez,
                    "node_id": node_id, "hgnc_id": mapped["hgnc_id"], "mapping_method": method,
                    "detail": mapped["detail"],
                })
            if stats["parsed_cells"] >= 1000 and stats["illegal_cells"] / stats["parsed_cells"] > ILLEGAL_LIMIT:
                raise CnaFormatError(
                    "Too many non-discrete GISTIC tokens in " + path.name + "; the continuous matrix was not substituted")
    if stats["parsed_cells"] >= 1000 and stats["illegal_cells"] / stats["parsed_cells"] > ILLEGAL_LIMIT:
        raise CnaFormatError("Too many non-discrete GISTIC tokens in " + path.name)

    for node_id, slot in state.items():
        if slot["quarantined"]:
            stats["conflicting_nodes"] += 1
            issues.append({
                "issue_type": "duplicate_row_conflict", "severity": "error", "study": path.parent.name,
                "line_number": ",".join(str(line) for line in slot["lines"][:20]),
                "source_symbol": "|".join(sorted(set(slot["symbols"]))),
                "source_entrez": "|".join(sorted(set(slot["entrez"]))),
                "node_id": node_id, "hgnc_id": slot["hgnc_id"],
                "mapping_method": "|".join(sorted(set(slot["methods"]))),
                "detail": "source rows disagree; node quarantined for this study",
            })
            continue
        if slot["rows"] > 1:
            issues.append({
                "issue_type": "duplicate_row_identical", "severity": "info", "study": path.parent.name,
                "line_number": ",".join(str(line) for line in slot["lines"][:20]),
                "source_symbol": "|".join(sorted(set(slot["symbols"]))),
                "source_entrez": "|".join(sorted(set(slot["entrez"]))),
                "node_id": node_id, "hgnc_id": slot["hgnc_id"],
                "mapping_method": "|".join(sorted(set(slot["methods"]))),
                "detail": f"collapsed_{slot['rows']}_identical_rows",
            })
        links[node_id] = {
            "node_id": node_id,
            "hgnc_id": slot["hgnc_id"],
            "source_symbol": "|".join(sorted(set(slot["symbols"]))),
            "source_entrez": "|".join(sorted(set(slot["entrez"]))),
            "mapping_method": "|".join(sorted(set(slot["methods"]))),
            "n_source_rows": slot["rows"],
            "study": path.parent.name,
        }
    counts = {}
    for node_id, slot in state.items():
        if slot["quarantined"]:
            counts[node_id] = {cancer: "conflict" for cancer in selected_by_cancer}
        else:
            counts[node_id] = slot["counts"]
    return {
        "samples": samples,
        "counts": counts,
        "links": links,
        "issues": issues,
        "illegal": illegal,
        "stats": stats,
    }
