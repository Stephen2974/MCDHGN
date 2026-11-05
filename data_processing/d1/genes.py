"""Map CNA gene rows onto the frozen P0 node identities."""
from .select import clean_token, is_missing_token


def nodes_by_hgnc(gene_rows):
    grouped = {}
    for row in gene_rows:
        hid = row.get("hgnc_id", "")
        if not hid:
            continue
        grouped.setdefault(hid, []).append(row)
    return grouped


def _result(status, method="", node_id="", hgnc_id="", detail=""):
    return {
        "status": status,
        "method": method,
        "node_id": "" if node_id == "" else str(node_id),
        "hgnc_id": hgnc_id,
        "detail": detail,
    }


def _assignable(hgnc_id, grouped):
    nodes = grouped.get(hgnc_id, [])
    resolved = [row for row in nodes if row.get("mapping_status") == "resolved"]
    if len(nodes) == 1 and len(resolved) == 1:
        return resolved[0], ""
    if not nodes:
        return None, "outside_node_set"
    statuses = "|".join(sorted({row.get("mapping_status", "") for row in nodes}))
    return None, "identity_not_assignable:" + statuses


def _symbol_fallback(symbol, index, grouped, detail):
    if not symbol:
        return _result("rejected", detail=detail or "no_identifier")
    found, method = index.resolve(symbol, "symbol")
    note = detail
    if len(found) != 1:
        reason = "ambiguous_symbol" if found else "unmapped_symbol"
        extra = "|".join(sorted(found))
        return _result("rejected", detail=";".join(x for x in (note, reason, extra) if x))
    hgnc_id = next(iter(found))
    row, why = _assignable(hgnc_id, grouped)
    if row is None:
        return _result("rejected", hgnc_id=hgnc_id, detail=";".join(x for x in (note, why) if x))
    return _result("accepted", "symbol_fallback_" + method, row["node_id"], hgnc_id, note)


def map_source_row(symbol, entrez, index, grouped):
    """Map one CNA row. Symbol fallback is used only when Entrez is absent or unknown.

    A unique Entrez identity that points at an unresolved or duplicated node is
    not replaced by a symbol guess. Conflicting symbol and Entrez identities are
    rejected instead of picking the first candidate.
    """
    symbol_c = "" if is_missing_token(symbol) else clean_token(symbol)
    entrez_c = "" if is_missing_token(entrez) else clean_token(entrez)
    if entrez_c and not entrez_c.isdigit():
        mapped = _symbol_fallback(symbol_c, index, grouped, "invalid_entrez")
        if mapped["status"] != "accepted":
            mapped["detail"] = ";".join(x for x in (mapped["detail"], "invalid_entrez") if x)
        return mapped
    if entrez_c:
        found, _namespace = index.resolve(entrez_c, "entrez_id")
        if len(found) > 1:
            return _result("rejected", detail="ambiguous_entrez:" + "|".join(sorted(found)))
        if len(found) == 1:
            hgnc_id = next(iter(found))
            note = ""
            if symbol_c:
                symbol_found, symbol_method = index.resolve(symbol_c, "symbol")
                if symbol_found and hgnc_id not in symbol_found:
                    return _result(
                        "rejected", hgnc_id=hgnc_id,
                        detail="symbol_entrez_conflict:entrez=" + hgnc_id + ";symbol=" + "|".join(sorted(symbol_found)))
                if not symbol_found:
                    note = "source_symbol_not_in_hgnc"
                elif len(symbol_found) > 1:
                    note = "source_symbol_ambiguous_consistent_with_entrez"
                elif symbol_method != "approved_symbol":
                    note = "source_symbol_" + symbol_method
            else:
                note = "source_symbol_missing"
            row, why = _assignable(hgnc_id, grouped)
            if row is None:
                return _result("rejected", hgnc_id=hgnc_id, detail=why)
            return _result("accepted", "entrez_hgnc", row["node_id"], hgnc_id, note)
        return _symbol_fallback(symbol_c, index, grouped, "entrez_not_in_hgnc_snapshot")
    return _symbol_fallback(symbol_c, index, grouped, "")
