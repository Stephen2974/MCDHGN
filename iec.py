"""Intrinsic Evidence Calibration (IEC), isolated from E0-E7.

The sealed training files are not modified. ``iec_mode="off"`` uses the
original MCDHGN class. ``iec_mode="on"`` subclasses it, calibrates the
mutation block by protein length, and gates the graph embedding with that
gene's own evidence. Lengths never enter the original 48-d feature matrix.
"""

import contextlib
import csv
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")

import torch
from torch import nn

from model import MCDHGN

# Hashes recorded by results/p0_legacy_D0/E1/seed42/fold0/result.json.
# The IEC launcher refuses to start if these files have changed.
SEALED_CODE_SHA256 = {
    "model.py": "0c667afd9e9ffd00bf2f04290d3b880c35b303cb0ca19e24230097f9be3c18c0",
    "fusion.py": "a8cad7be137a9705052d9c88c3fd269e1058a28825e85fb4a63987ffb06e8ee5",
    "run_ablation.py": "909802329d37513858d03391628e8cb520c9e6d50f3f28147d2fc0c1dccdea0c",
    "ablation_config.py": "6f5e81a470d86048d61a57ab6ada8c8eba11c4e2aa2f9c86136ea72c1b560280",
}

# E1 / seed42 validation genes with label 0 and rank <= 100.
# Fixed from the audit before any IEC training. Not a loss weight or a filter.
PRESPECIFIED_FALSE_POSITIVES = (
    "ATN1", "BLTP1", "COL4A5", "DNAH11", "DPF3", "EXOSC7", "FBP2", "FEM1B",
    "FRAS1", "HELZ", "HR", "LRP5", "MBD5", "MTMR6", "NLGN1", "NUP155",
    "PCDHA9", "PLXNA4", "ROBO4", "SLITRK1", "SVIL", "TAF1A", "WDHD1",
)

LENGTH_FORMULA = "mutation_columns / log(clamp(length, min=2))"
IMPUTED_SOURCE = "imputed_median_of_resolved_lengths"
UNIPROT_SOURCE = "uniprot"


class _LengthToken:
    def __init__(self):
        self._var = __import__("contextvars").ContextVar("iec_protein_lengths", default=None)

    def get(self):
        return self._var.get()

    @contextlib.contextmanager
    def bind(self, lengths):
        token = self._var.set(lengths)
        try:
            yield
        finally:
            self._var.reset(token)


_LENGTHS = _LengthToken()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_sealed_code(root):
    root = Path(root)
    mismatches = []
    for name, expected in SEALED_CODE_SHA256.items():
        actual = sha256_file(root / name)
        if actual != expected:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            "Sealed E1 code changed; IEC will not start: " + ", ".join(mismatches)
        )


def assert_output_isolated(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    allowed = (root / "results" / "iec_v1").resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError("IEC output must stay inside results/iec_v1")
    sealed = (
        root / "results" / "p0_legacy_D0",
        root / "results" / "ablation",
        root / "data",
        root / "data_versions" / "p0_v1",
        root / "Intermediate",
    )
    for path in sealed:
        path = path.resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError("IEC output overlaps a sealed path: " + str(path))
    return output


def e1_architecture(mode, mutation_width=16):
    """Return a new dict. Does not mutate EXPERIMENTS."""
    from ablation_config import EXPERIMENTS

    if mode not in ("on", "off"):
        raise ValueError("iec mode must be 'on' or 'off'")
    architecture = dict(EXPERIMENTS["E1"])
    if mode == "off":
        return architecture
    architecture["iec_mode"] = "on"
    architecture["mutation_width"] = int(mutation_width)
    return architecture


class IntrinsicEvidenceCalibration(nn.Module):
    """Map length-calibrated features to the graph width and gate per gene.

    The gate is a scalar for each gene. It mixes that gene's graph embedding
    with its own evidence and does not mix genes.
    """

    def __init__(self, in_size, graph_dim, mutation_width=16):
        super().__init__()
        if mutation_width <= 0 or mutation_width >= in_size:
            raise ValueError("mutation_width must be a proper prefix of the feature vector")
        self.in_size = int(in_size)
        self.graph_dim = int(graph_dim)
        self.mutation_width = int(mutation_width)
        self.evidence = nn.Linear(self.in_size, self.graph_dim)
        self.gate = nn.Linear(self.graph_dim * 2, 1)
        nn.init.xavier_normal_(self.evidence.weight, gain=1.414)
        nn.init.zeros_(self.evidence.bias)
        nn.init.xavier_normal_(self.gate.weight, gain=1.0)
        nn.init.zeros_(self.gate.bias)
        self.last_gate = None

    def forward(self, graph_h, raw, log_length):
        if graph_h.ndim != 2 or raw.ndim != 2:
            raise ValueError("IEC expects (genes, features)")
        if graph_h.shape[0] != raw.shape[0] or raw.shape[1] != self.in_size:
            raise ValueError("IEC feature width does not match this module")
        if graph_h.shape[1] != self.graph_dim:
            raise ValueError("IEC graph width does not match this module")
        if log_length.shape[0] != graph_h.shape[0]:
            raise ValueError("Protein length rows do not match genes")
        scale = log_length.to(device=raw.device, dtype=raw.dtype).reshape(-1, 1)
        calibrated = torch.cat(
            (raw[:, : self.mutation_width] / scale, raw[:, self.mutation_width :]),
            dim=1,
        )
        own = self.evidence(calibrated)
        gate = torch.sigmoid(self.gate(torch.cat((graph_h, own), dim=1)))
        self.last_gate = gate.detach()
        return gate * graph_h + (1.0 - gate) * own


class CalibratedMCDHGN(MCDHGN):
    """E1 backbone plus an optional IEC block in front of the classifier."""

    def __init__(self, *args, iec_mode="off", mutation_width=16, **kwargs):
        if iec_mode not in ("off", "on"):
            raise ValueError("iec_mode must be 'off' or 'on'")
        super().__init__(*args, **kwargs)
        self.iec_mode = iec_mode
        self.mutation_width = int(mutation_width)
        self.iec = None
        if iec_mode == "off":
            return
        in_size = self.layers[0].gat_layers[0].fc_src.in_features
        graph_dim = self.predict.fc1.in_features
        self.iec = IntrinsicEvidenceCalibration(in_size, graph_dim, self.mutation_width)
        pending = _LENGTHS.get()
        if pending is not None:
            self.bind_lengths(pending)
        if self.primary_device is not None:
            self.iec.to(self.primary_device)

    def bind_lengths(self, lengths):
        if self.iec_mode != "on":
            raise RuntimeError("protein lengths are refused while IEC is off")
        values = torch.as_tensor(lengths, dtype=torch.float32).reshape(-1).detach().clone()
        if values.numel() == 0 or not torch.isfinite(values).all() or bool((values < 1).any()):
            raise ValueError("protein lengths must be finite and at least 1")
        log_length = torch.log(values.clamp(min=2)).reshape(-1, 1)
        if self.primary_device is not None:
            log_length = log_length.to(self.primary_device)
        if "protein_log_length" in self._buffers:
            self.protein_log_length = log_length
        else:
            self.register_buffer("protein_log_length", log_length, persistent=True)

    def forward(self, g, h, return_attention=False):
        if self.iec_mode == "off":
            return super().forward(g, h, return_attention=return_attention)
        if not hasattr(self, "protein_log_length"):
            raise RuntimeError("IEC is on but protein lengths were not bound")
        if self.primary_device is not None and h.device != self.primary_device:
            raw = h.to(self.primary_device)
        else:
            raw = h
        encoded, _, alpha, beta = super().forward(g, raw, return_attention=return_attention)
        calibrated = self.iec(encoded, raw, self.protein_log_length)
        return calibrated, self.predict(calibrated), alpha, beta


@contextlib.contextmanager
def use_training_model(mode, lengths=None):
    """Point the sealed trainer at IEC or back at the original class."""
    import run_ablation as runner
    from model import MCDHGN as original_model

    if not hasattr(runner, "torch"):
        runner.dependencies()
    previous = runner.MCDHGN
    try:
        if mode == "on":
            if lengths is None:
                raise ValueError("IEC on requires protein lengths")
            with _LENGTHS.bind(lengths):
                runner.MCDHGN = CalibratedMCDHGN
                yield
        elif mode == "off":
            runner.MCDHGN = original_model
            yield
        else:
            raise ValueError("iec mode must be 'on' or 'off'")
    finally:
        runner.MCDHGN = previous


def _uniprot_lengths(accessions, pause_seconds=0.2):
    found = {}
    accessions = list(dict.fromkeys(accessions))
    for start in range(0, len(accessions), 100):
        batch = accessions[start : start + 100]
        query = " OR ".join("accession:" + item for item in batch)
        url = (
            "https://rest.uniprot.org/uniprotkb/search?query="
            + urllib.parse.quote(query)
            + "&fields=accession,length&format=tsv&size=500"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "MCDHGN-iec-length-table"})
        with urllib.request.urlopen(request, timeout=90) as response:
            text = response.read().decode("utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines or lines[0].split("\t")[:2] != ["Entry", "Length"]:
            raise RuntimeError("Unexpected UniProt response")
        for line in lines[1:]:
            entry, length = line.split("\t")[:2]
            length = int(length)
            if entry in found and found[entry] != length:
                raise RuntimeError("Conflicting UniProt length for " + entry)
            found[entry] = length
        time.sleep(pause_seconds)
    missing = [item for item in accessions if item not in found]
    return found, missing


def write_length_table(root, destination=None):
    """Query UniProt lengths. Does not edit gene_map or biological features."""
    root = Path(root)
    destination = Path(destination or root / "data_versions/iec_v1/protein_length.tsv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    gene_map = root / "data_versions/p0_v1/gene_map.tsv"
    rows = []
    with gene_map.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            accession = (row["uniprot_ids"] or "").split("|")[0].strip()
            rows.append(
                {
                    "node_id": int(row["node_id"]),
                    "source_id": row["source_id"],
                    "symbol": row["symbol"],
                    "uniprot": accession,
                }
            )
    if [row["node_id"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("gene_map node ids are not contiguous")
    found, missing_remote = _uniprot_lengths([row["uniprot"] for row in rows if row["uniprot"]])
    resolved = []
    unresolved = []
    for row in rows:
        if row["uniprot"] and row["uniprot"] in found:
            row["length"] = found[row["uniprot"]]
            row["length_source"] = UNIPROT_SOURCE
            resolved.append(row["length"])
        else:
            unresolved.append(row)
    if not resolved:
        raise RuntimeError("UniProt returned no lengths")
    median = int(round(float(torch.tensor(resolved, dtype=torch.float64).median())))
    for row in unresolved:
        row["length"] = median
        row["length_source"] = IMPUTED_SOURCE
    fieldnames = ["node_id", "source_id", "symbol", "uniprot", "length", "length_source"]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "formula": LENGTH_FORMULA,
        "log_base": "e",
        "clamp_min": 2,
        "gene_map_sha256": sha256_file(gene_map),
        "length_table_sha256": sha256_file(destination),
        "n_nodes": len(rows),
        "n_uniprot": len(rows) - len(unresolved),
        "n_imputed": len(unresolved),
        "median_length": median,
        "imputed_node_ids": [row["node_id"] for row in unresolved],
        "imputed_symbols": [row["symbol"] or row["source_id"] for row in unresolved],
        "uniprot_accessions_missing_remotely": missing_remote,
        "uniprot_query": "https://rest.uniprot.org/uniprotkb/search fields=accession,length",
        "uniprot_release": "unpinned live query",
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination, manifest


def load_length_table(path, num_nodes, gene_map_path):
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("formula") != LENGTH_FORMULA or manifest.get("log_base") != "e":
        raise ValueError("Length table formula does not match IEC")
    if sha256_file(path) != manifest["length_table_sha256"]:
        raise ValueError("Length table does not match its manifest")
    if sha256_file(gene_map_path) != manifest["gene_map_sha256"]:
        raise ValueError("gene_map changed after the IEC length table was written")
    if int(manifest["n_nodes"]) != num_nodes:
        raise ValueError("Length table node count differs from the graph")
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != num_nodes:
        raise ValueError("Length table row count differs from the graph")
    lengths = torch.empty(num_nodes, dtype=torch.float32)
    seen = set()
    imputed = []
    for row in rows:
        node_id = int(row["node_id"])
        if node_id in seen or not 0 <= node_id < num_nodes:
            raise ValueError("Length table node ids are not a full gene index")
        seen.add(node_id)
        length = float(row["length"])
        if length < 1 or length != int(length):
            raise ValueError("Protein length must be a positive integer")
        source = row["length_source"]
        if source == IMPUTED_SOURCE:
            imputed.append(node_id)
        elif source != UNIPROT_SOURCE:
            raise ValueError("Unknown length source for node " + str(node_id))
        lengths[node_id] = length
    if seen != set(range(num_nodes)):
        raise ValueError("Length table is missing node ids")
    if imputed != list(manifest["imputed_node_ids"]):
        raise ValueError("Imputed length rows differ from the manifest")
    return lengths, manifest
