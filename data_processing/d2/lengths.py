"""Audit-only exonic gene lengths from local GENCODE v28 (no frequency normalization)."""
import gzip
from collections import defaultdict
from pathlib import Path


def exon_lengths_from_gff3_gz(path):
    """Return {gene_name: exonic_bp_sum} from GENCODE basic GFF3.gz."""
    lengths = defaultdict(int)
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            start, end = int(parts[3]), int(parts[4])
            length = abs(end - start)
            gene_name = None
            for elem in parts[8].split(";"):
                elem = elem.strip()
                if elem.startswith("gene_name="):
                    gene_name = elem.split("=", 1)[1].strip()
                    break
            if gene_name:
                lengths[gene_name] += length
    return dict(lengths)
