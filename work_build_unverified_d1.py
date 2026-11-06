import csv, json
import os, subprocess
from pathlib import Path
import torch, dgl

root = Path(r"G:\LLLM\MCDHGN-main")
src = root / "data_versions/d1_cnv_v1/cnv_features_finite_median.tsv"
out = root / "data_variants/d1_cnv_unverified_v1"
if out.exists():
    raise SystemExit("variant exists")
rows = list(csv.DictReader(src.open(encoding="utf-8-sig"), delimiter="\t"))
rows.sort(key=lambda r: int(r["node_id"]))
cols = [k for k in rows[0] if k.startswith("CNV_")]
graphs, labels = dgl.load_graphs(str(root / "data/network/hetero/new_9nodes_graph.bin"))
g = graphs[0]
old = g.nodes["Gene"].data["feature"].float().cpu()
if tuple(old.shape) != (len(rows), 48):
    raise RuntimeError(f"old shape {tuple(old.shape)}")
extra = torch.tensor([[float(r[c]) for c in cols] for r in rows], dtype=torch.float32)
if tuple(extra.shape) != (len(rows), 32) or not torch.isfinite(extra).all():
    raise RuntimeError("bad CNV")
g.nodes["Gene"].data["feature"] = torch.cat([old, extra], 1)
(out / "data/network/hetero").mkdir(parents=True)
dgl.save_graphs(str(out / "data/network/hetero/new_9nodes_graph.bin"), [g], labels)
(out / "data").mkdir(exist_ok=True)
for name in ["step1"]:
    subprocess.check_call(["cmd", "/c", "mklink", "/J", str(out / "data" / name), str(root / "data" / name)])
subprocess.check_call(["cmd", "/c", "mklink", "/J", str(out / "Intermediate"), str(root / "Intermediate")])
(out / "data_versions").mkdir(exist_ok=True)
(out / "data_versions/missing_policy.json").write_text(
    (root / "data_versions/d1_cnv_v1/missing_policy_median_v1.txt").read_text(encoding="utf-8"),
    encoding="utf-8",
)
m = {"status":"unverified_temporary","feature_shape":[len(rows),80],
     "source_verification":"pending raw cBioPortal files",
     "warning":"formal D1 pending verify and graph build"}
(out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
print(m)
