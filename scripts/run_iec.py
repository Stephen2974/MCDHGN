"""Train IEC against the sealed E1 protocol, in a separate output tree.

``--iec off`` runs the original MCDHGN class with the E1 architecture dict.
``--iec on`` adds Intrinsic Evidence Calibration and a protein-length table.
Neither choice writes into results/p0_legacy_D0, results/ablation, or the
P0 data directories. model.py, fusion.py, run_ablation.py and
ablation_config.py are hashed and must still match the sealed E1 run.

Examples
--------
python scripts/run_iec.py --write-lengths
python scripts/run_iec.py --check-files-only
python scripts/run_iec.py --iec on --smoke --cpu
python scripts/run_iec.py --iec on --devices 0
python scripts/run_iec.py --iec on --devices 0 1 2
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iec import (  # noqa: E402
    PRESPECIFIED_FALSE_POSITIVES,
    assert_output_isolated,
    assert_sealed_code,
    e1_architecture,
    load_length_table,
    sha256_file,
    use_training_model,
    write_length_table,
)
import run_ablation as runner  # noqa: E402


E1_HYPERPARAMETERS = {
    "epochs": 1000,
    "eval_every": 10,
    "patience": 20,
    "lr": 0.01,
    "weight_decay": 0.0001,
    "positive_weight": 1.904,
    "dropout": 0.4,
    "hidden_size": 256,
    "heads": 4,
    "fusion_dim": 64,
    "fusion_chunk_size": 1024,
}


def fold_assignments(devices, folds):
    folds = list(folds)
    buckets = {device: [] for device in devices}
    for index, fold in enumerate(folds):
        buckets[devices[index % len(devices)]].append(fold)
    return [(device, buckets[device]) for device in devices if buckets[device]]


def code_hashes():
    names = (
        "model.py",
        "fusion.py",
        "run_ablation.py",
        "ablation_config.py",
        "iec.py",
        "scripts/run_iec.py",
    )
    return {name: sha256_file(ROOT / name) for name in names}


def build_config(mode, seed, fold, smoke, length_sha=None):
    hyper = dict(E1_HYPERPARAMETERS)
    if smoke:
        hyper.update(hidden_size=8, heads=2, fusion_dim=4, epochs=2, eval_every=1, patience=2)
    width = 2 if smoke else 16
    return dict(
        hyper,
        smoke=bool(smoke),
        code_sha256=code_hashes(),
        protocol="iec-v1-softmax-ap-checkpoint",
        experiment="IEC" if mode == "on" else "IEC-off",
        architecture=e1_architecture(mode, mutation_width=width),
        seed=int(seed),
        fold=int(fold),
        length_sha256=length_sha,
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_gates(mode, lengths, config, features, graphs, primary, ids, checkpoint, destination):
    if mode != "on":
        return
    runner.dependencies()
    with use_training_model(mode, lengths):
        model, placed = runner.make_model(config, features, graphs, primary, ids)
    state = runner.load_torch(checkpoint)["state_dict"]
    model.load_state_dict(state)
    model.eval()
    with runner.torch.no_grad():
        model(placed, features.to(primary))
    gate = model.iec.last_gate.detach().cpu().reshape(-1)
    runner.np.savez_compressed(
        destination,
        node_ids=runner.np.arange(gate.numel()),
        gate=gate.numpy(),
    )
    del model, placed


def prespecified_report(output, gene_map):
    symbols = {}
    with gene_map.open(encoding="utf-8") as stream:
        for row in __import__("csv").DictReader(stream, delimiter="\t"):
            symbols[row["symbol"]] = int(row["node_id"])
    missing = [name for name in PRESPECIFIED_FALSE_POSITIVES if name not in symbols]
    if missing:
        raise RuntimeError("Prespecified genes missing from gene_map: " + ", ".join(missing))
    rows = []
    for prediction in sorted(output.rglob("validation_predictions.npz")):
        result_path = prediction.with_name("result.json")
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        store = runner.np.load(prediction)
        ids = store["node_ids"].astype(int)
        labels = store["labels"].astype(int)
        probabilities = store["probabilities"].astype(float)
        order = runner.np.argsort(-probabilities, kind="stable")
        rank = runner.np.empty(len(ids), dtype=int)
        rank[order] = runner.np.arange(1, len(ids) + 1)
        position = {int(node): index for index, node in enumerate(ids)}
        gate_path = prediction.with_name("iec_gate.npz")
        gates = None
        if gate_path.is_file():
            gate_store = runner.np.load(gate_path)
            gates = gate_store["gate"].astype(float)
        for name in PRESPECIFIED_FALSE_POSITIVES:
            node = symbols[name]
            if node not in position:
                continue
            index = position[node]
            rows.append(
                {
                    "symbol": name,
                    "node_id": node,
                    "seed": result["config"]["seed"],
                    "fold": result["config"]["fold"],
                    "label": int(labels[index]),
                    "rank": int(rank[index]),
                    "probability": float(probabilities[index]),
                    "gate": None if gates is None else float(gates[node]),
                    "in_top_100": bool(rank[index] <= 100),
                }
            )
    destination = output / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "prespecified_genes.json", {"genes": rows})
    return rows


def train_folds(args, output, folds):
    runner.dependencies()
    if args.smoke:
        runner.torch.set_num_threads(1)
    primary, ids = runner.choose_devices(args)
    features, graphs, splits, fingerprint = runner.load_data(args.data_root, args.smoke)
    lengths = None
    length_sha = None
    if args.iec == "on" and not args.smoke:
        lengths, _manifest = load_length_table(
            args.length_table, features.shape[0], ROOT / "data_versions/p0_v1/gene_map.tsv"
        )
        length_sha = sha256_file(args.length_table)
        fingerprint = dict(fingerprint)
        fingerprint[args.length_table.resolve().relative_to(ROOT).as_posix()] = length_sha
    elif args.iec == "on":
        generator = runner.torch.Generator().manual_seed(71)
        lengths = runner.torch.randint(50, 2000, (features.shape[0],), generator=generator)
    for seed in dict.fromkeys(args.seeds):
        for fold in folds:
            config = build_config(args.iec, seed, fold, args.smoke, length_sha)
            run_dir = output / config["experiment"] / f"seed{seed}" / f"fold{fold}"
            with use_training_model(args.iec, lengths):
                runner.train_one(config, (features, graphs, splits, fingerprint), run_dir, primary, ids)
            if args.iec == "on" and (run_dir / "best.pt").is_file():
                save_gates(
                    args.iec, lengths, config, features, graphs, primary, ids,
                    run_dir / "best.pt", run_dir / "iec_gate.npz",
                )
    if not args.smoke:
        prespecified_report(output, ROOT / "data_versions/p0_v1/gene_map.tsv")


def launch_workers(args, output, assignments):
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for device, folds in assignments:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--iec",
            args.iec,
            "--worker",
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--folds",
            *[str(fold) for fold in folds],
            "--devices",
            str(device),
            "--output",
            str(output),
            "--length-table",
            str(args.length_table),
            "--data-root",
            str(args.data_root),
        ]
        log = logs / f"gpu{device}_folds{'_'.join(str(fold) for fold in folds)}.log"
        print("Running:", subprocess.list2cmdline(command), flush=True)
        stream = log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=dict(os.environ, DGLBACKEND="pytorch", PYTHONUNBUFFERED="1"),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, stream, log))
    failed = []
    for process, stream, log in processes:
        code = process.wait()
        stream.close()
        if code != 0:
            failed.append(f"{log} exited {code}")
    if failed:
        raise RuntimeError("IEC worker failed: " + "; ".join(failed))
    prespecified_report(output, ROOT / "data_versions/p0_v1/gene_map.tsv")


def default_output(mode):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "iec_v1" / f"{'IEC' if mode == 'on' else 'IEC-off'}_{stamp}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iec", choices=("on", "off"))
    parser.add_argument("--write-lengths", action="store_true")
    parser.add_argument("--check-files-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--length-table", type=Path, default=ROOT / "data_versions/iec_v1/protein_length.tsv")
    args = parser.parse_args()
    if args.write_lengths:
        path, manifest = write_length_table(ROOT, args.length_table)
        print(json.dumps({"length_table": str(path), "n_imputed": manifest["n_imputed"]}, ensure_ascii=False))
        return
    assert_sealed_code(ROOT)
    if args.check_files_only:
        if args.iec is None:
            args.iec = "on"
        architecture = e1_architecture(args.iec)
        from ablation_config import EXPERIMENTS

        if EXPERIMENTS["E1"] != {key: EXPERIMENTS["E1"][key] for key in ("use_gatv2", "semantic_mode", "fusion_mode", "fusion_alpha")}:
            raise RuntimeError("E1 architecture dict changed")
        if "iec_mode" in EXPERIMENTS["E1"]:
            raise RuntimeError("E1 architecture contains IEC")
        print(json.dumps({
            "iec": args.iec,
            "architecture": architecture,
            "hyperparameters": E1_HYPERPARAMETERS,
            "seeds": args.seeds,
            "length_table": str(args.length_table),
            "sealed_code": "unchanged",
        }, indent=2, ensure_ascii=False))
        return
    if args.iec is None:
        parser.error("--iec on or --iec off is required")
    output = assert_output_isolated(ROOT, args.output or default_output(args.iec))
    if not args.worker:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", {
            "iec": args.iec,
            "seeds": args.seeds,
            "hyperparameters": E1_HYPERPARAMETERS,
            "architecture": e1_architecture(args.iec, 2 if args.smoke else 16),
            "sealed_code_sha256": code_hashes(),
        })
    args.data_root = args.data_root.resolve()
    args.length_table = args.length_table.resolve()
    folds = args.folds
    if folds is None:
        if args.smoke:
            folds = [0, 1]
        else:
            folds = [0, 1, 2, 3, 4]
    devices = args.devices
    if not args.cpu and devices and len(devices) > 1 and not args.worker and not args.smoke:
        launch_workers(args, output, fold_assignments(devices, folds))
        return
    train_folds(args, output, folds)


if __name__ == "__main__":
    main()
