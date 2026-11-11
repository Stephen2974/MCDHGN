"""Run controlled MCDHGN ablations on the training server.

No test labels are read during selection. --test-run evaluates a previously
selected checkpoint on the held-out test set, without choosing an epoch.
Use --smoke for a tiny synthetic CPU/GPU end-to-end check before real runs.
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

from ablation_config import EXPERIMENTS, METAPATHS

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependencies():
    # Keep --list usable even before the server environment is activated.
    os.environ.setdefault("DGLBACKEND", "pytorch")
    global torch, np, dgl, F, metrics, MCDHGN, place_metapath_graphs
    import torch
    import numpy as np
    import dgl
    import torch.nn.functional as F
    from sklearn import metrics
    from model import MCDHGN, place_metapath_graphs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # DGL/CUDA scatter operations can still be nondeterministic.


def as_ids(value):
    return torch.as_tensor(value, dtype=torch.long, device="cpu").reshape(-1)


def load_torch(path):
    # Support both the original torch 1.12 environment and newer releases.
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False  # repository/generated trusted artifacts
    return torch.load(path, **options)


def global_graph(block, num_nodes):
    """Map cached block-local IDs back to gene IDs; preserve every edge."""
    block = block.to("cpu")
    src, dst = block.edges(order="eid")
    if block.is_block:
        if dgl.NID not in block.srcdata or dgl.NID not in block.dstdata:
            raise ValueError("Cached block lacks original node IDs; cannot align features safely")
        src = block.srcdata[dgl.NID][src]
        dst = block.dstdata[dgl.NID][dst]
    elif block.num_nodes() != num_nodes:
        raise ValueError("Expected a full gene graph or a DGL block with original node IDs")
    g = dgl.graph((src.long(), dst.long()), num_nodes=num_nodes)
    if (g.in_degrees() == 0).any():
        raise ValueError("A cached metapath has zero-in-degree genes; inspect sampling/self loops")
    return g


def validate_splits(splits, num_nodes):
    universe, validation_ids, known_labels = None, set(), {}
    for train_ids, val_ids, train_y, val_y in splits:
        train_set, val_set = set(train_ids.tolist()), set(val_ids.tolist())
        if len(train_set) != len(train_ids) or len(val_set) != len(val_ids):
            raise ValueError("Duplicate gene IDs in a split")
        if train_set & val_set or validation_ids & val_set:
            raise ValueError("Training/validation leakage or overlapping validation folds")
        if universe is None:
            universe = train_set | val_set
        if train_set | val_set != universe:
            raise ValueError("Folds do not share the same training+validation universe")
        validation_ids |= val_set
        for ids, labels in ((train_ids, train_y), (val_ids, val_y)):
            if len(ids) != len(labels) or set(labels.tolist()) != {0, 1}:
                raise ValueError("Every split must contain aligned labels of both classes")
            if ids.min() < 0 or ids.max() >= num_nodes:
                raise ValueError("Gene IDs outside feature matrix")
            for node, label in zip(ids.tolist(), labels.tolist()):
                if node in known_labels and known_labels[node] != label:
                    raise ValueError("Inconsistent label for a gene across folds")
                known_labels[node] = label
    if validation_ids != universe:
        raise ValueError("Validation folds must partition the development set")


def validate_node_mapping(mapping, num_nodes):
    if not isinstance(mapping, dict) or len(mapping) != num_nodes:
        raise ValueError("Gene node mapping and graph feature rows differ")
    if any(not isinstance(gene, str) or not gene for gene in mapping):
        raise ValueError("Gene node mapping has invalid gene identifiers")
    if any(type(node) is not int for node in mapping.values()):
        raise ValueError("Gene node IDs must be integers")
    if sorted(mapping.values()) != list(range(num_nodes)):
        raise ValueError("Gene node IDs must be unique and contiguous from zero")


def load_data(data_root, smoke=False):
    if smoke:
        gen = torch.Generator().manual_seed(71)
        x = torch.randn(24, 8, generator=gen)
        nodes = torch.arange(24)
        graphs = [dgl.graph((torch.cat([nodes, nodes]), torch.cat([nodes, (nodes + k) % 24])),
                            num_nodes=24) for k in (1, 3, 5)]
        splits = []
        for val in (torch.arange(0, 10), torch.arange(10, 20)):
            train = torch.tensor([i for i in range(20) if i not in set(val.tolist())])
            splits.append((train, val, train % 2, val % 2))
        return x, graphs, splits, {"synthetic": "smoke-v1-seed71"}
    graph_file = data_root / "data/network/hetero/new_9nodes_graph.bin"
    mapping_file = data_root / "data/step1/gene_toids.pkl"
    split_file = data_root / "Intermediate/label_set/train_batch.pkl"
    block_files = [data_root / f"Intermediate/blocks/{i}gs.bin" for i in range(9)]
    files = [graph_file, mapping_file, split_file] + block_files
    # Future data variants can carry these explicit contracts at their root.
    files += [data_root / name for name in ("gene_map.tsv", "feature_schema.tsv", "metapaths.json", "split_manifest.json")
              if (data_root / name).is_file()]
    fingerprint = {p.relative_to(data_root).as_posix(): sha256(p) for p in files}
    source_graphs, _ = dgl.load_graphs(str(graph_file))
    if len(source_graphs) != 1:
        raise ValueError("Expected exactly one training heterograph")
    x = source_graphs[0].nodes["Gene"].data["feature"].float().cpu()
    with mapping_file.open("rb") as stream:
        mapping = pickle.load(stream)
    if x.ndim != 2 or source_graphs[0].num_nodes("Gene") != len(x):
        raise ValueError("Invalid Gene feature shape")
    validate_node_mapping(mapping, len(x))
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite input features")
    graphs = []
    for path in block_files:
        # These are the repository's trusted, precomputed sampling artifacts.
        with path.open("rb") as stream:
            graphs.append(global_graph(pickle.load(stream), len(x)))
    with split_file.open("rb") as stream:
        splits = [tuple(as_ids(v) for v in row) for row in pickle.load(stream)]
    if len(splits) != 5:
        raise ValueError(f"Expected 5 stored folds; found {len(splits)}")
    validate_splits(splits, len(x))
    return x, graphs, splits, fingerprint


def choose_devices(args):
    if args.cpu or not torch.cuda.is_available():
        if args.devices:
            raise ValueError("--devices requested but CPU mode/CUDA unavailable")
        return torch.device("cpu"), None
    ids = args.devices if args.devices is not None else list(range(torch.cuda.device_count()))
    if len(set(ids)) != len(ids) or any(i < 0 or i >= torch.cuda.device_count() for i in ids):
        raise ValueError("Invalid or duplicate GPU IDs")
    return torch.device(f"cuda:{ids[0]}"), ids


def sync(ids):
    for device in ids or []:
        torch.cuda.synchronize(device)


def make_model(config, features, graphs, primary, ids):
    parallel_ids = ids if ids and len(ids) > 1 else None
    model = MCDHGN(
        num_meta_paths=len(graphs), in_size=features.shape[1],
        hidden_size=config["hidden_size"], out_size=2, num_heads=[config["heads"]],
        dropout=config["dropout"], share_weights=False,
        metapath_device_ids=parallel_ids,
        primary_device_id=primary.index or 0,
        edge_counts=[g.num_edges() for g in graphs], parallel_metapaths=False,
        fusion_dim=config["fusion_dim"], fusion_chunk_size=config["fusion_chunk_size"],
        **config["architecture"],
    )
    if parallel_ids is None:
        model.to(primary)
    placed = place_metapath_graphs(graphs, model.metapath_devices, primary)
    return model, placed


def classification_metrics(logits, labels):
    scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()
    predicted = logits.argmax(1).detach().cpu().numpy()
    result = {
        "ap": float(metrics.average_precision_score(truth, scores)),
        "auc": float(metrics.roc_auc_score(truth, scores)),
        "accuracy": float(metrics.accuracy_score(truth, predicted)),
        "macro_f1": float(metrics.f1_score(truth, predicted, average="macro", zero_division=0)),
        "loss": float(F.cross_entropy(logits, labels).detach().cpu()),
    }
    for k in (50, 100):
        count = min(k, len(truth))
        top = np.argsort(-scores, kind="stable")[:count]
        result[f"precision_at_{k}"] = float(truth[top].mean())
        result[f"recall_at_{k}"] = float(truth[top].sum() / truth.sum())
        result[f"effective_k_{k}"] = count
    return result, scores


def export_predictions(path, ids, labels, logits):
    result, scores = classification_metrics(logits, labels)
    np.savez_compressed(path, node_ids=ids.cpu().numpy(), labels=labels.cpu().numpy(),
                        probabilities=scores, logits=logits.detach().cpu().numpy())
    return result


def export_weights(path, beta, num_nodes, num_paths):
    weights = beta.detach().cpu()
    if weights.ndim == 2:
        weights = weights.unsqueeze(0).expand(num_nodes, -1, -1)
    weights = weights.squeeze(-1).numpy()
    np.savez_compressed(path, node_ids=np.arange(num_nodes), beta=weights,
                        metapaths=np.array(METAPATHS[:num_paths]))
    return {"mean_by_path": weights.mean(0).tolist(),
            "std_by_path": weights.std(0).tolist(),
            "mean_entropy": float(-(weights * np.log(np.maximum(weights, 1e-12))).sum(1).mean())}


def train_one(config, data, run_dir, primary, ids):
    x, graphs, splits, fingerprint = data
    result_path = run_dir / "result.json"
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if old["config"] != config or old["data_sha256"] != fingerprint:
            raise ValueError(f"Existing run differs: {run_dir}; use a new --output")
        print(f"SKIP complete {run_dir}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {"config": config, "data_sha256": fingerprint})
    set_seed(config["seed"] + 1000 * config["fold"])
    model, placed = make_model(config, x, graphs, primary, ids)
    x = x.to(primary)
    train_ids, val_ids, train_y, val_y = [v.to(primary) for v in splits[config["fold"]]]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    class_weight = torch.tensor([1.0, config["positive_weight"]], device=primary)
    best_ap, best_epoch, stale, last_epoch = -1.0, 0, 0, 0
    for device in ids or []:
        torch.cuda.reset_peak_memory_stats(device)
    sync(ids)
    started = time.perf_counter()
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, config["epochs"] + 1):
            last_epoch = epoch
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, logits, _, _ = model(placed, x)
            loss = F.cross_entropy(logits[train_ids], train_y, weight=class_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss = float(loss.detach().cpu())
            del logits, loss
            if epoch % config["eval_every"] != 0 and epoch != config["epochs"]:
                continue
            model.eval()
            with torch.no_grad():
                _, logits, _, _ = model(placed, x)
                val_metrics, _ = classification_metrics(logits[val_ids], val_y)
            row = dict(epoch=epoch, train_loss=train_loss, **val_metrics)
            history.write(json.dumps(row, allow_nan=False) + "\n")
            history.flush()
            print(f"{config['experiment']} seed={config['seed']} fold={config['fold']} "
                  f"epoch={epoch} AP={val_metrics['ap']:.6f} AUC={val_metrics['auc']:.6f}", flush=True)
            if val_metrics["ap"] > best_ap:
                best_ap, best_epoch, stale = val_metrics["ap"], epoch, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"state_dict": state, "config": config, "data_sha256": fingerprint,
                            "best_epoch": epoch}, run_dir / "best.pt")
            else:
                stale += 1
            del logits
            if config["patience"] and stale >= config["patience"]:
                break
    sync(ids)
    elapsed = time.perf_counter() - started
    checkpoint = load_torch(run_dir / "best.pt")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        _, logits, _, beta = model(placed, x)
        best_metrics = export_predictions(run_dir / "validation_predictions.npz", val_ids, val_y, logits[val_ids])
        weight_stats = export_weights(run_dir / "gene_metapath_weights.npz", beta, len(x), len(graphs))
    result = {
        "config": config, "data_sha256": fingerprint, "best_epoch": best_epoch,
        "epochs_trained": last_epoch, "validation": best_metrics, "weight_diagnostics": weight_stats,
        "train_seconds": elapsed, "parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": {str(i): torch.cuda.max_memory_allocated(i) for i in ids or []},
        "environment": {"python": sys.version, "torch": torch.__version__, "dgl": dgl.__version__,
                        "cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in ids or []]},
    }
    write_json(result_path, result)
    print(f"DONE {run_dir} | best epoch={best_epoch} AP={best_metrics['ap']:.6f}", flush=True)
    del model, optimizer, placed, x, checkpoint, logits, beta
    gc.collect()
    for device in ids or []:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def test_checkpoint(args, primary, ids):
    run_dir = args.test_run.resolve()
    checkpoint = load_torch(run_dir / "best.pt")
    config = checkpoint["config"]
    current_hashes = {name: sha256(ROOT / name) for name in config["code_sha256"]}
    if current_hashes != config["code_sha256"]:
        raise ValueError("Code changed since training; evaluate using the recorded code version")
    if config["smoke"]:
        raise ValueError("--test-run is reserved for real, prespecified final comparisons")
    data = load_data(args.data_root)
    x, graphs, splits, fingerprint = data
    if fingerprint != checkpoint["data_sha256"]:
        raise ValueError("Training data changed since checkpoint was created")
    test_mask_file = args.data_root / "Intermediate/label_set/mydatatest_mask.pt"
    test_y_file = args.data_root / "Intermediate/label_set/mydatatest_label.pt"
    mask = as_ids(load_torch(test_mask_file))
    labels = as_ids(load_torch(test_y_file))
    development_ids = set(splits[0][0].tolist()) | set(splits[0][1].tolist())
    if development_ids & set(mask.tolist()):
        raise ValueError("Held-out test set overlaps training/validation")
    if len(mask) != len(labels) or len(set(mask.tolist())) != len(mask) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Invalid test IDs/labels")
    if mask.min() < 0 or mask.max() >= len(x):
        raise ValueError("Invalid test gene ID")
    model, placed = make_model(config, x, graphs, primary, ids)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        _, logits, _, _ = model(placed, x.to(primary))
        result = export_predictions(run_dir / "test_predictions.npz", mask, labels.to(primary), logits[mask.to(primary)])
    write_json(run_dir / "test_result.json", {"test": result, "best_epoch": checkpoint["best_epoch"],
               "checkpoint_sha256": sha256(run_dir / "best.pt"),
               "test_data_sha256": {str(p.name): sha256(p) for p in (test_mask_file, test_y_file)}})
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=["E1", "E2", "E3", "E4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/ablation")
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20, help="validation checks without AP improvement; 0 disables")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--positive-weight", type=float, default=1.904)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--fusion-dim", type=int, default=64)
    parser.add_argument("--fusion-chunk-size", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true", help="synthetic data; force small dimensions and 2 epochs")
    parser.add_argument("--test-run", type=Path, help="evaluate a selected run's best.pt on held-out test labels")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(EXPERIMENTS, indent=2))
        return
    if args.smoke:
        args.hidden_size, args.heads, args.fusion_dim = 8, 2, 4
        args.epochs, args.eval_every, args.patience = 2, 1, 2
    if min(args.epochs, args.eval_every, args.hidden_size, args.heads, args.fusion_dim, args.fusion_chunk_size) <= 0:
        parser.error("epochs/dimensions/intervals must be positive")
    if args.patience < 0 or args.lr <= 0 or args.weight_decay < 0 or args.positive_weight <= 0 or not 0 <= args.dropout < 1:
        parser.error("invalid optimization settings")
    dependencies()
    if args.smoke:
        torch.set_num_threads(1)
    primary, ids = choose_devices(args)
    args.data_root = args.data_root.resolve()
    if args.test_run:
        test_checkpoint(args, primary, ids)
        return
    print("Loading fixed graphs/splits and hashing inputs...", flush=True)
    data = load_data(args.data_root, args.smoke)
    selected_folds = args.folds if args.folds is not None else list(range(len(data[2])))
    if not selected_folds or any(f < 0 or f >= len(data[2]) for f in selected_folds):
        parser.error("fold index outside stored splits")
    code_hashes = {name: sha256(ROOT / name) for name in ("model.py", "fusion.py", "run_ablation.py", "ablation_config.py")}
    common = {key: getattr(args, key) for key in ("epochs", "eval_every", "patience", "lr", "weight_decay",
              "positive_weight", "dropout", "hidden_size", "heads", "fusion_dim", "fusion_chunk_size", "smoke")}
    common["code_sha256"] = code_hashes
    common["protocol"] = "softmax-ap-checkpoint-v1"
    for exp in dict.fromkeys(args.experiments):
        for seed in dict.fromkeys(args.seeds):
            for fold in dict.fromkeys(selected_folds):
                config = dict(common, experiment=exp, architecture=EXPERIMENTS[exp], seed=seed, fold=fold)
                train_one(config, data, args.output / exp / f"seed{seed}" / f"fold{fold}", primary, ids)


if __name__ == "__main__":
    main()
