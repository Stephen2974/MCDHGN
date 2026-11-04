"""Run a fresh E1 learning-rate sweep on three GPUs.

E1 stays GATv2 + global metapath weights + no interaction. Learning rates are
0.01, 0.005, and 0.003. Every other training setting is copied from the sealed
E1 seed42 results. This launcher does not edit model or training code, does not
add a learning-rate scheduler, and does not evaluate the held-out test set.

Windows/PowerShell: python scripts/run_e1_lr_sweep.py
Read-only preflight: python scripts/run_e1_lr_sweep.py --check-files-only
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = (
    "epochs", "eval_every", "patience", "lr", "weight_decay", "positive_weight",
    "dropout", "hidden_size", "heads", "fusion_dim", "fusion_chunk_size",
)
INTEGER_OPTIONS = {
    "epochs", "eval_every", "patience", "hidden_size", "heads",
    "fusion_dim", "fusion_chunk_size",
}
LEARNING_RATES = ("0.01", "0.005", "0.003")
METRICS = ("ap", "auc", "macro_f1")
COMPARE_FILES = ("config.json", "result.json", "history.jsonl", "validation_predictions.npz")
DEFAULT_BASELINE = ROOT / "results/p0_legacy_D0/E1/seed42"
ASSIGNMENTS = ((0, (0, 3)), (1, (1, 4)), (2, (2,)))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def format_arg(value):
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def lr_dirname(text):
    return "lr" + text


def validate_baseline(baseline):
    sys.path.insert(0, str(ROOT))
    from ablation_config import EXPERIMENTS
    paths = sorted(baseline.rglob("result.json"))
    require(len(paths) == 5, "E1 baseline must contain exactly five result.json files")
    records = {}
    for path in paths:
        row = read_json(path)
        cfg = row["config"]
        require(cfg["experiment"] == "E1" and cfg["seed"] == 42 and not cfg["smoke"],
                "Expected real E1, base seed 42")
        architecture = EXPERIMENTS["E1"]
        require(architecture["use_gatv2"] is True and architecture["semantic_mode"] == "global"
                and architecture["fusion_mode"] == "none",
                "E1 must stay GATv2 + global weights + no interaction")
        require(cfg["architecture"] == architecture, "Unexpected E1 architecture")
        fold = cfg["fold"]
        require(fold not in records, "Duplicate E1 fold")
        saved = read_json(path.with_name("config.json"))
        require(saved["config"] == cfg and saved["data_sha256"] == row["data_sha256"],
                "E1 config/result mismatch")
        records[fold] = (path, row)
    require(set(records) == set(range(5)), "E1 must contain folds 0 through 4")
    reference = records[0][1]
    common = {key: value for key, value in reference["config"].items() if key != "fold"}
    for _, row in records.values():
        require({key: value for key, value in row["config"].items() if key != "fold"} == common,
                "E1 folds have different training configurations")
        require(row["data_sha256"] == reference["data_sha256"], "E1 data fingerprints differ")
        require(row["environment"] == reference["environment"], "E1 environments differ")
    for name, expected in dict(reference["data_sha256"], **common["code_sha256"]).items():
        path = (ROOT / name).resolve()
        require(ROOT in path.parents, "Fingerprint path is outside project")
        require(path.is_file() and digest(path) == expected, "E1 hash mismatch: " + name)
    for name in OPTIONS:
        require(name in common, "Missing E1 hyperparameter: " + name)
    require(common["lr"] == 0.01, "Recorded E1 learning rate must be 0.01")
    require("scheduler" not in common, "Recorded E1 config unexpectedly contains a scheduler")
    return records


ENV_CHECK = r'''
import json, sys, torch, dgl
import dgl.function as fn
expected = json.load(open(sys.argv[1], encoding="utf-8"))["environment"]
actual = dict(python=sys.version, torch=torch.__version__, dgl=dgl.__version__,
              cuda=torch.version.cuda)
print(json.dumps(actual, indent=2), flush=True)
for name in ("python", "torch", "dgl", "cuda"):
    assert actual[name] == expected[name], ("Environment differs from E1", name, expected[name], actual[name])
assert torch.cuda.is_available() and torch.cuda.device_count() >= 3, "Need three visible CUDA GPUs"
for i in range(3):
    name = torch.cuda.get_device_name(i)
    assert name == expected["gpus"][0], ("Unexpected GPU", i, name)
    device = torch.device("cuda:" + str(i))
    graph = dgl.graph(([0, 1, 2], [1, 2, 0])).to(device)
    graph.ndata["x"] = torch.ones(3, 2, device=device)
    graph.update_all(fn.copy_u("x", "m"), fn.sum("m", "y"))
    assert torch.allclose(graph.ndata["y"], graph.ndata["x"])
    torch.cuda.synchronize(i)
    print("PASS GPU", i, name, flush=True)
'''


def training_args(reference, gpu, folds, output, lr, probe=False):
    cfg = reference["config"]
    args = [sys.executable, str(ROOT / "run_ablation.py"), "--experiments", "E1",
            "--seeds", str(cfg["seed"]), "--folds"] + [str(fold) for fold in folds]
    args += ["--devices", str(gpu), "--data-root", str(ROOT), "--output", str(output)]
    changes = {"epochs": 2, "eval_every": 1, "patience": 0} if probe else {}
    for key in OPTIONS:
        value = lr if key == "lr" else changes.get(key, cfg[key])
        args += ["--" + key.replace("_", "-"), format_arg(value)]
    require("--test-run" not in args and "--smoke" not in args, "Formal launcher must not call test or smoke mode")
    require(not any("scheduler" in item.lower() for item in args), "Launcher must not add a scheduler")
    return args


def execute(argv, log, env):
    print("Running:", subprocess.list2cmdline(argv), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            code = process.wait()
        except BaseException:
            stop(process)
            raise
    require(code == 0, "Step failed; see " + str(log))


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_workers(commands, logs, env):
    workers, streams = [], []
    try:
        for argv, log in zip(commands, logs):
            stream = log.open("w", encoding="utf-8")
            streams.append(stream)
            workers.append(subprocess.Popen(
                argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
            print("Started PID", workers[-1].pid, "log:", log, flush=True)
        next_status = 0
        while True:
            states = [process.poll() for process in workers]
            require(all(state in (None, 0) for state in states),
                    "A GPU worker failed; stopping this run. Check worker logs.")
            if all(state == 0 for state in states):
                break
            if time.monotonic() >= next_status:
                print("GPU 0/1/2 worker states:", states, "(None = running)", flush=True)
                next_status = time.monotonic() + 60
            time.sleep(5)
    finally:
        for process in workers:
            stop(process)
        for stream in streams:
            stream.close()


def training_processes(rows):
    """WDDM lists desktop apps as compute apps. Only Python workloads block a launch."""
    return [row for row in rows if "python" in row.lower()]


def gpu_compute_apps():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        text=True, stderr=subprocess.STDOUT)
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    return training_processes(rows)


def wait_for_idle_gpus():
    while True:
        rows = gpu_compute_apps()
        if not rows:
            print("PASS: no other GPU compute process", flush=True)
            return
        print("Other GPU compute work is still running; waiting and not stopping it:", rows, flush=True)
        time.sleep(60)


def child_env():
    return dict(os.environ, DGLBACKEND="pytorch", PYTHONUNBUFFERED="1",
                PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="0,1,2")


def copy_reference(records, destination):
    manifest = []
    for fold, (path, _) in records.items():
        dest = destination / "E1" / "seed42" / ("fold" + str(fold))
        dest.mkdir(parents=True)
        for name in COMPARE_FILES:
            source = path.with_name(name)
            target = dest / name
            shutil.copy2(source, target)
            require(digest(target) == digest(source), "Reference copy hash mismatch: " + name)
            manifest.append({"fold": fold, "name": name, "sha256": digest(target)})
    return manifest


def verify_lr(output, records, lr_text):
    root = output / "train" / lr_dirname(lr_text)
    for fold, (_, record) in records.items():
        run_dir = root / "E1" / "seed42" / ("fold" + str(fold))
        row = read_json(run_dir / "result.json")
        expected = dict(record["config"])
        expected["lr"] = float(lr_text)
        require(row["config"] == expected, "Completed E1 differs from the recorded protocol: lr %s fold %s" % (lr_text, fold))
        require(row["data_sha256"] == record["data_sha256"], "Data fingerprint differs: lr %s fold %s" % (lr_text, fold))
        require(row["environment"] == record["environment"], "Environment differs: lr %s fold %s" % (lr_text, fold))
        require((run_dir / "best.pt").is_file(), "Missing checkpoint: " + str(run_dir))
        require(not (run_dir / "test_result.json").exists(), "Test-set output is not part of this sweep")
        require(isinstance(row["epochs_trained"], int) and row["epochs_trained"] <= expected["epochs"],
                "Epoch count exceeds the recorded maximum")


def metric_block(rows):
    block = {}
    for metric in METRICS:
        values = [row["validation"][metric] for row in rows]
        block[metric] = {
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values),
            "n": len(values),
            "folds": values,
        }
    return block


def paired_block(candidate, baseline):
    block = {}
    for metric in METRICS:
        deltas = [new["validation"][metric] - old["validation"][metric] for new, old in zip(candidate, baseline)]
        block[metric] = {
            "mean_delta": statistics.mean(deltas),
            "mean_delta_pp": 100 * statistics.mean(deltas),
            "sd_delta": statistics.stdev(deltas),
            "wins": int(sum(delta > 0 for delta in deltas)),
            "ties": int(sum(delta == 0 for delta in deltas)),
            "losses": int(sum(delta < 0 for delta in deltas)),
            "pairs": [{"fold": fold, "delta": delta, "delta_pp": 100 * delta} for fold, delta in enumerate(deltas)],
        }
    return block


def write_comparison(output, records):
    grouped = {}
    cap = []
    for lr_text in LEARNING_RATES:
        rows = []
        for fold in range(5):
            row = read_json(output / "train" / lr_dirname(lr_text) / "E1" / "seed42" / ("fold" + str(fold)) / "result.json")
            rows.append(row)
            if row["epochs_trained"] >= row["config"]["epochs"]:
                cap.append({"lr": lr_text, "fold": fold, "epochs_trained": row["epochs_trained"],
                            "best_epoch": row["best_epoch"], "epoch_limit": row["config"]["epochs"]})
        grouped[lr_text] = rows
    old_rows = [records[fold][1] for fold in range(5)]
    comparison = output / "comparison"
    comparison.mkdir(parents=True)
    summary = {
        "experiment": "E1",
        "architecture": "GATv2 + global weights + no interaction",
        "seed": 42,
        "learning_rates": list(LEARNING_RATES),
        "sd_definition": "sample standard deviation across five folds, statistics.stdev (ddof=1)",
        "selection": "validation AP checkpoint only; held-out test set was not evaluated",
        "groups": {lr_text: metric_block(grouped[lr_text]) for lr_text in LEARNING_RATES},
        "paired_vs_this_run_lr001": {
            lr_text: paired_block(grouped[lr_text], grouped["0.01"])
            for lr_text in LEARNING_RATES if lr_text != "0.01"
        },
        "epoch_cap": cap,
        "reproduction_new_lr001_minus_recorded_lr001": paired_block(grouped["0.01"], old_rows),
        "recorded_lr001_epochs": [
            {"fold": fold, "best_epoch": old_rows[fold]["best_epoch"], "epochs_trained": old_rows[fold]["epochs_trained"]}
            for fold in range(5)
        ],
        "new_epochs": [
            {"lr": lr_text, "fold": fold, "best_epoch": grouped[lr_text][fold]["best_epoch"],
             "epochs_trained": grouped[lr_text][fold]["epochs_trained"]}
            for lr_text in LEARNING_RATES for fold in range(5)
        ],
    }
    (comparison / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    records_out = []
    for lr_text in LEARNING_RATES:
        for fold, row in enumerate(grouped[lr_text]):
            records_out.append({
                "lr": lr_text, "seed": 42, "fold": fold,
                "best_epoch": row["best_epoch"], "epochs_trained": row["epochs_trained"],
                "hit_epoch_cap": row["epochs_trained"] >= row["config"]["epochs"],
                "train_seconds": row["train_seconds"],
                **{metric: row["validation"][metric] for metric in METRICS},
            })
    import csv
    with (comparison / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records_out[0]))
        writer.writeheader()
        writer.writerows(records_out)
    lines = [
        "E1 学习率对比（seed 42，每组完整五折）",
        "验证集 AP 选最佳 checkpoint。未调用测试集。折间标准差为样本标准差（n-1）。",
        "配对差值 = 候选学习率 − 本轮 0.01。正值表示候选更高。",
        "",
    ]
    for lr_text in LEARNING_RATES:
        lines.append("学习率 %s" % lr_text)
        for metric in METRICS:
            stats = summary["groups"][lr_text][metric]
            lines.append("  %s: mean %.6f, sd %.6f" % (metric, stats["mean"], stats["sd"]))
        lines.append("")
    for lr_text, block in summary["paired_vs_this_run_lr001"].items():
        lines.append("相对本轮 0.01：%s" % lr_text)
        for metric in METRICS:
            item = block[metric]
            lines.append("  %s: mean delta %+.6f (%+.4f pp), wins %d/5, ties %d, losses %d" % (
                metric, item["mean_delta"], item["mean_delta_pp"], item["wins"], item["ties"], item["losses"]))
        lines.append("")
    lines.append("达到记录轮数上限的折：")
    lines.extend(["  lr %s fold %s: epochs_trained %s / %s, best_epoch %s" % (
        item["lr"], item["fold"], item["epochs_trained"], item["epoch_limit"], item["best_epoch"]) for item in cap] or ["  无"])
    lines.append("")
    lines.append("新旧 0.01 复现差异（本轮 − 既有 E1 seed42）：")
    reproduced = summary["reproduction_new_lr001_minus_recorded_lr001"]
    for metric in METRICS:
        item = reproduced[metric]
        lines.append("  %s: mean delta %+.6f (%+.4f pp), sd_delta %.6f" % (
            metric, item["mean_delta"], item["mean_delta_pp"], item["sd_delta"]))
        for pair in item["pairs"]:
            lines.append("    fold %s: %+.6f" % (pair["fold"], pair["delta"]))
    text = "\n".join(lines) + "\n"
    (comparison / "comparison.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return summary


def pack(output):
    archive = output.with_name(output.name + "_feedback.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in {"best.pt", "gene_metapath_weights.npz"}:
                tar.add(path, arcname=str(Path(output.name) / path.relative_to(output)))
    sidecar = Path(str(archive) + ".sha256")
    sidecar.write_text("%s  %s\n" % (digest(archive), archive.name), encoding="utf-8")
    print("Feedback archive:", archive, flush=True)
    print("Checkpoints retained under:", output / "train", flush=True)
    return archive


def git_text(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return "unavailable: " + repr(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-files-only", action="store_true",
                        help="Check E1 configs and code/data hashes; do not train")
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    records = validate_baseline(baseline)
    print("PASS: five E1 folds and current code/data hashes agree", flush=True)
    if args.check_files_only:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output or ROOT / "results" / ("e1_lr_sweep_" + stamp)).resolve()
    require(output != baseline and baseline not in output.parents and output not in baseline.parents,
            "Output and baseline must be separate directories")
    output.mkdir(parents=True, exist_ok=False)
    print("Output directory:", output, flush=True)
    logs = output / "logs"
    logs.mkdir()
    env = child_env()
    status = {
        "status": "running", "baseline": str(baseline), "output": str(output),
        "python": sys.executable, "learning_rates": list(LEARNING_RATES),
        "assignments": ASSIGNMENTS, "launcher_sha256": digest(Path(__file__)),
        "git_head": git_text(["git", "rev-parse", "HEAD"]),
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
    }
    reference = records[0][1]
    try:
        shutil.copy2(__file__, output / "run_e1_lr_sweep.py")
        status["reference_files"] = copy_reference(records, output / "reference_lr001")
        (output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        wait_for_idle_gpus()
        execute([sys.executable, "-m", "data_processing.p0.release", "verify"], logs / "01_integrity.log", env)
        execute([sys.executable, "-c", ENV_CHECK, str(records[0][0])], logs / "02_environment.log", env)
        execute([sys.executable, "-m", "pytest", "tests", "-q"], logs / "03_tests.log", env)
        execute([sys.executable, "-m", "data_processing.p0.validate_graph", "--report",
                 str(output / "graph_validation_server.json")], logs / "04_graph.log", env)
        wait_for_idle_gpus()
        execute(training_args(reference, 0, (0,), output / "probe", "0.01", probe=True),
                logs / "05_real_E1_probe.log", env)
        for lr_text in LEARNING_RATES:
            status["status"] = "training_" + lr_text
            (output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            wait_for_idle_gpus()
            commands = [training_args(reference, gpu, folds, output / "train" / lr_dirname(lr_text), lr_text)
                        for gpu, folds in ASSIGNMENTS]
            status.setdefault("commands", {})[lr_text] = commands
            (output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            run_workers(commands, [logs / ("train_%s_gpu%d.log" % (lr_dirname(lr_text), gpu))
                                   for gpu, _ in ASSIGNMENTS], env)
            verify_lr(output, records, lr_text)
        validate_baseline(output / "reference_lr001" / "E1" / "seed42")
        summary = write_comparison(output, records)
        status["status"] = "completed"
        status["epoch_cap_count"] = len(summary["epoch_cap"])
    except BaseException as exc:
        status.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if output.exists():
            (output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            try:
                pack(output)
            except Exception as pack_error:
                print("Feedback pack failed:", repr(pack_error), flush=True)


if __name__ == "__main__":
    main()
