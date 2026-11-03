"""Run paired E3 folds on three GPUs without changing sealed training code.

Windows/PowerShell: python scripts/run_e3_parallel.py
Read-only preflight: python scripts/run_e3_parallel.py --check-files-only
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = (
    "epochs", "eval_every", "patience", "lr", "weight_decay", "positive_weight",
    "dropout", "hidden_size", "heads", "fusion_dim", "fusion_chunk_size",
)
DEFAULT_BASELINE = ROOT / "results/p0_legacy_D0/E1/seed42"
ASSIGNMENTS = ((0, (0, 3)), (1, (1, 4)), (2, (2,)))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


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
        require(cfg["architecture"] == EXPERIMENTS["E1"], "Unexpected E1 architecture")
        fold = cfg["fold"]
        require(fold not in records, "Duplicate E1 fold")
        saved = read_json(path.with_name("config.json"))
        require(saved["config"] == cfg and saved["data_sha256"] == row["data_sha256"],
                "E1 config/result mismatch")
        records[fold] = (path, row)
    require(set(records) == set(range(5)), "E1 must contain folds 0 through 4")
    reference = records[0][1]
    common = {k: v for k, v in reference["config"].items() if k != "fold"}
    for _, row in records.values():
        require({k: v for k, v in row["config"].items() if k != "fold"} == common,
                "E1 folds have different training configurations")
        require(row["data_sha256"] == reference["data_sha256"], "E1 data fingerprints differ")
        require(row["environment"] == reference["environment"], "E1 environments differ")
    for name, expected in dict(reference["data_sha256"], **common["code_sha256"]).items():
        path = (ROOT / name).resolve()
        require(ROOT in path.parents, "Fingerprint path is outside project")
        require(path.is_file() and digest(path) == expected, "E1 hash mismatch: " + name)
    e3_expected = dict(EXPERIMENTS["E1"], fusion_mode="cross")
    require(EXPERIMENTS["E3"] == e3_expected, "E3 must change only fusion_mode to cross")
    for name in OPTIONS:
        require(name in common, "Missing E1 hyperparameter: " + name)
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


def training_args(reference, gpu, folds, output, probe=False):
    cfg = reference["config"]
    args = [sys.executable, str(ROOT / "run_ablation.py"), "--experiments", "E3",
            "--seeds", str(cfg["seed"]), "--folds"] + [str(f) for f in folds]
    args += ["--devices", str(gpu), "--data-root", str(ROOT), "--output", str(output)]
    changes = {"epochs": 2, "eval_every": 1, "patience": 0} if probe else {}
    for key in OPTIONS:
        args += ["--" + key.replace("_", "-"), str(changes.get(key, cfg[key]))]
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
            process.wait(timeout=15)
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
            states = [p.poll() for p in workers]
            require(all(s in (None, 0) for s in states),
                    "A GPU worker failed; stopping this run. Check worker logs.")
            if all(s == 0 for s in states):
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


def pack(output):
    archive = output.with_name(output.name + "_feedback.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in {"best.pt", "gene_metapath_weights.npz"}:
                tar.add(path, arcname=str(Path(output.name) / path.relative_to(output)))
    print("Feedback archive:", archive, flush=True)


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
    output = (args.output or ROOT / "results" / ("d0_E3_parallel_" + stamp + "_" + str(os.getpid()))).resolve()
    require(output != baseline and baseline not in output.parents and output not in baseline.parents,
            "Output and baseline must be separate directories")
    output.mkdir(parents=True, exist_ok=False)
    print("Output directory:", output, flush=True)
    logs = output / "logs"
    logs.mkdir()
    env = dict(os.environ, DGLBACKEND="pytorch", PYTHONUNBUFFERED="1")
    status = {"status": "running", "baseline": str(baseline), "output": str(output),
              "python": sys.executable, "assignments": ASSIGNMENTS,
              "launcher_sha256": digest(Path(__file__)),
              "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES")}
    reference = records[0][1]
    snapshot = output / "reference_E1"
    try:
        shutil.copy2(__file__, output / "run_e3_parallel.py")
        for fold, (path, _) in records.items():
            dest = snapshot / "E1" / "seed42" / ("fold" + str(fold))
            dest.mkdir(parents=True)
            for name in ("config.json", "result.json", "history.jsonl", "validation_predictions.npz"):
                shutil.copy2(path.with_name(name), dest / name)
        execute([sys.executable, "-m", "data_processing.p0.release", "verify"], logs / "01_integrity.log", env)
        execute([sys.executable, "-c", ENV_CHECK, str(records[0][0])], logs / "02_environment.log", env)
        execute([sys.executable, "-m", "pytest", "tests", "-q"], logs / "03_tests.log", env)
        execute([sys.executable, "-m", "data_processing.p0.validate_graph", "--report",
                 str(output / "graph_validation_server.json")], logs / "04_graph.log", env)
        execute(training_args(reference, 0, (0,), output / "probe", probe=True), logs / "05_real_E3_probe.log", env)
        commands = [training_args(reference, gpu, folds, output / "train") for gpu, folds in ASSIGNMENTS]
        status["commands"] = commands
        (output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        run_workers(commands, [logs / ("gpu" + str(g) + ".log") for g, _ in ASSIGNMENTS], env)
        # Re-check files before combining results, including the preserved reference.
        validate_baseline(snapshot)
        for fold in range(5):
            row = read_json(output / "train/E3/seed42" / ("fold" + str(fold)) / "result.json")
            expected = dict(records[fold][1]["config"], experiment="E3")
            expected["architecture"] = dict(expected["architecture"], fusion_mode="cross")
            require(row["config"] == expected and row["data_sha256"] == reference["data_sha256"],
                    "Completed E3 differs from the paired E1 protocol")
            require(row["environment"] == reference["environment"], "E3 environment differs from E1")
        execute([sys.executable, str(ROOT / "summarize_ablations.py"), str(snapshot), str(output / "train"),
                 "--expected-folds", "5", "--output", str(output / "comparison")], logs / "06_comparison.log", env)
        comparison = read_json(output / "comparison/summary.json")
        metrics = comparison["paired_contrasts"]["cross_view_with_global_weights"]["metrics"]
        lines = ["E3 - E1 paired validation comparison (seed 42, five folds)",
                 "Validation model selection only; no independent-test or significance claim."]
        for metric, value in metrics.items():
            lines.append("{}: {:+.4f} percentage points; wins {}/5".format(
                metric, value["mean_delta_pp"], value["wins"]))
        (output / "comparison/comparison.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines), flush=True)
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        (output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        pack(output)


if __name__ == "__main__":
    main()
