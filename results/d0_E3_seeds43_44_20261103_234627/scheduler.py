"""Schedule E3 seeds 43 and 44 without modifying project training code.

Probe, then seed 43, then seed 44. One fold per GPU. This file is a run record.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


ROOT = Path(r"G:\LLLM\MCDHGN-main")
PY = Path(r"G:\LLLM\MCDHGN-main\.conda\python.exe")
OUT = Path(sys.argv[1]).resolve()
E1_ROOT = ROOT / "results" / "p0_legacy_D0" / "E1"
E3_SEED42 = ROOT / "results" / "d0_E3_seed42_20261103_153158" / "train" / "E3" / "seed42"
COMPARE_NAMES = ("config.json", "result.json", "history.jsonl", "validation_predictions.npz")
ASSIGNMENTS = ((0, (0, 3)), (1, (1, 4)), (2, (2,)))
TRAIN_FLAGS = [
    "--epochs", "1000", "--eval-every", "10", "--patience", "20",
    "--lr", "0.01", "--weight-decay", "0.0001", "--positive-weight", "1.904",
    "--dropout", "0.4", "--hidden-size", "256", "--heads", "4",
    "--fusion-dim", "64", "--fusion-chunk-size", "1024",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_status(status):
    (OUT / "run_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def env():
    values = dict(os.environ)
    values["DGLBACKEND"] = "pytorch"
    values["PYTHONUNBUFFERED"] = "1"
    values["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    return values


def command(seed, folds, gpu, output, probe=False):
    argv = [str(PY), str(ROOT / "run_ablation.py"), "--experiments", "E3",
            "--seeds", str(seed), "--folds", *[str(fold) for fold in folds],
            "--devices", str(gpu), "--data-root", str(ROOT), "--output", str(output)]
    if probe:
        argv += ["--epochs", "2", "--eval-every", "1", "--patience", "0",
                 "--lr", "0.01", "--weight-decay", "0.0001", "--positive-weight", "1.904",
                 "--dropout", "0.4", "--hidden-size", "256", "--heads", "4",
                 "--fusion-dim", "64", "--fusion-chunk-size", "1024"]
    else:
        argv += TRAIN_FLAGS
    return argv


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_group(jobs, status, label):
    processes = []
    try:
        for argv, log in jobs:
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                argv, cwd=ROOT, env=env(), stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            processes.append((process, stream, log, argv))
            print("Started PID", process.pid, "log:", log, flush=True)
        status.setdefault("pids", {})[label] = [process.pid for process, _, _, _ in processes]
        status.setdefault("commands", {})[label] = [argv for _, _, _, argv in processes]
        write_status(status)
        next_report = 0.0
        while True:
            codes = [process.poll() for process, _, _, _ in processes]
            if any(code not in (None, 0) for code in codes):
                for process, _, _, _ in processes:
                    stop(process)
                failed = [(log, code) for (process, _, log, _), code in zip(processes, codes) if code not in (None, 0)]
                raise RuntimeError("Worker failed in %s: %s" % (label, [(str(log), code) for log, code in failed]))
            if all(code == 0 for code in codes):
                status.setdefault("exit_codes", {})[label] = codes
                write_status(status)
                return
            now = time.monotonic()
            if now >= next_report:
                print(label, "states:", codes, "(None = running)", flush=True)
                next_report = now + 60
            time.sleep(5)
    finally:
        for process, stream, _, _ in processes:
            stop(process)
            stream.close()


def expected_from_e1(seed, fold):
    base = read_json(E1_ROOT / ("seed" + str(seed)) / ("fold" + str(fold)) / "result.json")
    architecture = dict(base["config"]["architecture"])
    architecture["fusion_mode"] = "cross"
    expected = dict(base["config"], experiment="E3", architecture=architecture)
    return base, expected


def check_completed(seed):
    for fold in range(5):
        path = OUT / "train" / "E3" / ("seed" + str(seed)) / ("fold" + str(fold)) / "result.json"
        if not path.is_file():
            raise RuntimeError("Missing result: " + str(path))
        row = read_json(path)
        base, expected = expected_from_e1(seed, fold)
        if row["config"] != expected:
            raise RuntimeError("E3 config differs from paired E1 plus cross: seed %s fold %s" % (seed, fold))
        if row["data_sha256"] != base["data_sha256"]:
            raise RuntimeError("E3 data fingerprint differs: seed %s fold %s" % (seed, fold))
        if row["environment"] != base["environment"]:
            raise RuntimeError("E3 environment differs from E1: seed %s fold %s" % (seed, fold))


def copy_checked(src, dest, manifest):
    src_hash = sha256(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    dest_hash = sha256(dest)
    if src_hash != dest_hash or sha256(src) != src_hash:
        raise RuntimeError("Copy hash mismatch: " + str(src))
    manifest.append({"source": str(src), "dest": str(dest), "sha256": dest_hash})


def copy_references():
    manifest = []
    for seed in (42, 43, 44):
        for fold in range(5):
            source = E1_ROOT / ("seed" + str(seed)) / ("fold" + str(fold))
            dest = OUT / "reference_E1" / "E1" / ("seed" + str(seed)) / ("fold" + str(fold))
            for name in COMPARE_NAMES:
                copy_checked(source / name, dest / name, manifest)
    for fold in range(5):
        source = E3_SEED42 / ("fold" + str(fold))
        dest = OUT / "reference_E3_seed42" / "E3" / "seed42" / ("fold" + str(fold))
        for name in COMPARE_NAMES:
            copy_checked(source / name, dest / name, manifest)
    (OUT / "reference_copy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def execute(argv, log):
    print("Running:", subprocess.list2cmdline(argv), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            argv, cwd=ROOT, env=env(), stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        code = process.wait()
    if code != 0:
        raise RuntimeError("Command failed (%s): %s" % (code, subprocess.list2cmdline(argv)))


def write_comparison():
    summary = read_json(OUT / "comparison" / "summary.json")
    contrast = summary["paired_contrasts"]["cross_view_with_global_weights"]
    metrics = contrast["metrics"]
    lines = [
        "E3 - E1 paired validation comparison (seeds 42, 43, 44; five folds each)",
        "Validation model selection only. The 15 folds share splits and are not 15 independent samples.",
        "No naive significance test and no independent-test claim.",
        "",
        "Overall mean difference, E3 minus E1, percentage points; wins are folds with a positive difference.",
    ]
    for name in ("ap", "auc", "accuracy", "macro_f1"):
        row = metrics[name]
        lines.append("%s: %+.4f percentage points; wins %d/15; ties %d" % (
            name, row["mean_delta_pp"], row["wins"], row["ties"]))
    lines.append("")
    lines.append("Per-seed mean difference, percentage points")
    for seed in ("42", "43", "44"):
        parts = []
        for name in ("ap", "auc", "accuracy", "macro_f1"):
            parts.append("%s %+.4f" % (name, 100 * metrics[name]["per_seed_mean_delta"][seed]))
        wins = sum(1 for pair in metrics["ap"]["pairs"] if str(pair["seed"]) == seed and pair["delta"] > 0)
        lines.append("seed %s: %s; AP wins %d/5" % (seed, "; ".join(parts), wins))
    lines.append("")
    lines.append("Per-fold AP difference, percentage points")
    fold3 = []
    for pair in metrics["ap"]["pairs"]:
        lines.append("seed %s fold %s: %+.4f" % (pair["seed"], pair["fold"], 100 * pair["delta"]))
        if pair["fold"] == 3:
            fold3.append(pair)
    lines.append("")
    lines.append("fold3 across seeds")
    for pair in fold3:
        lines.append("seed %s fold3 AP difference: %+.4f percentage points" % (pair["seed"], 100 * pair["delta"]))
    repeated = all(pair["delta"] < 0 for pair in fold3) and len(fold3) == 3
    lines.append("fold3 AP is lower than paired E1 for every seed: %s" % ("yes" if repeated else "no"))
    lines.append("")
    lines.append("Training time and recorded peak allocated memory")
    anomalies = []
    for seed in (42, 43, 44):
        root = E3_SEED42 if seed == 42 else OUT / "train" / "E3" / ("seed" + str(seed))
        for fold in range(5):
            row = read_json(root / ("fold" + str(fold)) / "result.json")
            peak = max(row["peak_allocated_bytes"].values()) if row["peak_allocated_bytes"] else None
            lines.append("E3 seed %s fold %s: best_epoch=%s epochs_trained=%s seconds=%.1f peak_allocated_bytes=%s" % (
                seed, fold, row["best_epoch"], row["epochs_trained"], row["train_seconds"], peak))
            history = (root / ("fold" + str(fold)) / "history.jsonl").read_text(encoding="utf-8").splitlines()
            last = json.loads(history[-1])
            if row["epochs_trained"] >= 1000:
                anomalies.append("seed %s fold %s reached the 1000-epoch cap" % (seed, fold))
            if row["validation"]["ap"] - last["ap"] >= 0.05:
                anomalies.append("seed %s fold %s best AP %.4f, last logged AP %.4f" % (
                    seed, fold, row["validation"]["ap"], last["ap"]))
    lines.append("")
    lines.append("Notable training records")
    lines.extend(anomalies or ["No worker failure, epoch-cap run, or last-log AP drop of 0.05 or more was recorded."])
    (OUT / "comparison" / "comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def pack(output):
    archive = output.with_name(output.name + "_feedback.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in {"best.pt", "gene_metapath_weights.npz"}:
                tar.add(path, arcname=str(Path(output.name) / path.relative_to(output)))
    digest = sha256(archive)
    sidecar = Path(str(archive) + ".sha256")
    sidecar.write_text("%s  %s\n" % (digest, archive.name), encoding="utf-8")
    print("Feedback archive:", archive, flush=True)
    print("SHA256:", digest, flush=True)
    print("Bytes:", archive.stat().st_size, flush=True)
    return {"archive": str(archive), "sha256": digest, "bytes": archive.stat().st_size}


def main():
    if OUT.exists() and any(OUT.iterdir()) and (OUT / "train").exists():
        raise RuntimeError("Refusing to reuse an output directory that already contains train/")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    status = {
        "status": "running",
        "output": str(OUT),
        "python": str(PY),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True),
        "scheduler_sha256": sha256(Path(__file__).resolve()),
    }
    write_status(status)
    if Path(sys.executable).resolve() != PY.resolve():
        raise RuntimeError("Scheduler Python is not the confirmed interpreter: " + sys.executable)
    try:
        status["status"] = "probe"
        write_status(status)
        probe = command(43, (0,), 0, OUT / "probe", probe=True)
        run_group([(probe, OUT / "logs" / "probe.log")], status, "probe")
        status["status"] = "seed43"
        write_status(status)
        jobs = [(command(43, folds, gpu, OUT / "train"), OUT / "logs" / ("seed43_gpu%d.log" % gpu))
                for gpu, folds in ASSIGNMENTS]
        run_group(jobs, status, "seed43")
        check_completed(43)
        status["status"] = "seed44"
        write_status(status)
        jobs = [(command(44, folds, gpu, OUT / "train"), OUT / "logs" / ("seed44_gpu%d.log" % gpu))
                for gpu, folds in ASSIGNMENTS]
        run_group(jobs, status, "seed44")
        check_completed(44)
        status["status"] = "compare"
        write_status(status)
        execute([str(PY), "-m", "data_processing.p0.release", "verify"], OUT / "logs" / "p0_verify_after.log")
        copy_references()
        compare = OUT / "comparison"
        execute([str(PY), str(ROOT / "summarize_ablations.py"),
                 str(OUT / "reference_E1"), str(OUT / "reference_E3_seed42"), str(OUT / "train"),
                 "--expected-folds", "5", "--output", str(compare)], OUT / "logs" / "comparison.log")
        rows = (compare / "runs.csv").read_text(encoding="utf-8-sig").splitlines()
        if len(rows) != 31:
            raise RuntimeError("runs.csv must contain 30 data rows, found %s" % (len(rows) - 1))
        summary = read_json(compare / "summary.json")
        contrast = summary["paired_contrasts"]["cross_view_with_global_weights"]
        if contrast["candidate"] != "E3" or contrast["baseline"] != "E1":
            raise RuntimeError("Unexpected contrast direction")
        if set(summary["experiments"]["E1"]["coverage"]) != {"42", "43", "44"}:
            raise RuntimeError("E1 coverage is not seeds 42, 43, 44")
        if set(summary["experiments"]["E3"]["coverage"]) != {"42", "43", "44"}:
            raise RuntimeError("E3 coverage is not seeds 42, 43, 44")
        write_comparison()
        status["status"] = "completed"
    except BaseException as exc:
        status["status"] = "failed"
        status["error"] = repr(exc)
        print("FAILED:", repr(exc), flush=True)
        raise
    finally:
        write_status(status)
        try:
            status["feedback"] = pack(OUT)
        except BaseException as exc:
            status["pack_error"] = repr(exc)
            print("PACK FAILED:", repr(exc), flush=True)
        write_status(status)


if __name__ == "__main__":
    main()
