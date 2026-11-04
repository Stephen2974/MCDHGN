"""Run E1 and E3 at lr=0.003 without modifying project training code."""

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
OLD = {
    "E1": ROOT / "results" / "p0_legacy_D0" / "E1" / "seed42",
    "E3": ROOT / "results" / "d0_E3_seed42_20261103_153158" / "train" / "E3" / "seed42",
}
COMPARE_NAMES = ("config.json", "result.json", "history.jsonl", "validation_predictions.npz")
ASSIGNMENTS = ((0, (0, 3)), (1, (1, 4)), (2, (2,)))
FLAGS = [
    "--epochs", "1000", "--eval-every", "10", "--patience", "20", "--lr", "0.003",
    "--weight-decay", "0.0001", "--positive-weight", "1.904", "--dropout", "0.4",
    "--hidden-size", "256", "--heads", "4", "--fusion-dim", "64", "--fusion-chunk-size", "1024",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(status):
    (OUT / "run_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def env():
    values = dict(os.environ)
    values["DGLBACKEND"] = "pytorch"
    values["PYTHONUNBUFFERED"] = "1"
    values["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    return values


def command(experiment, folds, gpu, output, probe=False):
    argv = [str(PY), str(ROOT / "run_ablation.py"), "--experiments", experiment,
            "--seeds", "42", "--folds", *[str(fold) for fold in folds],
            "--devices", str(gpu), "--data-root", str(ROOT), "--output", str(output)]
    if probe:
        argv += ["--epochs", "2", "--eval-every", "1", "--patience", "0", "--lr", "0.003",
                 "--weight-decay", "0.0001", "--positive-weight", "1.904", "--dropout", "0.4",
                 "--hidden-size", "256", "--heads", "4", "--fusion-dim", "64",
                 "--fusion-chunk-size", "1024"]
    else:
        argv += FLAGS
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
                failed = [(str(log), code) for (_, _, log, _), code in zip(processes, codes) if code not in (None, 0)]
                raise RuntimeError("Worker failed in %s: %s" % (label, failed))
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


def require_results(experiment):
    for fold in range(5):
        path = OUT / "train" / experiment / "seed42" / ("fold" + str(fold)) / "result.json"
        if not path.is_file():
            raise RuntimeError("Missing result: " + str(path))


def copy_checked(src, dest, manifest):
    src_hash = sha256(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    if sha256(dest) != src_hash or sha256(src) != src_hash:
        raise RuntimeError("Copy hash mismatch: " + str(src))
    manifest.append({"source": str(src), "dest": str(dest), "sha256": src_hash})


def copy_references():
    manifest = []
    for experiment, source_root in OLD.items():
        for fold in range(5):
            source = source_root / ("fold" + str(fold))
            dest = OUT / "reference_lr001" / experiment / "seed42" / ("fold" + str(fold))
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


def pack(output):
    included = []
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"best.pt", "gene_metapath_weights.npz", "transfer_manifest.json"}:
            relative = str(Path(output.name) / path.relative_to(output)).replace("\\", "/")
            included.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
            files.append(path)
    manifest_path = output / "transfer_manifest.json"
    manifest_path.write_text(json.dumps({
        "note": "Transfer manifest for this feedback archive only. This is not a P0 sealed manifest.",
        "files": included,
    }, indent=2), encoding="utf-8")
    files.append(manifest_path)
    archive = output.with_name(output.name + "_feedback.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(Path(output.name) / path.relative_to(output)))
    digest = sha256(archive)
    Path(str(archive) + ".sha256").write_text("%s  %s\n" % (digest, archive.name), encoding="utf-8")
    print("Feedback archive:", archive, flush=True)
    print("SHA256:", digest, flush=True)
    print("Bytes:", archive.stat().st_size, flush=True)
    return {"archive": str(archive), "sha256": digest, "bytes": archive.stat().st_size}


def main():
    if (OUT / "train").exists():
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
        "compare_sha256": sha256(Path(__file__).with_name("compare_lr.py")),
    }
    write_status(status)
    if Path(sys.executable).resolve() != PY.resolve():
        raise RuntimeError("Scheduler Python is not the confirmed interpreter: " + sys.executable)
    try:
        for experiment in ("E1", "E3"):
            status["status"] = "probe_" + experiment
            write_status(status)
            run_group([(command(experiment, (0,), 0, OUT / "probe", probe=True),
                        OUT / "logs" / ("probe_" + experiment + ".log"))], status, "probe_" + experiment)
        for experiment in ("E1", "E3"):
            status["status"] = experiment
            write_status(status)
            jobs = [(command(experiment, folds, gpu, OUT / "train"),
                     OUT / "logs" / ("%s_gpu%d.log" % (experiment, gpu)))
                    for gpu, folds in ASSIGNMENTS]
            run_group(jobs, status, experiment)
            require_results(experiment)
        status["status"] = "compare"
        write_status(status)
        execute([str(PY), "-m", "data_processing.p0.release", "verify"], OUT / "logs" / "p0_verify_after.log")
        copy_references()
        execute([str(PY), str(ROOT / "summarize_ablations.py"), str(OUT / "train"),
                 "--expected-folds", "5", "--output", str(OUT / "comparison_lr003")],
                OUT / "logs" / "summary_lr003.log")
        execute([str(PY), str(ROOT / "summarize_ablations.py"), str(OUT / "reference_lr001"),
                 "--expected-folds", "5", "--output", str(OUT / "comparison_lr001")],
                OUT / "logs" / "summary_lr001.log")
        execute([str(PY), str(OUT / "compare_lr.py"), str(OUT)], OUT / "logs" / "lr_comparison.log")
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
