"""TensorBoard view of ablation history.jsonl files.

The training queue re-reads run_ablation.py for every fold and stores its
hash, so this module never imports the trainer and never writes into a
campaign whose queue.pid is still alive. Curves come only from values that
were already stored.
"""

import ctypes
import json
import math
import os
import socket
import struct
import time
from pathlib import Path

from ablation_config import METAPATHS

# history.jsonl keys that already have a stable TensorBoard tag.
# Any other finite number is kept as extra/<key>, so a later training
# change shows up without rewriting old files.
SCALAR_TAGS = {
    "train_loss": "train/loss",
    "loss": "val/loss",
    "ap": "val/ap",
    "auc": "val/auc",
    "accuracy": "val/accuracy",
    "macro_f1": "val/macro_f1",
    "precision_at_50": "val/precision_at_50",
    "precision_at_100": "val/precision_at_100",
    "recall_at_50": "val/recall_at_50",
    "recall_at_100": "val/recall_at_100",
    "effective_k_50": "val/effective_k_50",
    "effective_k_100": "val/effective_k_100",
    "grad_norm": "train/grad_norm",
    "lr": "train/lr",
    "step_seconds": "time/step_seconds",
    "stale": "train/stale",
    "score_mean": "val/score_mean",
    "score_std": "val/score_std",
    "pred_pos_rate": "val/pred_pos_rate",
    "train_ap": "train/ap",
    "train_auc": "train/auc",
    "weight_entropy": "weights/entropy",
}

_CRC32C_POLY = 0x82F63B78
_CRC32C_TABLE = []
for _index in range(256):
    _crc = _index
    for _ in range(8):
        _crc = (_crc >> 1) ^ _CRC32C_POLY if _crc & 1 else _crc >> 1
    _CRC32C_TABLE.append(_crc)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def scalars_from_row(row):
    """Yield (tag, step, value) for one history.jsonl object."""
    step = int(row["epoch"])
    for key, value in row.items():
        if key == "epoch" or not finite_number(value):
            continue
        yield SCALAR_TAGS.get(key, f"extra/{key}"), step, float(value)


def _path_name(index):
    if index < len(METAPATHS):
        return METAPATHS[index]
    return f"path_{index}"


def scalars_from_result(result):
    """Single-point series stored only in result.json, not in the curve."""
    diagnostics = result.get("weight_diagnostics") or {}
    step = int(result.get("best_epoch") or 0)
    points = []
    for key, prefix in (("mean_by_path", "weights/mean"), ("std_by_path", "weights/std")):
        for index, value in enumerate(diagnostics.get(key) or []):
            if finite_number(value):
                points.append((f"{prefix}/{_path_name(index)}", step, float(value)))
    entropy = diagnostics.get("mean_entropy")
    if finite_number(entropy):
        points.append(("weights/entropy", step, float(entropy)))
    seconds = result.get("train_seconds")
    if finite_number(seconds):
        end = int(result.get("epochs_trained") or step)
        points.append(("time/train_seconds", end, float(seconds)))
    return points


def load_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc.msg}") from exc
    return rows


def points_for_run(history_path):
    history_path = Path(history_path)
    points = []
    for row in load_jsonl(history_path):
        points.extend(scalars_from_row(row))
    result_path = history_path.with_name("result.json")
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        points.extend(scalars_from_result(result))
    return points


def pid_alive(pid):
    """True when a process exists. Does not signal it."""
    pid = int(pid)
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetLastError.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if handle:
        kernel.CloseHandle(handle)
        return True
    return kernel.GetLastError() == 5


def _pidfile_alive(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    return text.isdigit() and pid_alive(int(text))


def live_campaigns(results):
    results = Path(results)
    if not results.is_dir():
        return []
    return [pidfile.parent.resolve() for pidfile in results.glob("*/queue.pid") if _pidfile_alive(pidfile)]


def _is_under(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def iter_histories(root):
    root = Path(root)
    if root.is_file() and root.name == "history.jsonl":
        yield root
        return
    if root.is_dir():
        yield from sorted(root.rglob("history.jsonl"))


def runs_to_convert(results, logdir, selected=None):
    """Return (histories, skipped live campaigns). Live queues are not read."""
    results = Path(results).resolve()
    logdir = Path(logdir).resolve()
    live = set(live_campaigns(results))
    roots = [Path(item).resolve() for item in selected] if selected else [results]
    for root in roots:
        pidfile = root / "queue.pid"
        if pidfile.is_file() and _pidfile_alive(pidfile):
            live.add(root)
    chosen = []
    skipped = []
    seen = set()
    for root in roots:
        if any(root == campaign or _is_under(root, campaign) for campaign in live):
            skipped.append(root)
            continue
        for history in iter_histories(root):
            resolved = history.resolve()
            if resolved in seen or _is_under(resolved, logdir):
                continue
            if "tensorboard" in resolved.parts:
                continue
            owner = next((campaign for campaign in live if _is_under(resolved, campaign)), None)
            if owner is not None:
                if owner not in skipped:
                    skipped.append(owner)
                continue
            seen.add(resolved)
            chosen.append(resolved)
    return chosen, skipped


def _crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data):
    # Same mask as TensorBoard's record writer: rotate the uint32 CRC, then add.
    crc = _crc32c(data) & 0xFFFFFFFF
    rotated = ((crc >> 15) | ((crc << 17) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return (rotated + 0xA282EAD8) & 0xFFFFFFFF


def _varint(value):
    if value < 0:
        raise ValueError("negative protobuf varint")
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field(field_number, wire_type, payload):
    return _varint((field_number << 3) | wire_type) + payload


def _record(payload):
    header = struct.pack("<Q", len(payload))
    return header + struct.pack("<I", _masked_crc32c(header)) + payload + struct.pack("<I", _masked_crc32c(payload))


def _summary_event(wall_time, step, pairs):
    values = b""
    for tag, value in pairs:
        body = _field(1, 2, _varint(len(tag)) + tag.encode("utf-8"))
        body += _field(2, 5, struct.pack("<f", float(value)))
        values += _field(1, 2, _varint(len(body)) + body)
    event = _field(1, 1, struct.pack("<d", wall_time))
    event += _field(2, 0, _varint(int(step)))
    event += _field(5, 2, _varint(len(values)) + values)
    return event


def _file_version_event(wall_time):
    version = b"brain.Event:2"
    event = _field(1, 1, struct.pack("<d", wall_time))
    event += _field(3, 2, _varint(len(version)) + version)
    return event


def write_events(log_dir, points):
    """Write one TensorBoard event file. Replaces older event files in that run."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for old in log_dir.glob("events.out.tfevents.*"):
        old.unlink()
    if not points:
        return None
    grouped = {}
    for tag, step, value in points:
        if not finite_number(value):
            continue
        grouped.setdefault(int(step), {})[str(tag)] = float(value)
    hostname = "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in socket.gethostname()) or "host"
    path = log_dir / f"events.out.tfevents.{int(time.time())}.{hostname}"
    wall = time.time()
    with path.open("wb") as handle:
        handle.write(_record(_file_version_event(wall)))
        for step in sorted(grouped):
            handle.write(_record(_summary_event(wall, step, grouped[step].items())))
    return path


def write_run(history_path, log_dir):
    points = points_for_run(history_path)
    write_events(log_dir, points)
    return len(points)


def _relative_run(history, results):
    parent = history.parent.resolve()
    try:
        return parent.relative_to(Path(results).resolve())
    except ValueError:
        return Path(parent.name)


def convert_results(results, logdir, selected=None):
    histories, skipped = runs_to_convert(results, logdir, selected)
    written = []
    errors = []
    for history in histories:
        destination = Path(logdir) / _relative_run(history, results)
        try:
            count = write_run(history, destination)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append((history, str(exc)))
            continue
        written.append((history, destination, count))
    return {"written": written, "skipped": skipped, "errors": errors}
