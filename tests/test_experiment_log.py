"""TensorBoard export from history.jsonl. No trainer import and no GPU."""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment_log as log


def test_crc32c_known_vector():
    assert log._crc32c(b"123456789") == 0xE3069283


def test_row_maps_known_metrics_and_keeps_future_numbers():
    points = dict((tag, (step, value)) for tag, step, value in log.scalars_from_row({
        "epoch": 180,
        "train_loss": 0.2,
        "ap": 0.73,
        "loss": 0.4,
        "grad_norm": 0.8,
        "note": "skip",
        "flag": True,
        "missing": float("nan"),
    }))
    assert points["val/ap"] == (180, 0.73)
    assert points["train/loss"] == (180, 0.2)
    assert points["val/loss"] == (180, 0.4)
    assert points["train/grad_norm"] == (180, 0.8)
    assert "extra/note" not in points
    assert "extra/flag" not in points
    assert "extra/missing" not in points


def test_result_weights_use_metapath_names():
    points = dict((tag, (step, value)) for tag, step, value in log.scalars_from_result({
        "best_epoch": 180,
        "epochs_trained": 380,
        "train_seconds": 12.5,
        "weight_diagnostics": {"mean_by_path": [0.2, 0.1], "std_by_path": [0.01], "mean_entropy": 2.1},
    }))
    assert points["weights/mean/gg"] == (180, 0.2)
    assert points["weights/mean/ga-ag"] == (180, 0.1)
    assert points["weights/std/gg"] == (180, 0.01)
    assert points["weights/entropy"] == (180, 2.1)
    assert points["time/train_seconds"] == (380, 12.5)


def _records(payload):
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        length = struct.unpack_from("<Q", view, offset)[0]
        offset += 12
        body = bytes(view[offset:offset + length])
        offset += length + 4
        yield body


def test_event_file_roundtrip_contains_file_version_and_scalar(tmp_path):
    path = log.write_events(tmp_path, [("val/ap", 180, 0.737788)])
    bodies = list(_records(path.read_bytes()))
    assert b"brain.Event:2" in bodies[0]
    assert b"val/ap" in bodies[1]
    assert struct.pack("<f", 0.737788) in bodies[1]
    assert struct.pack("<d", 180.0) not in bodies[1]


def test_live_queue_is_not_read(tmp_path):
    results = tmp_path / "results"
    live = results / "running_campaign" / "E1" / "seed42" / "fold0"
    done = results / "finished_campaign" / "E4" / "seed42" / "fold4"
    live.mkdir(parents=True)
    done.mkdir(parents=True)
    row = json.dumps({"epoch": 10, "ap": 0.5}) + "\n"
    (live / "history.jsonl").write_text(row, encoding="utf-8")
    (done / "history.jsonl").write_text(row, encoding="utf-8")
    (results / "running_campaign" / "queue.pid").write_text(str(log.os.getpid()), encoding="utf-8")
    chosen, skipped = log.runs_to_convert(results, results / "tensorboard")
    assert chosen == [done / "history.jsonl"]
    assert results.resolve() / "running_campaign" in skipped
    assert log.pid_alive(log.os.getpid())
    assert not log.pid_alive(2**31 - 1)
