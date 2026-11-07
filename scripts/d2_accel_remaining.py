"""Multi-connection fetch for D2 studies the sequential downloader has not started.

Does not stop or write over the live sequential download. Large LFS objects are
pulled into a staging directory, checked against the pinned OID, then moved into
sources/<study> only when that directory does not exist yet.
Signed URLs are never written to the log.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data_processing.d2.download as d2dl  # noqa: E402
from data_processing.d2.download import (  # noqa: E402
    _batch_href,
    _write_receipt,
    download_relpath,
    fetch_pinned_config,
    parse_pointer,
)
from data_processing.d2.spec import COMMIT, RAW_ROOT, STUDY_FILES, unique_studies  # noqa: E402

MIRROR_ROOT = f"https://cdn.jsdelivr.net/gh/cBioPortal/datahub@{COMMIT}"


def read_via_mirror(url, timeout=60, retries=4):
    """raw.githubusercontent.com stalls on this network; jsDelivr serves the same pinned blobs."""
    if not url.startswith(RAW_ROOT):
        raise RuntimeError("unexpected raw URL")
    mirror = MIRROR_ROOT + url[len(RAW_ROOT):]
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(mirror, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError("mirror read failed: " + type(last).__name__) from last

SOURCE_ROOT = ROOT / "data_versions" / "d2_mutation_v1" / "sources"
STAGE_ROOT = ROOT / "data_versions" / "d2_mutation_v1" / "_accel_stage"
LOG_PATH = ROOT / "data_versions" / "d2_mutation_v1" / "accel.log"
CONNECTIONS = 4
USER_AGENT = "MCDHGN-d2-accel/1.0"
STAMP = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def official_busy(study: str) -> str | None:
    study_dir = SOURCE_ROOT / study
    final = study_dir / "data_mutations.txt"
    partial = study_dir / "data_mutations.txt.partial"
    if final.is_file():
        return "done"
    if partial.exists() or study_dir.exists():
        return "in_progress"
    return None


def ranges_for(size: int, connections: int):
    connections = max(1, min(connections, max(1, size // (8 * 1024 * 1024))))
    chunk = (size + connections - 1) // connections
    bounds = []
    start = 0
    while start < size:
        end = min(size - 1, start + chunk - 1)
        bounds.append((start, end))
        start = end + 1
    return bounds


def fetch_range(url: str, start: int, end: int, dest: Path) -> None:
    expected = end - start + 1
    if dest.is_file() and dest.stat().st_size == expected:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        status = getattr(response, "status", None) or response.getcode()
        if status != 206:
            raise RuntimeError(f"range request returned HTTP {status}")
        got = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            got += len(block)
            if got > expected:
                raise RuntimeError("range part exceeded requested length")
            handle.write(block)
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"range part size {got} != {expected}")
    tmp.replace(dest)


def stream_ranged(url: str, dest: Path, expected_size: int, connections: int) -> str:
    part_dir = dest.with_suffix(dest.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    bounds = ranges_for(expected_size, connections)
    log(f"  ranged connections={len(bounds)} bytes={expected_size}")

    def one(index_bounds):
        index, (start, end) = index_bounds
        part = part_dir / f"part_{index:02d}.bin"
        last = None
        for attempt in range(4):
            try:
                fetch_range(url, start, end, part)
                return
            except Exception as exc:  # noqa: BLE001 - retry transient network errors
                last = exc
                part.with_suffix(part.suffix + ".part").unlink(missing_ok=True)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"part {index} failed: {type(last).__name__}") from last

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
        futures = [pool.submit(one, item) for item in enumerate(bounds)]
        for future in as_completed(futures):
            future.result()
    digest = hashlib.sha256()
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with tmp.open("wb") as handle:
        for index, (start, end) in enumerate(bounds):
            part = part_dir / f"part_{index:02d}.bin"
            data = part.read_bytes()
            if len(data) != end - start + 1:
                raise RuntimeError(f"part {index} length changed")
            digest.update(data)
            handle.write(data)
    elapsed = max(time.perf_counter() - started, 0.001)
    log(f"  ranged done MB/s={tmp.stat().st_size / elapsed / (1024 * 1024):.3f}")
    tmp.replace(dest)
    for part in part_dir.glob("part_*.bin"):
        part.unlink(missing_ok=True)
    part_dir.rmdir()
    return digest.hexdigest()


def download_mutations(study: str, stage_dir: Path, config: dict) -> None:
    name = "data_mutations.txt"
    dest = stage_dir / name
    relpath = f"public/{study}/{name}"
    raw = read_via_mirror(RAW_ROOT + "/" + relpath)
    pointer = parse_pointer(raw)
    if pointer is None:
        download_relpath(relpath, dest, config, STAMP)
        return
    oid, size = pointer
    if dest.is_file() and dest.stat().st_size == size:
        if hashlib.sha256(dest.read_bytes()).hexdigest() == oid:
            log(f"  {name} already staged bytes={size}")
            return
        dest.unlink()
    href = _batch_href(config["lfs_endpoint"], oid, size)
    try:
        try:
            digest = stream_ranged(href, dest, size, CONNECTIONS)
        except Exception as exc:  # noqa: BLE001
            log(f"  ranged failed ({type(exc).__name__}); single connection fallback")
            dest.unlink(missing_ok=True)
            from data_processing.d2.download import _stream_to

            digest, actual = _stream_to(href, dest, size)
            if actual != size:
                raise RuntimeError(f"single stream size {actual} != {size}") from exc
    finally:
        del href
    if digest != oid or dest.stat().st_size != size:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"OID mismatch for {study}/{name}")
    _write_receipt(dest, {
        "commit": COMMIT,
        "source_path": relpath,
        "sha256": digest,
        "size": size,
        "lfs": True,
        "lfs_oid": oid,
        "lfs_size": size,
        "downloaded_at": STAMP,
        "reused": False,
        "accelerator": "d2_accel_remaining",
    })
    log(f"  {name} verified bytes={size}")


def stage_study(study: str, config: dict) -> Path:
    stage_dir = STAGE_ROOT / study
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name in STUDY_FILES:
        if name == "data_mutations.txt":
            continue
        relpath = f"public/{study}/{name}"
        dest = stage_dir / name
        if dest.is_file() and (Path(str(dest) + ".receipt.json")).is_file():
            continue
        download_relpath(relpath, dest, config, STAMP)
        log(f"  {name} ok")
    download_mutations(study, stage_dir, config)
    return stage_dir


def publish(study: str, stage_dir: Path) -> str:
    official = SOURCE_ROOT / study
    if official.exists():
        return "left_staged_official_exists"
    try:
        os.rename(stage_dir, official)
    except OSError as exc:
        log(f"  publish skipped: {type(exc).__name__}")
        return "left_staged_rename_failed"
    return "published"


def main() -> int:
    # Steam++ system proxy answers 127.0.0.1:443 and stalls new GitHub requests.
    # The already-running downloader is a direct CDN connection; match that.
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    d2dl._read_bytes = read_via_mirror
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log("proxy bypass enabled; raw blobs via jsDelivr")
    config = fetch_pinned_config()
    log("pinned config loaded")
    pending = []
    for study in reversed(unique_studies()):
        state = official_busy(study)
        if state:
            log(f"skip {study} ({state})")
            continue
        pending.append(study)
    log(f"accelerate {len(pending)} studies from the tail: {', '.join(pending)}")
    for study in pending:
        state = official_busy(study)
        if state:
            log(f"skip {study} ({state}) before start")
            continue
        log(f"start {study}")
        try:
            stage_dir = stage_study(study, config)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
            log(f"FAIL {study}: {type(exc).__name__}: {exc}")
            continue
        state = official_busy(study)
        if state:
            log(f"keep staged {study}; official became {state}")
            continue
        result = publish(study, stage_dir)
        log(f"{result} {study}")
    log("accelerator finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
