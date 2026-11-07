"""Download pinned DataHub mutation sources into d2_mutation_v1/sources only.

Reuses D1 LFS helpers for pointer parsing and batch URLs, but allows larger
mutation tables (UCEC ~886 MB). Never writes into preprocess_data/.../mut_data.
"""
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

_RETRYABLE = (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout, OSError)

from data_processing.d1.download import (
    POINTER_PREFIX,
    DownloadError,
    ensure_table_bytes,
    lfs_patterns,
    parse_lfsconfig,
    path_is_lfs,
    redact,
    sha256_file,
)

from .spec import COMMIT, MAX_LFS_BYTES, RAW_ROOT, STUDY_FILES, unique_studies

USER_AGENT = "MCDHGN-d2-mutation/1.0"


def parse_pointer(data):
    """Return (oid, size) or None. Uses D2 MAX_LFS_BYTES (mutation tables exceed D1's 250MB)."""
    if not data.startswith(POINTER_PREFIX):
        return None
    if len(data) > 1024:
        raise DownloadError("Oversized content starts like an LFS pointer")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DownloadError("LFS pointer is not ASCII") from exc
    oid = size = None
    for line in text.splitlines():
        if line.startswith("version "):
            continue
        if line.startswith("oid sha256:"):
            oid = line[len("oid sha256:"):].strip()
        elif line.startswith("size "):
            try:
                size = int(line.split()[1])
            except (IndexError, ValueError) as exc:
                raise DownloadError("LFS pointer size is not an integer") from exc
        elif line:
            raise DownloadError("Unexpected LFS pointer field")
    if not oid or size is None:
        raise DownloadError("Incomplete LFS pointer")
    if len(oid) != 64 or any(char not in "0123456789abcdef" for char in oid):
        raise DownloadError("LFS pointer OID is not a sha256 hex digest")
    if size < 0 or size > MAX_LFS_BYTES:
        raise DownloadError(f"LFS object size {size} is outside the D2 allowed range")
    return oid, size


def _request(url, data=None, headers=None, timeout=60):
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, data=data, headers=merged)
    return urllib.request.urlopen(req, timeout=timeout)


def _read_bytes(url, timeout=180, retries=8):
    last = None
    for attempt in range(retries):
        try:
            with _request(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500 and exc.code != 429:
                raise DownloadError("Raw download failed: HTTP " + str(exc.code)) from None
            last = exc
        except _RETRYABLE as exc:
            last = exc
        if attempt + 1 == retries:
            break
        time.sleep(min(60, 2 ** attempt))
    raise DownloadError("Raw download failed: " + type(last).__name__) from None


def _stream_to(url, dest, expected_size, timeout=900, retries=8):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".partial")
    last = None
    for attempt in range(retries):
        digest = hashlib.sha256()
        size = 0
        try:
            with _request(url, timeout=timeout) as response, temp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > expected_size:
                        raise DownloadError("Downloaded object exceeds the pointer size")
                    if size > MAX_LFS_BYTES:
                        raise DownloadError(f"Object size {size} exceeds D2 MAX_LFS_BYTES")
                    digest.update(chunk)
                    handle.write(chunk)
            if size != expected_size:
                raise DownloadError(f"Downloaded size {size} does not match pointer size {expected_size}")
            temp.replace(dest)
            return digest.hexdigest(), size
        except DownloadError:
            temp.unlink(missing_ok=True)
            raise
        except _RETRYABLE as exc:
            temp.unlink(missing_ok=True)
            last = exc
            if attempt + 1 == retries:
                break
            time.sleep(min(60, 2 ** attempt))
    raise DownloadError("LFS object transfer failed: " + type(last).__name__) from None


def _batch_href(endpoint, oid, size, timeout=180, retries=8):
    if size > MAX_LFS_BYTES:
        raise DownloadError(f"LFS object size {size} exceeds D2 MAX_LFS_BYTES")
    url = endpoint.rstrip("/") + "/objects/batch"
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}],
    }).encode("utf-8")
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
    }
    last = None
    for attempt in range(retries):
        try:
            with _request(url, data=payload, headers=headers, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            obj = body["objects"][0]
            if obj.get("oid") != oid or int(obj.get("size")) != size:
                raise DownloadError("LFS batch response does not match the requested object")
            if obj.get("error"):
                raise DownloadError("LFS batch returned an object error")
            href = obj["actions"]["download"]["href"]
            if not href.startswith("https://"):
                raise DownloadError("LFS download action is not https")
            return href
        except DownloadError:
            raise
        except (_RETRYABLE, KeyError, ValueError, json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 == retries:
                break
            time.sleep(min(60, 2 ** attempt))
    raise DownloadError("LFS batch request failed: " + type(last).__name__) from None


def _receipt_path(dest):
    return Path(str(dest) + ".receipt.json")


def _load_receipt(dest):
    path = _receipt_path(dest)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_receipt(dest, receipt):
    _receipt_path(dest).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_pinned_config():
    lfsconfig = _read_bytes(RAW_ROOT + "/.lfsconfig").decode("utf-8")
    attributes = _read_bytes(RAW_ROOT + "/.gitattributes").decode("utf-8")
    readme = _read_bytes(RAW_ROOT + "/README.md").decode("utf-8")
    endpoint = parse_lfsconfig(lfsconfig)
    patterns = lfs_patterns(attributes)
    return {
        "commit": COMMIT,
        "lfsconfig_text": lfsconfig,
        "lfs_endpoint": endpoint,
        "gitattributes_text": attributes,
        "lfs_patterns": patterns,
        "readme_sha256": hashlib.sha256(readme.encode("utf-8")).hexdigest(),
        "readme_excerpt": readme[:400],
    }


def download_relpath(relpath, dest, config, downloaded_at):
    dest = Path(dest)
    raw = _read_bytes(RAW_ROOT + "/" + relpath)
    tracked = path_is_lfs(relpath, config["lfs_patterns"])
    pointer = parse_pointer(raw)
    if tracked and pointer is None:
        raise DownloadError(f"{relpath} is LFS-tracked but the pinned blob is not a pointer")
    if pointer is not None and not tracked:
        raise DownloadError(f"{relpath} returned an LFS pointer without an LFS attribute")
    if pointer is None:
        ensure_table_bytes(raw, relpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(raw).hexdigest()
        reused = dest.is_file() and dest.stat().st_size == len(raw) and sha256_file(dest) == digest
        if not reused:
            dest.write_bytes(raw)
        receipt = {
            "commit": COMMIT,
            "source_path": relpath,
            "sha256": digest,
            "size": len(raw),
            "lfs": False,
            "lfs_oid": None,
            "lfs_size": None,
            "downloaded_at": downloaded_at,
            "reused": reused,
        }
        _write_receipt(dest, receipt)
        return receipt

    oid, size = pointer
    if size > MAX_LFS_BYTES:
        raise DownloadError(f"LFS object size {size} exceeds D2 MAX_LFS_BYTES")
    if dest.is_file() and dest.stat().st_size == size and sha256_file(dest) == oid:
        receipt = {
            "commit": COMMIT,
            "source_path": relpath,
            "sha256": oid,
            "size": size,
            "lfs": True,
            "lfs_oid": oid,
            "lfs_size": size,
            "downloaded_at": (_load_receipt(dest) or {}).get("downloaded_at", downloaded_at),
            "downloaded_at_verified": downloaded_at,
            "reused": True,
        }
        _write_receipt(dest, receipt)
        return receipt
    href = _batch_href(config["lfs_endpoint"], oid, size)
    try:
        digest, actual = _stream_to(href, dest, size)
    finally:
        del href
    if digest != oid or actual != size:
        dest.unlink(missing_ok=True)
        raise DownloadError(f"OID mismatch for {relpath}")
    receipt = {
        "commit": COMMIT,
        "source_path": relpath,
        "sha256": digest,
        "size": actual,
        "lfs": True,
        "lfs_oid": oid,
        "lfs_size": size,
        "downloaded_at": downloaded_at,
        "reused": False,
    }
    _write_receipt(dest, receipt)
    return receipt


def download_studies(source_root, downloaded_at, progress=None, studies=None):
    """Download mutation sources one study at a time (no parallel transfers)."""
    config = fetch_pinned_config()
    files = []
    for study in (studies or unique_studies()):
        if progress:
            progress(f"download {study}")
        for name in STUDY_FILES:
            relpath = f"public/{study}/{name}"
            dest = Path(source_root) / study / name
            receipt = download_relpath(relpath, dest, config, downloaded_at)
            files.append(dict(receipt, study=study, filename=name))
            if progress:
                progress(f"  {name} bytes={receipt['size']} reused={receipt.get('reused')}")
    return config, files


__all__ = ["DownloadError", "download_studies", "download_relpath", "fetch_pinned_config", "redact", "sha256_file"]
