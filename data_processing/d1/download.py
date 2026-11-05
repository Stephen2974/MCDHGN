"""Download pinned DataHub files, resolving Git LFS pointers.

Signed batch URLs are used only in memory. Logs keep the commit path, OID and size.
"""
import fnmatch
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .spec import COMMIT, MAX_LFS_BYTES, RAW_ROOT, STUDY_FILES, UNUSED_COMPANIONS, unique_studies

POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
USER_AGENT = "MCDHGN-d1-cnv/1.0"


class DownloadError(RuntimeError):
    pass


def parse_lfsconfig(text):
    endpoint = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("url") and "=" in stripped:
            endpoint = stripped.split("=", 1)[1].strip()
    if not endpoint or not endpoint.startswith("https://"):
        raise DownloadError("Pinned .lfsconfig has no https LFS endpoint")
    return endpoint


def lfs_patterns(text):
    patterns = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if "filter=lfs" in parts:
            patterns.append(parts[0])
    if not patterns:
        raise DownloadError("Pinned .gitattributes has no LFS patterns")
    return tuple(patterns)


def path_is_lfs(relpath, patterns):
    name = relpath.split("/")[-1]
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(relpath, pattern) for pattern in patterns)


def parse_pointer(data):
    """Return (oid, size) or None when the bytes are not an LFS pointer."""
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
        raise DownloadError(f"LFS object size {size} is outside the allowed range")
    return oid, size


def ensure_table_bytes(data, label="table"):
    if data.startswith(POINTER_PREFIX) or parse_pointer(data) is not None:
        raise DownloadError(f"Refusing to read an LFS pointer as {label}")
    return data


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(url, data=None, headers=None, timeout=60):
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, data=data, headers=merged)
    return urllib.request.urlopen(req, timeout=timeout)


def _read_bytes(url, timeout=60, retries=4):
    last = None
    for attempt in range(retries):
        try:
            with _request(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise DownloadError("Raw download failed: HTTP " + str(exc.code)) from None
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        if attempt + 1 == retries:
            break
        time.sleep(2 ** attempt)
    raise DownloadError("Raw download failed: " + type(last).__name__) from None


def _stream_to(url, dest, expected_size, timeout=180, retries=4):
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
                    digest.update(chunk)
                    handle.write(chunk)
            if size != expected_size:
                raise DownloadError(f"Downloaded size {size} does not match pointer size {expected_size}")
            temp.replace(dest)
            return digest.hexdigest(), size
        except DownloadError:
            temp.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            temp.unlink(missing_ok=True)
            last = exc
            if attempt + 1 == retries:
                break
            time.sleep(2 ** attempt)
    raise DownloadError("LFS object transfer failed: " + type(last).__name__) from None


def _batch_href(endpoint, oid, size, timeout=60, retries=4):
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
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 == retries:
                break
            time.sleep(2 ** attempt)
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
    """Download one repository path from the pinned commit into dest.

    LFS-tracked paths must arrive as pointers and are replaced by the object
    whose sha256 equals the pointer OID. A pointer is never left as the file.
    """
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


def inspect_unused_companion(study, name):
    relpath = f"public/{study}/{name}"
    try:
        raw = _read_bytes(RAW_ROOT + "/" + relpath)
    except DownloadError:
        return {"path": relpath, "present": False, "used": False}
    try:
        pointer = parse_pointer(raw) if raw.startswith(POINTER_PREFIX) else None
    except DownloadError as exc:
        return {"path": relpath, "present": True, "used": False, "note": redact(str(exc))}
    record = {"path": relpath, "present": True, "used": False, "reason": "not_the_discrete_gistic_profile"}
    if pointer:
        record.update(lfs=True, lfs_oid=pointer[0], lfs_size=pointer[1])
    else:
        record.update(lfs=False, sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
    return record


def download_studies(source_root, downloaded_at, progress=None):
    config = fetch_pinned_config()
    files = []
    unused = []
    for study in unique_studies():
        if progress:
            progress(f"download {study}")
        unused.extend(inspect_unused_companion(study, name) for name in UNUSED_COMPANIONS)
        for name in STUDY_FILES:
            relpath = f"public/{study}/{name}"
            dest = Path(source_root) / study / name
            receipt = download_relpath(relpath, dest, config, downloaded_at)
            files.append(dict(receipt, study=study, filename=name))
            if progress:
                progress(f"  {name} bytes={receipt['size']} reused={receipt.get('reused')}")
    return config, files, unused


_URL_RE = re.compile(r"https?://\S+")


def redact(text):
    def replace(match):
        value = match.group(0)
        base = value.split("?", 1)[0]
        return base + "?REDACTED" if "?" in value else base
    return _URL_RE.sub(replace, text)
