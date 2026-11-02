"""Small, dependency-free utilities for the P0 audit. Never unpickle external data."""
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'data_versions' / 'p0_v1'
VALIDATION_CODE = ('data_processing/p0/validate_graph.py', 'data_processing/p0/audit.py',
                   'data_processing/p0/common.py', 'run_ablation.py')
RELEASE_CODE = ('data_processing/__init__.py', 'data_processing/p0/__init__.py',
                'data_processing/p0/common.py', 'data_processing/p0/prepare.py',
                'data_processing/p0/audit.py', 'data_processing/p0/validate_graph.py',
                'data_processing/p0/release.py', 'run_ablation.py', 'model.py', 'fusion.py',
                'ablation_config.py', 'summarize_ablations.py', 'preprocess/generate_network.ipynb',
                'tests/test_p0.py', 'data_processing/p0/README.md', 'readme.md')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def log(step, status, details, out=DEFAULT_OUT):
    event = dict(time=now(), step=step, status=status, details=details)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    with (ROOT / 'data_processing/p0/EXECUTION_LOG.md').open('a', encoding='utf-8') as f:
        f.write('\n## ' + event['time'] + ' · ' + step + ' · ' + status + '\n\n')
        f.write(details + '\n')


def verify_frozen(root, out, include_code=False):
    manifest = json.loads((Path(out) / 'baseline_manifest.json').read_text(encoding='utf-8'))
    errors = []
    for row in manifest['files']:
        if row['kind'] == 'code' and not include_code:
            continue
        path = Path(root) / row['path']
        if not path.is_file() or sha256(path) != row['sha256']:
            errors.append(row['path'])
    if errors:
        raise ValueError('Frozen inputs changed or missing: ' + ', '.join(errors))
    return manifest
