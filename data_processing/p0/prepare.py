"""Freeze local P0 inputs and download a pinned HGNC snapshot (stdlib only)."""
import argparse
import csv
import io
import json
import platform
import subprocess
import urllib.request
from pathlib import Path

from .common import ROOT, DEFAULT_OUT, log, now, sha256, write_json, write_tsv


def freeze(root, out):
    target = out / 'baseline_manifest.json'
    if target.exists():
        raise FileExistsError('Baseline is immutable; choose a new --out directory')
    data = {root / 'data/biological_features.csv'}
    for directory in ['data/step1', 'data/labels', 'data/network/hetero', 'Intermediate/blocks', 'Intermediate/label_set', 'data/msigdb']:
        data.update(p for p in (root / directory).rglob('*') if p.is_file() and p.name != '.DS_Store' and '.ipynb_checkpoints' not in p.parts)
    code = set(root.glob('*.py')) | set(root.glob('*.ipynb')) | set((root / 'preprocess').glob('*.ipynb'))
    code |= set((root / 'data_processing').rglob('*.py'))
    code.add(root / 'data_processing/p0/README.md')
    code |= {root / 'DATA_EXPANSION.md', root / 'EXPERIMENTS.md', root / '.gitignore'}
    rows = []
    for p in sorted(data | code):
        if not p.is_file():
            raise FileNotFoundError(p)
        rows.append(dict(path=p.relative_to(root).as_posix(), kind='data' if p in data else 'code',
                         size_bytes=p.stat().st_size, sha256=sha256(p)))
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    manifest = dict(schema_version=1, created_at=now(), authority='local_repository',
                    node_order='data/step1/gene_toids.pkl', test_usage_history='unknown',
                    git_head=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
                    python=platform.python_version(), files=rows,
                    limitations=['File hashes freeze identity, not a backup of binary content.',
                                 'Historical source download dates and licenses must not be inferred from file modification times.'])
    write_json(target, manifest)
    write_tsv(out / 'baseline_files.tsv', rows, ['path', 'kind', 'size_bytes', 'sha256'])
    log('01 版本冻结', 'completed', f'已固定 {len(rows)} 个文件的大小与 SHA256，保存 Git 状态和实际代码哈希。原始二进制保留在本地，清单不是独立备份。', out)
    print(json.dumps(dict(files=len(rows), manifest=str(target)), ensure_ascii=False))


def download_hgnc(out, url, release):
    directory = out / 'sources' / 'hgnc'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'hgnc_complete_set.tsv'
    if target.exists():
        raise FileExistsError('Pinned source already exists; do not silently replace it')
    req = urllib.request.Request(url, headers={'User-Agent': 'MCDHGN-P0/1.0'})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
        headers = dict(response.headers.items())
        final_url = response.geturl()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t')
    required = {'hgnc_id','symbol','status','alias_symbol','prev_symbol','ensembl_gene_id','entrez_id','uniprot_ids'}
    if not required <= set(reader.fieldnames or []):
        raise ValueError('Download is not the required HGNC TSV schema')
    count = sum(1 for _ in reader)
    if count < 10000:
        raise ValueError('Unexpectedly small HGNC complete set')
    target.write_bytes(data)
    metadata = dict(source='HGNC', release=release, url=url, final_url=final_url,
                    downloaded_at=now(), sha256=sha256(target), bytes=len(data), rows=count,
                    response_headers=headers, usage_terms_url='https://www.genenames.org/about/',
                    usage_terms_summary='HGNC permits use and requests attribution; checked 2026-10-23.',
                    version_policy='The local file plus SHA256 is immutable; release names refer to the supplied source URL, not evidence dates.')
    write_json(directory / 'source_manifest.json', metadata)
    log('02 HGNC 下载', 'completed', f'下载并固定 HGNC {release}，{count} 条记录。来源 {url}；SHA256 {metadata["sha256"]}。', out)
    print(json.dumps(metadata, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['freeze', 'download-hgnc'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--url')
    p.add_argument('--release')
    args = p.parse_args()
    try:
        if args.action == 'freeze':
            freeze(args.root.resolve(), args.out.resolve())
        else:
            if not args.url or not args.release:
                p.error('--url and --release are required')
            download_hgnc(args.out.resolve(), args.url, args.release)
    except Exception as exc:
        log(args.action, 'failed', f'{type(exc).__name__}: {exc}', args.out)
        raise


if __name__ == '__main__':
    main()
