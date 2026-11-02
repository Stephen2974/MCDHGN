"""Seal/verify the P0 audit bundle. A seal is integrity metadata, not scientific approval."""
import argparse
import json
from pathlib import Path

from .common import ROOT, DEFAULT_OUT, RELEASE_CODE, log, now, sha256, verify_frozen, write_json


def seal(root, out):
    target=out/'manifest.json'
    if target.exists():
        raise FileExistsError('Audit bundle already sealed. Use a new version directory for new results.')
    baseline=verify_frozen(root,out)
    quality=json.loads((out/'quality_report.json').read_text(encoding='utf-8'))
    if quality['splits']['status']!='passed' or quality['graph_validation']['status']!='passed':
        raise ValueError('Local graph and split checks must pass before sealing')
    if not quality['features']['float32_source_exact_match']:
        raise ValueError('Source features do not match')
    artifacts=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in {'manifest.json','events.jsonl'} and not p.name.endswith('.tmp'):
            artifacts.append(dict(path=p.relative_to(out).as_posix(),size_bytes=p.stat().st_size,sha256=sha256(p)))
    code_names=RELEASE_CODE
    code={name:sha256(root/name) for name in code_names}
    manifest=dict(schema_version=1,created_at=now(),version='p0_v1',
                  status='local_audit_sealed_strict_training_and_temporal_evaluation_not_released',
                  baseline_manifest_sha256=sha256(out/'baseline_manifest.json'),
                  preserved_data_files=sum(r['kind']=='data' for r in baseline['files']),
                  artifacts=artifacts,code_sha256=code,mutable_logs=['events.jsonl','data_processing/p0/EXECUTION_LOG.md'],
                  user_decisions=dict(authoritative_data='local',server_role='validation_and_training',
                                      strict_labels='candidate_lists_only',historical_inputs='unavailable',old_test_usage='unknown'),
                  incomplete=['server_environment_revalidation','server_D0_training',
                              'strict_negative_proxy_protocol','first_evidence_dates','independent_temporal_validation'],
                  changes='Explicit node ordering and loader integrity checks only; original data and old splits unchanged.')
    write_json(target,manifest)
    log('P0 本地成果封存','sealed_with_explicit_pending_items',f'封存 {len(artifacts)} 个产物及 {len(code)} 个代码/说明文件；严格版不发布训练划分，时间外验证未建立，服务器训练未执行。',out)
    print(json.dumps(dict(status=manifest['status'],artifacts=len(artifacts)),ensure_ascii=False))


def verify(root,out):
    manifest_path=out/'layout_manifest.json'
    if not manifest_path.exists():
        manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest_path.name=='layout_manifest.json':
        if sha256(out/'manifest.json')!=manifest['previous_manifest_sha256']:
            raise ValueError('Original pre-migration manifest changed')
        original=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        current_artifacts={row['path']:row for row in manifest['artifacts']}
        if any(current_artifacts.get(row['path'])!=row for row in original['artifacts']):
            raise ValueError('Migration must preserve all previously sealed artifacts')
        if manifest['baseline_manifest_sha256']!=original['baseline_manifest_sha256']:
            raise ValueError('Migration changed the original baseline reference')
    if sha256(out/'baseline_manifest.json')!=manifest['baseline_manifest_sha256']:
        raise ValueError('Baseline manifest changed')
    verify_frozen(root,out)
    failures=[]
    for row in manifest['artifacts']:
        p=out/row['path']
        if not p.is_file() or sha256(p)!=row['sha256']:
            failures.append('artifact:'+row['path'])
    for name,digest in manifest['code_sha256'].items():
        p=root/name
        if not p.is_file() or sha256(p)!=digest:
            failures.append('code:'+name)
    if failures:
        raise ValueError('Audit bundle mismatch: '+', '.join(failures))
    log('P0 封存完整性验证','passed','原始数据、外部快照、审计产物与当前交付代码的 SHA256 全部一致。此结果不代表已完成服务器训练或严格标签发布。',out)
    print('PASS: original inputs, HGNC snapshot, audit artifacts, and delivery code match the sealed manifest.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['seal','verify'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=p.parse_args()
    try:
        (seal if args.action=='seal' else verify)(args.root.resolve(),args.out.resolve())
    except Exception as exc:
        log('P0 release '+args.action,'failed',f'{type(exc).__name__}: {exc}',args.out)
        raise


if __name__=='__main__':
    main()
