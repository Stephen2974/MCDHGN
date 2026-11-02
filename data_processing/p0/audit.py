"""Reproducible P0 identity, feature, evidence and split audit; no model training.

Only load pickle/PT files from this trusted repository. External sources are TSV.
"""
import argparse
from collections import Counter, defaultdict
import csv
import inspect
import json
import pickle
import re
from pathlib import Path

from .common import ROOT, DEFAULT_OUT, VALIDATION_CODE, log, now, read_tsv, sha256, verify_frozen, write_json, write_tsv


def load_pickle(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def load_tensor(path):
    import torch
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = True
    return torch.load(path, **options)


def ordered_genes(mapping):
    if any(type(v) is not int for v in mapping.values()):
        raise ValueError('Node IDs must be Python integers')
    genes = sorted(mapping, key=mapping.get)
    if [mapping[g] for g in genes] != list(range(len(genes))):
        raise ValueError('Node IDs are not unique and contiguous from zero')
    return genes


def aligned_features(mapping, features):
    import torch
    genes = ordered_genes(mapping)
    if set(features) != set(mapping):
        raise ValueError('Feature and node dictionary keys differ')
    x = torch.stack([features[g].detach().cpu() for g in genes])
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError('Invalid feature matrix')
    return x


class HgncIndex:
    def __init__(self, records):
        self.records = {}
        self.index = {key: defaultdict(set) for key in ('symbol','prev_symbol','alias_symbol','hgnc_id','entrez_id','ensembl_gene_id','uniprot_ids')}
        for row in records:
            hid = row['hgnc_id']
            if hid in self.records:
                raise ValueError('Duplicate HGNC ID: ' + hid)
            self.records[hid] = row
            if row['status'] != 'Approved':
                continue
            for key, index in self.index.items():
                for value in row.get(key, '').split('|'):
                    if value:
                        index[value].add(hid)

    def resolve(self, source_id, namespace='symbol'):
        value = source_id.strip()
        if namespace == 'ensembl_gene_id':
            value = re.sub(r'\.\d+$', '', value)
        if namespace == 'symbol':
            approved = self.index['symbol'].get(value, set())
            if approved:
                return set(approved), 'approved_symbol'
            previous = self.index['prev_symbol'].get(value, set())
            aliases = self.index['alias_symbol'].get(value, set())
            return set(previous | aliases), 'previous_symbol' if previous and not aliases else 'alias_or_previous'
        return set(self.index[namespace].get(value, set())), namespace


MAP_FIELDS = ['node_id','source_id','hgnc_id','symbol','alias_symbol','prev_symbol','ensembl_gene_id','entrez_id','uniprot_ids','status','mapping_status','mapping_method','candidate_hgnc_ids','ambiguity_reason']


def build_gene_map(mapping, index):
    rows = []
    for i, gene in enumerate(ordered_genes(mapping)):
        candidates, method = index.resolve(gene)
        row = dict.fromkeys(MAP_FIELDS, '')
        row.update(node_id=i, source_id=gene, mapping_method=method,
                   candidate_hgnc_ids='|'.join(sorted(candidates)))
        if len(candidates) == 1:
            record = index.records[next(iter(candidates))]
            row.update({k: record.get(k, '') for k in MAP_FIELDS if k in record})
            row['mapping_status'] = 'resolved'
        else:
            row['mapping_status'] = 'ambiguous' if candidates else 'unmapped'
            row['ambiguity_reason'] = 'multiple_hgnc_candidates' if candidates else 'no_approved_record'
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        if row['hgnc_id']:
            groups[row['hgnc_id']].append(row)
    for group in groups.values():
        if len(group) > 1:
            for row in group:
                row['mapping_status'] = 'duplicate_identity'
                row['ambiguity_reason'] = 'multiple_existing_nodes_share_hgnc_id'
    return rows


def mapping_audit(root, out):
    mapping = load_pickle(root / 'data/step1/gene_toids.pkl')
    source = out / 'sources/hgnc/hgnc_complete_set.tsv'
    source_meta = json.loads((source.parent / 'source_manifest.json').read_text(encoding='utf-8'))
    if sha256(source) != source_meta['sha256']:
        raise ValueError('HGNC snapshot hash mismatch')
    index = HgncIndex(read_tsv(source))
    rows = build_gene_map(mapping, index)
    write_tsv(out / 'gene_map.tsv', rows, MAP_FIELDS)
    write_tsv(out / 'unmapped.tsv', [r for r in rows if r['mapping_status'] == 'unmapped'], MAP_FIELDS)
    write_tsv(out / 'mapping_conflicts.tsv', [r for r in rows if r['mapping_status'] in {'ambiguous','duplicate_identity'}], MAP_FIELDS)
    report = dict(nodes=len(rows), node_order_unchanged=True, statuses=dict(Counter(r['mapping_status'] for r in rows)),
                  methods=dict(Counter(r['mapping_method'] for r in rows)), hgnc_source_sha256=sha256(source),
                  policy='Exact approved symbol first; otherwise union of previous/alias candidates. Ambiguity and duplicate identities never auto-resolved.')
    write_json(out / 'mapping_report.json', report)
    log('02 身份映射', 'completed', json.dumps(report, ensure_ascii=False), out)
    return mapping, index, rows


def feature_audit(root, out, mapping):
    import torch
    features = load_pickle(root / 'data/step1/gene_feature_bio.pkl')
    x = aligned_features(mapping, features)
    raw = read_tsv(root / 'data/biological_features.csv')
    columns = list(raw[0])
    if columns[0] != 'Hugosymbol':
        raise ValueError('Unexpected feature gene identifier column')
    by_gene = {r['Hugosymbol']: r for r in raw}
    if len(by_gene) != len(raw):
        raise ValueError('Duplicate source feature symbols')
    missing = set(mapping) - set(by_gene)
    if missing:
        raise ValueError('Missing source features: ' + str(sorted(missing)))
    expected = torch.tensor([[float(by_gene[g][c]) for c in columns[1:]] for g in ordered_genes(mapping)], dtype=x.dtype)
    if expected.shape != x.shape or x.shape[1] != 48 or not torch.isfinite(expected).all():
        raise ValueError('Invalid source feature shape/values')
    mismatch = torch.any(x != expected, dim=1).nonzero().flatten().tolist()
    aliases = {}
    for rel, reverse in [('data/network/hetero/name_to_ids/Gene.pkl', False), ('data/network/hetero/ids_to_name/Gene.pkl', True)]:
        saved = load_pickle(root / rel)
        aliases[rel] = saved == ({v:k for k,v in mapping.items()} if reverse else mapping)
    report = dict(source_rows=len(raw), mapped_rows=len(mapping), columns=columns[1:], shape=list(x.shape),
                  all_finite=True, dictionary_order_matches_node_id=list(features)==ordered_genes(mapping),
                  float32_source_exact_match=not mismatch, mismatch_node_ids=mismatch,
                  maximum_absolute_difference=float((x-expected).abs().max()), mapping_copies_match=aliases,
                  dgl_graph_status='pending_server_validation', blocks_status='pending_server_validation')
    write_json(out / 'feature_alignment_report.json', report)
    schema = [dict(position=i, column_name=c, modality=c.split(':')[0].strip(), cancer_type=c.split(':')[-1].strip(),
                   source='data/biological_features.csv', source_release='historical_unknown',
                   missing_rule='existing_values_preserved', unit='historical_preprocessing_not_reconstructed') for i,c in enumerate(columns[1:])]
    write_tsv(out / 'feature_schema.tsv', schema, list(schema[0]))
    log('03 本地特征审计', 'completed' if not mismatch and all(aliases.values()) else 'failed', json.dumps({k:v for k,v in report.items() if k!='columns'}, ensure_ascii=False), out)
    return report


class NodeResolver:
    def __init__(self, index, rows):
        self.index = index
        self.nodes = defaultdict(list)
        for r in rows:
            if r['hgnc_id']:
                self.nodes[r['hgnc_id']].append(int(r['node_id']))

    def resolve(self, symbol='', entrez='', hgnc=''):
        evidence = []
        for value, namespace in [(symbol,'symbol'),(entrez,'entrez_id'),(hgnc,'hgnc_id')]:
            if value:
                found, _ = self.index.resolve(value, namespace)
                if found:
                    evidence.append(found)
        if not evidence:
            return '', '', 'unmapped'
        common = set.intersection(*evidence)
        if len(common) != 1:
            return '', '|'.join(sorted(set.union(*evidence))), 'identifier_conflict' if not common else 'ambiguous'
        hid = next(iter(common))
        nodes = self.nodes.get(hid, [])
        if len(nodes) != 1:
            return '', hid, 'outside_nodes' if not nodes else 'duplicate_identity'
        return nodes[0], hid, 'resolved'


EVIDENCE_FIELDS = ['node_id','hgnc_id','source_symbol','source_entrez','source','source_release','source_file','source_row','source_sha256','mapping_status','label_status','driver_class','cancer_type','PMID','evidence_date','evidence_date_status','role','evidence_level','conflict_flag','training_eligible','usage_status']


def label_audit(root, out, mapping, index, gene_rows):
    resolver = NodeResolver(index, gene_rows)
    positives = set(load_pickle(root / 'data/labels/now_pos427.pkl'))
    negatives = set(load_pickle(root / 'data/labels/now_neg427.pkl'))
    if not (positives | negatives) <= set(mapping.values()):
        raise ValueError('Legacy labels outside node universe')
    canonical = set((root / 'data/labels/NCG7.0/canonical_drivers.txt').read_text().splitlines())
    candidate = set((root / 'data/labels/NCG7.0/candidate_drivers.txt').read_text().splitlines())
    evidence, conflicts = [], []
    hashes = {}
    def add(rel, line, symbol, entrez='', hgnc='', **values):
        node, hid, status = resolver.resolve(symbol, entrez, hgnc)
        if rel not in hashes:
            hashes[rel] = sha256(root / rel)
        row = dict.fromkeys(EVIDENCE_FIELDS, '')
        row.update(node_id=node, hgnc_id=hid, source_symbol=symbol, source_entrez=entrez, source_file=rel,
                   source_row=line, source_sha256=hashes[rel], mapping_status=status,
                   source_release='unknown_local_snapshot', label_status='unlabeled', driver_class='unspecified',
                   evidence_date_status='unknown', training_eligible='false', usage_status='needs_source_terms_review',
                   conflict_flag='false')
        row.update(values)
        if status in {'identifier_conflict','ambiguous','duplicate_identity'}:
            row['conflict_flag'] = 'true'
        if status == 'resolved' and node in negatives:
            row['conflict_flag'] = 'true'
            conflicts.append(dict(node_id=node, hgnc_id=hid, source=values.get('source',''), source_row=line,
                                  reason='legacy_negative_has_driver_related_evidence', evidence_level=values.get('evidence_level','')))
        evidence.append(row)
    rel = 'data/labels/NCG7.0/NCG_cancerdrivers.tsv'
    for line, r in enumerate(read_tsv(root / rel), 2):
        cls = 'canonical' if r['type'] == 'Canonical Cancer Driver' else 'candidate'
        role = '|'.join(x for x,key in [('oncogene','NCG_oncogene'),('TSG','NCG_tsg')] if r.get(key)=='1')
        add(rel,line,r['symbol'],r['entrez'],source='NCG',source_release='7.0 (local directory; upstream authenticity unverified)',
            label_status='positive' if cls=='canonical' else 'unlabeled', driver_class=cls,
            cancer_type=r.get('cancer_type',''),PMID=r.get('pubmed_id',''),role=role,evidence_level=r['type'])
    for name, symbols in [('canonical',canonical),('candidate',candidate)]:
        rel = f'data/labels/NCG7.0/{name}_drivers.txt'
        for line, symbol in enumerate((root/rel).read_text().splitlines(),1):
            add(rel,line,symbol,source='NCG_list',source_release='7.0 (local directory)',
                label_status='positive' if name=='canonical' else 'unlabeled', driver_class=name, evidence_level='local_'+name+'_list')
    rel = 'data/labels/new_add_gene/cancermine_sentences.tsv'
    with (root/rel).open(encoding='utf-8-sig', newline='') as f:
        for line,r in enumerate(csv.DictReader(f,delimiter='\t'),2):
            # A literature mention is supporting context, never automatic driver confirmation.
            parts = [r.get(k,'').strip() for k in ['year','month','day']]
            date = '-'.join(v.zfill(2) if i else v for i,v in enumerate(parts) if v)
            add(rel,line,r.get('gene_normalized',''),r.get('gene_entrez_id',''),r.get('gene_hugo_id',''),
                source='CancerMine',PMID=r.get('pmid',''),cancer_type=r.get('cancer_normalized',''),
                role=r.get('role',''),evidence_level='text_mined_mention_not_confirmation',
                evidence_date=date,evidence_date_status='source_publication_date_not_first_driver_evidence')
    rel = 'data/labels/new_add_gene/ongene_human.txt'
    for line,r in enumerate(read_tsv(root/rel),2):
        add(rel,line,r['OncogeneName'],r['OncogeneID'],source='ONGene',role='oncogene',evidence_level='catalog_membership')
    rel = 'data/labels/OncoKB/cancerGeneList.tsv'
    for line,r in enumerate(read_tsv(root/rel),2):
        add(rel,line,r['Hugo Symbol'],r['Entrez Gene ID'],source='OncoKB',evidence_level='catalog_membership_audit_only',
            usage_status='excluded_from_new_training_current_terms; historical_terms_unknown')
    grouped = defaultdict(list)
    for e in evidence:
        if e['mapping_status']=='resolved':
            grouped[int(e['node_id'])].append(e)
    summaries = []
    for g in gene_rows:
        node = int(g['node_id'])
        rows = grouped[node]
        classes = {r['driver_class'] for r in rows if r['source'] in {'NCG','NCG_list'}}
        canonical_evidence = any(r['source']=='NCG' and r['driver_class']=='canonical' and re.fullmatch(r'[0-9]+',r['PMID']) for r in rows)
        canonical_list = any(r['source']=='NCG_list' and r['driver_class']=='canonical' for r in rows)
        strict_positive = canonical_evidence and canonical_list and g['mapping_status']=='resolved'
        # Proxy candidates only; cannot be released as strict negatives while the old construction is unresolved.
        # OncoKB is inventory/audit only and must not determine new training membership.
        decision_rows = [r for r in rows if r['source'] != 'OncoKB']
        proxy_candidate = node in negatives and not decision_rows and g['mapping_status']=='resolved'
        summaries.append(dict(node_id=node,source_id=g['source_id'],hgnc_id=g['hgnc_id'],mapping_status=g['mapping_status'],
                              legacy_label='conflict' if node in positives&negatives else 'positive' if node in positives else 'negative_proxy' if node in negatives else 'unlabeled',
                              ncg_classes='|'.join(sorted(classes-{'unspecified'})),sources='|'.join(sorted({r['source'] for r in rows})),
                              evidence_records=len(rows),strict_positive_candidate=int(strict_positive),strict_negative_proxy_candidate=int(proxy_candidate),
                              decision_status='proposed_not_released',conflict_flag=int(node in positives&negatives or any(r['conflict_flag']=='true' for r in rows))))
    write_tsv(out/'labels_provenance.tsv',evidence,EVIDENCE_FIELDS)
    write_tsv(out/'labels_summary.tsv',summaries,list(summaries[0]))
    write_tsv(out/'candidates/strict_positive_candidates.tsv',[r for r in summaries if r['strict_positive_candidate']],list(summaries[0]))
    write_tsv(out/'candidates/negative_proxy_candidates_NOT_RELEASED.tsv',[r for r in summaries if r['strict_negative_proxy_candidate']],list(summaries[0]))
    write_tsv(out/'label_conflicts.tsv',conflicts,['node_id','hgnc_id','source','source_row','reason','evidence_level'])
    unmapped = [r for r in evidence if r['mapping_status']!='resolved']
    write_tsv(out/'label_mapping_issues.tsv',unmapped,EVIDENCE_FIELDS)
    report = dict(legacy_positive_count=len(positives),legacy_negative_proxy_count=len(negatives),legacy_overlap=sorted(positives&negatives),
                  evidence_records=len(evidence),mapping_statuses=dict(Counter(r['mapping_status'] for r in evidence)),
                  source_records=dict(Counter(r['source'] for r in evidence)),
                  strict_positive_candidates=sum(r['strict_positive_candidate'] for r in summaries),
                  strict_positive_candidate_legacy_labels=dict(Counter(r['legacy_label'] for r in summaries if r['strict_positive_candidate'])),
                  strict_negative_proxy_candidates=sum(r['strict_negative_proxy_candidate'] for r in summaries),
                  legacy_negative_with_evidence_nodes=len({r['node_id'] for r in conflicts}),
                  canonical_candidate_list_overlap=sorted(canonical&candidate),
                  release_status='not_released_user_requested_candidate_lists_only',
                  historical_missing_inputs='user_confirmed_unavailable_2026-10-23',
                  unresolved=['Legacy negative construction not reproducible from available notebook inputs.',
                              'NCG source usage conditions not yet verified; website unavailable.',
                              'NCG PMIDs have not been adjudicated as first driver evidence.',
                              'mmc2.xlsx and extract_name.txt require identification before scientific interpretation.',
                              'No temporal evaluation set established; old test usage unknown.'])
    write_json(out/'labels_audit.json',report)
    log('04 标签证据审计', 'completed_with_unresolved_items', json.dumps(report,ensure_ascii=False),out)
    return report


def inspect_splits(splits, test_ids, test_labels, mapping, gene_rows, positives, negatives):
    issues, output = [], []
    genes = ordered_genes(mapping)
    identity = {int(r['node_id']):r['hgnc_id'] for r in gene_rows}
    all_labels, validation_counts, development = {}, Counter(), None
    fold_summary = []
    def add(ids, labels, fold, split):
        if len(ids)!=len(labels):
            issues.append(f'{fold}/{split}: label length mismatch')
        if len(ids)!=len(set(ids)):
            issues.append(f'{fold}/{split}: duplicate IDs')
        if set(labels)!={0,1}:
            issues.append(f'{fold}/{split}: both classes not present or invalid labels')
        for node,y in zip(ids,labels):
            if type(node) is not int or not 0<=node<len(genes):
                issues.append(f'{fold}/{split}: out-of-range or noninteger node {node}')
                continue
            if node in all_labels and all_labels[node]!=y:
                issues.append(f'Inconsistent label for node {node}')
            all_labels[node]=y
            expected = 1 if node in positives else 0 if node in negatives else None
            if y!=expected:
                issues.append(f'Cached label differs from legacy PKL for node {node}')
            output.append(dict(fold=fold,split=split,node_id=node,source_id=genes[node],hgnc_id=identity[node],label=y))
    for fold,batch in enumerate(splits):
        train,val,ty,vy = [x.tolist() if hasattr(x,'tolist') else list(x) for x in batch]
        ts,vs = set(train),set(val)
        if ts&vs:
            issues.append(f'Fold {fold}: train/validation overlap')
        if development is None:
            development=ts|vs
        elif development!=ts|vs:
            issues.append(f'Fold {fold}: development universe changed')
        validation_counts.update(val)
        train_h={identity[i] for i in ts if identity.get(i)}
        val_h={identity[i] for i in vs if identity.get(i)}
        if train_h&val_h:
            issues.append(f'Fold {fold}: HGNC identity leakage')
        add(train,ty,fold,'train'); add(val,vy,fold,'val')
        fold_summary.append(dict(fold=fold,train=len(train),validation=len(val),train_classes=dict(Counter(ty)),validation_classes=dict(Counter(vy))))
    development=development or set()
    if len(splits)!=5:
        issues.append('Expected exactly five folds')
    if set(validation_counts)!=development or any(c!=1 for c in validation_counts.values()):
        issues.append('Validation folds do not cover development exactly once')
    if development&set(test_ids):
        issues.append('Development/test node overlap')
    if {identity[i] for i in development if identity.get(i)} & {identity[i] for i in test_ids if identity.get(i)}:
        issues.append('Development/test HGNC identity overlap')
    add(test_ids,test_labels,'heldout','test')
    if development|set(test_ids)!=positives|negatives:
        issues.append('Split union differs from legacy label union')
    report=dict(status='passed' if not issues else 'failed', issues=sorted(set(issues)), folds=fold_summary,
                development_count=len(development),test_count=len(test_ids),validation_unique=len(validation_counts),
                never_validated_nodes=sorted(development-set(validation_counts)),test_usage_history='unknown',
                test_independence='not_established', unresolved_identity_nodes=sum(not r['hgnc_id'] for r in gene_rows),
                scope='Structural checks only; passing does not establish scientific label validity or historical independence.')
    return report, output


def split_audit(root,out,mapping,gene_rows):
    splits=load_pickle(root/'Intermediate/label_set/train_batch.pkl')
    test_ids=load_tensor(root/'Intermediate/label_set/mydatatest_mask.pt').tolist()
    test_y=load_tensor(root/'Intermediate/label_set/mydatatest_label.pt').tolist()
    pos=set(load_pickle(root/'data/labels/now_pos427.pkl'))
    neg=set(load_pickle(root/'data/labels/now_neg427.pkl'))
    report, rows=inspect_splits(splits,test_ids,test_y,mapping,gene_rows,pos,neg)
    write_json(out/'split_manifest.json',report)
    write_tsv(out/'splits/legacy_membership.tsv',rows,['fold','split','node_id','source_id','hgnc_id','label'])
    for fold in range(5):
        for split in ['train','val']:
            write_tsv(out/f'splits/fold_{fold}_{split}.tsv',[r for r in rows if r['fold']==fold and r['split']==split],['node_id','source_id','hgnc_id','label'])
    write_tsv(out/'splits/test.tsv',[r for r in rows if r['split']=='test'],['node_id','source_id','hgnc_id','label'])
    log('05 现有划分审计',report['status'],json.dumps(report,ensure_ascii=False),out)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=p.parse_args(); root=args.root.resolve(); out=args.out.resolve()
    try:
        verify_frozen(root,out)
        mapping,index,rows=mapping_audit(root,out)
        features=feature_audit(root,out,mapping)
        labels=label_audit(root,out,mapping,index,rows)
        splits=split_audit(root,out,mapping,rows)
        graph_report=out/'graph_validation_local.json'
        graph_validation=json.loads(graph_report.read_text(encoding='utf-8')) if graph_report.exists() else {'status':'not_run'}
        if graph_validation.get('status')=='passed':
            expected_codes={name:sha256(ROOT/name) for name in VALIDATION_CODE}
            if graph_validation.get('baseline_manifest_sha256')!=sha256(out/'baseline_manifest.json') or graph_validation.get('validation_code_sha256')!=expected_codes:
                graph_validation=dict(status='stale_report_rerun_graph_validation')
        if graph_validation.get('status')=='passed':
            features['dgl_graph_status']='passed_local_actual_graph'
            features['blocks_status']='passed_local_actual_caches'
            write_json(out/'feature_alignment_report.json',features)
        report=dict(created_at=now(),status='local_audit_complete_server_and_protocol_pending',
                    features=features,labels=labels,splits=splits,
                    graph_validation=graph_validation,
                    original_data_unchanged=True, training_started=False, independent_test_established=False)
        verify_frozen(root,out)
        write_json(out/'quality_report.json',report)
        if not features['float32_source_exact_match'] or not all(features['mapping_copies_match'].values()) or splits['status']!='passed':
            report['status']='failed_local_audit'
            write_json(out/'quality_report.json',report)
            raise ValueError('Feature alignment or split integrity failed; see quality_report.json')
        print(json.dumps({'status':report['status'],'labels':labels,'splits':splits['status']},ensure_ascii=False,indent=2))
    except Exception as exc:
        log('P0 audit','failed',f'{type(exc).__name__}: {exc}',out)
        raise


if __name__=='__main__':
    main()
