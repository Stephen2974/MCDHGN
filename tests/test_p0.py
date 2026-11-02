"""Tests of identity ambiguity, row alignment and leakage rejection."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from data_processing.p0.audit import HgncIndex, NodeResolver, aligned_features, build_gene_map, inspect_splits


def record(hid,symbol,**kw):
    return dict(hgnc_id=hid,symbol=symbol,status='Approved',**kw)


class IdentityTests(unittest.TestCase):
    def test_ambiguous_alias_is_never_first_match(self):
        idx=HgncIndex([record('HGNC:1','A',alias_symbol='OLD'),record('HGNC:2','B',prev_symbol='OLD')])
        rows=build_gene_map({'OLD':0},idx)
        self.assertEqual(rows[0]['mapping_status'],'ambiguous')
        self.assertEqual(rows[0]['hgnc_id'],'')

    def test_exact_approved_name_precedes_another_genes_alias(self):
        idx=HgncIndex([record('HGNC:1','A'),record('HGNC:2','B',alias_symbol='A')])
        self.assertEqual(build_gene_map({'A':0},idx)[0]['hgnc_id'],'HGNC:1')

    def test_duplicate_old_nodes_are_not_merged(self):
        idx=HgncIndex([record('HGNC:1','A',prev_symbol='OLD')])
        rows=build_gene_map({'A':0,'OLD':1},idx)
        self.assertEqual([r['node_id'] for r in rows],[0,1])
        self.assertTrue(all(r['mapping_status']=='duplicate_identity' for r in rows))
        self.assertEqual(NodeResolver(idx,rows).resolve('A')[2],'duplicate_identity')

    def test_ensembl_version_stripping(self):
        idx=HgncIndex([record('HGNC:1','A',ensembl_gene_id='ENSG000001')])
        self.assertEqual(idx.resolve('ENSG000001.7','ensembl_gene_id')[0],{'HGNC:1'})

    def test_disagreeing_stable_identifiers_rejected(self):
        idx=HgncIndex([record('HGNC:1','A',entrez_id='1'),record('HGNC:2','B',entrez_id='2')])
        rows=build_gene_map({'A':0,'B':1},idx)
        self.assertEqual(NodeResolver(idx,rows).resolve('A','2')[2],'identifier_conflict')

    def test_features_ignore_dictionary_insertion_order(self):
        actual=aligned_features({'A':1,'B':0},{'A':torch.tensor([10.]),'B':torch.tensor([20.])})
        self.assertEqual(actual.tolist(),[[20.],[10.]])

    def test_noncontiguous_nodes_fail(self):
        with self.assertRaises(ValueError):
            aligned_features({'A':2},{'A':torch.tensor([1.])})

    def test_training_loader_mapping_contract(self):
        from run_ablation import validate_node_mapping
        validate_node_mapping({'B':1,'A':0},2)
        for mapping in [{'A':0,'B':0},{'A':1,'B':2},{'A':0.0,'B':1},{'A':0}]:
            with self.assertRaises(ValueError):
                validate_node_mapping(mapping,2)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.mapping={f'G{i}':i for i in range(12)}
        self.rows=[dict(node_id=i,hgnc_id=f'HGNC:{i}') for i in range(12)]
        self.pos=set(range(1,12,2)); self.neg=set(range(0,12,2))
        self.splits=[]
        for fold in range(5):
            val=[2*fold,2*fold+1]; train=sorted(set(range(10))-set(val))
            self.splits.append((train,val,[x%2 for x in train],[x%2 for x in val]))

    def check(self,splits=None,ids=None,rows=None):
        return inspect_splits(splits or self.splits,ids or [10,11],[0,1],self.mapping,rows or self.rows,self.pos,self.neg)[0]

    def test_valid_partition(self):
        self.assertEqual(self.check()['status'],'passed')

    def test_node_leakage(self):
        self.assertIn('Development/test node overlap',self.check(ids=[0,11])['issues'])

    def test_alias_identity_leakage(self):
        self.rows[10]['hgnc_id']=self.rows[0]['hgnc_id']
        self.assertIn('Development/test HGNC identity overlap',self.check()['issues'])

    def test_omitted_validation_gene(self):
        self.splits[4]=(self.splits[4][0]+[8],[9],self.splits[4][2]+[0],[1])
        self.assertIn('Validation folds do not cover development exactly once',self.check()['issues'])

    def test_corrupted_label(self):
        self.splits[0][2][0]=1-self.splits[0][2][0]
        self.assertTrue(any('Cached label differs' in x for x in self.check()['issues']))


class ReleaseLayoutTests(unittest.TestCase):
    def setUp(self):
        from data_processing.p0.common import sha256, write_json
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.out=self.root/'bundle'
        self.out.mkdir()
        (self.root/'data.bin').write_bytes(b'unchanged data')
        (self.root/'moved.py').write_text('current code',encoding='utf-8')
        (self.out/'report.json').write_text('{}',encoding='utf-8')
        write_json(self.out/'baseline_manifest.json',{'files':[{'kind':'data','path':'data.bin','sha256':sha256(self.root/'data.bin')}]})
        original=dict(baseline_manifest_sha256=sha256(self.out/'baseline_manifest.json'),
                      artifacts=[dict(path='report.json',sha256=sha256(self.out/'report.json'),size_bytes=2)],
                      code_sha256={'old_location.py':'historical_hash'})
        write_json(self.out/'manifest.json',original)
        self.layout=dict(original,previous_manifest_sha256=sha256(self.out/'manifest.json'),
                         code_sha256={'moved.py':sha256(self.root/'moved.py')})
        write_json(self.out/'layout_manifest.json',self.layout)

    def verify(self):
        from data_processing.p0.release import verify
        with patch('data_processing.p0.release.log'):
            verify(self.root,self.out)

    def test_migration_keeps_original_artifacts_and_checks_new_code(self):
        self.verify()
        (self.root/'moved.py').write_text('tampered code',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'code:moved.py'):
            self.verify()

    def test_migration_rejects_changed_original_manifest(self):
        (self.out/'manifest.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Original pre-migration manifest changed'):
            self.verify()

    def test_migration_cannot_drop_previously_sealed_artifact(self):
        from data_processing.p0.common import write_json
        self.layout['artifacts']=[]
        write_json(self.out/'layout_manifest.json',self.layout)
        with self.assertRaisesRegex(ValueError,'preserve all previously sealed artifacts'):
            self.verify()


if __name__=='__main__':
    unittest.main()
