import copy
import unittest

import dgl
import torch

from research_models import incoming_mean_features, make_research_model, MultiPathSIGN
from model import MCDHGN


class ResearchModelsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.x = torch.randn(6, 3)
        ids = torch.arange(6)
        self.graphs = [dgl.graph((torch.cat((ids, ids)), torch.cat((ids, (ids + k) % 6))),
                                 num_nodes=6) for k in (1, 2)]
        self.config = dict(hidden_size=4, heads=2, dropout=0.0, fusion_dim=2,
                           fusion_chunk_size=4,
                           architecture=dict(use_gatv2=False, semantic_mode='node', fusion_mode='none'))

    def model(self, **settings):
        cfg = copy.deepcopy(self.config)
        cfg['research'] = settings
        return make_research_model(cfg, self.x, self.graphs, torch.device('cpu'))

    def test_propagation_direction_degree_and_duplicate_edges(self):
        # Incoming: node0 <- node0; node1 <- node0 twice, node1 once.
        graph = dgl.graph(([0, 0, 0, 1], [0, 1, 1, 1]), num_nodes=2)
        x = torch.tensor([[3.0], [12.0]])
        actual = incoming_mean_features(graph, x, hops=2)
        torch.testing.assert_close(actual, torch.tensor([[3., 3., 3.], [12., 6., 4.]]))
        torch.testing.assert_close(incoming_mean_features(graph, x, 1, False), actual[:, 1:2])

    def test_zero_degree_rejected(self):
        g = dgl.graph(([0], [0]), num_nodes=2)
        with self.assertRaises(ValueError):
            incoming_mean_features(g, torch.ones(2, 3))

    def test_raw_does_not_use_graphs(self):
        model, placed = self.model(family='raw', width=8)
        self.assertEqual(placed, [])
        model.eval()
        a = model([], self.x)[1]
        b = model(None, self.x)[1]
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_all_families_forward_backward(self):
        settings = [dict(family='raw', width=8),
                    dict(family='mcdhgn'),
                    dict(family='mcdhgn', raw_residual=True),
                    dict(family='mcdhgn', raw_residual=True, layer_norm=True, head='linear'),
                    dict(family='sign', width=8, raw_branch=True),
                    dict(family='sign', width=8, raw_branch=False),
                    dict(family='sign', width=8, include_zero=False, layer_norm=True)]
        for item in settings:
            with self.subTest(item=item):
                model, placed = self.model(**item)
                h, logits, _, beta = model(placed, self.x)
                self.assertEqual(tuple(logits.shape), (6, 2))
                self.assertEqual(tuple(beta.shape), (6, 2, 1))
                torch.testing.assert_close(beta.sum(1), torch.ones(6, 1))
                self.assertTrue(torch.isfinite(h).all())
                loss = torch.nn.functional.cross_entropy(logits[:4], torch.tensor([0, 1, 0, 1]))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

    def test_sign_no_raw_branch_and_cache(self):
        model, _ = self.model(family='sign', width=8, raw_branch=False)
        self.assertIsNone(model.raw)
        self.assertFalse(any('propagated_' in k for k in model.state_dict()))
        self.assertTrue(all(not v.requires_grad for n, v in model.named_buffers() if n.startswith('propagated_')))
        model.eval()
        torch.testing.assert_close(model([], self.x)[1], model([], self.x + 10)[1])
        self.assertEqual(model.predict.in_features, 8)

    def test_baseline_exact_initialization_and_output(self):
        torch.manual_seed(19)
        reference = MCDHGN(num_meta_paths=2, in_size=3, hidden_size=4, out_size=2,
                           num_heads=[2], dropout=0.0, share_weights=False,
                           metapath_device_ids=None, primary_device_id=0,
                           edge_counts=[g.num_edges() for g in self.graphs], parallel_metapaths=False,
                           fusion_dim=2, fusion_chunk_size=4, **self.config['architecture'])
        torch.manual_seed(19)
        actual, placed = self.model(family='mcdhgn', raw_residual=False, layer_norm=False, head='original')
        self.assertEqual(set(reference.state_dict()), set(actual.state_dict()))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
        reference.eval()
        actual.eval()
        torch.testing.assert_close(reference(self.graphs, self.x)[1], actual(placed, self.x)[1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
