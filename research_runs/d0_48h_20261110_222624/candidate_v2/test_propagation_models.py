import unittest
import dgl
import torch
from propagation_models import MultiPathLogitPropagation, incoming_mean_matrix, ppr_coefficients, factory


class PropagationModelsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.tensor([[3., 1.], [12., 2.], [6., 4.]])
        self.graphs = [
            dgl.graph(([0, 0, 0, 1, 2, 1], [0, 1, 1, 1, 2, 2]), num_nodes=3),
            dgl.graph(([0, 1, 2, 2], [0, 1, 2, 0]), num_nodes=3),
        ]

    def model(self, **kwargs):
        torch.manual_seed(91)
        return MultiPathLogitPropagation(self.x, self.graphs, width=8, dropout=0., **kwargs)

    def test_directed_duplicate_mean_and_no_graph_mutation(self):
        graph = self.graphs[0]
        original_src, original_dst = [v.clone() for v in graph.edges(order='eid')]
        p = incoming_mean_matrix(graph, 3)
        torch.testing.assert_close(torch.sparse.mm(p, self.x[:, :1]), torch.tensor([[3.], [6.], [9.]]))
        torch.testing.assert_close(p.to_dense().sum(1), torch.ones(3))
        src, dst = graph.edges(order='eid')
        torch.testing.assert_close(src, original_src)
        torch.testing.assert_close(dst, original_dst)

    def test_ppr_polynomial_hand_calculation(self):
        model = self.model(hops=2, alpha=.5, path_mode='ppi')
        model.eval()
        raw = model.predict(model.encoder(self.x))
        p = torch.tensor([[1., 0., 0.], [2./3., 1./3., 0.], [0., .5, .5]])
        expected = .5 * raw + .25 * (p @ raw) + .25 * (p @ p @ raw)
        actual = model([], self.x)[1]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(ppr_coefficients(2, .5), torch.tensor([.5, .25, .25]))

    def test_k_zero_exact_mlp_after_parameter_changes(self):
        for mode in ('fixed', 'signed', 'softmax'):
            model = self.model(hops=0, gamma_mode=mode)
            with torch.no_grad():
                model.path_logits.copy_(torch.tensor([4., -3.]))
                if mode == 'signed':
                    model.gamma.fill_(7.)
            model.eval()
            expected = model.predict(model.encoder(self.x))
            torch.testing.assert_close(model([], self.x)[1], expected, rtol=0., atol=0.)

    def test_fixed_signed_softmax_same_initial_output(self):
        models = [self.model(hops=3, gamma_mode=mode) for mode in ('fixed', 'signed', 'softmax')]
        reference = models[0]([], self.x)[1]
        for model in models[1:]:
            torch.testing.assert_close(model.coefficients(), models[0].coefficients())
            torch.testing.assert_close(model([], self.x)[1], reference)

    def test_finite_gradients_and_beta(self):
        for mode in ('fixed', 'signed', 'softmax'):
            model = self.model(hops=2, gamma_mode=mode)
            _, logits, _, beta = model([], self.x)
            self.assertEqual(tuple(beta.shape), (3, 2, 1))
            torch.testing.assert_close(beta.sum(1), torch.ones(3, 1))
            torch.nn.functional.cross_entropy(logits[:2], torch.tensor([0, 1])).backward()
            for name, param in model.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertTrue(torch.isfinite(logits).all())

    def test_ppi_only_and_sparse_nonpersistent_buffers(self):
        model = self.model(hops=2, path_mode='ppi')
        beta = model([], self.x)[3]
        torch.testing.assert_close(beta[:, 0], torch.ones(3, 1))
        torch.testing.assert_close(beta[:, 1], torch.zeros(3, 1))
        self.assertTrue(model.transition_0.is_sparse)
        self.assertFalse(model.transition_0.requires_grad)
        self.assertFalse(any(k.startswith('transition_') for k in model.state_dict()))
        # A change to an unselected path cannot affect the model's output.
        g2 = dgl.graph(([0, 1, 2], [0, 1, 2]), num_nodes=3)
        torch.manual_seed(91)
        other = MultiPathLogitPropagation(self.x, [self.graphs[0], g2], width=8,
                                         dropout=0., hops=2, path_mode='ppi')
        torch.testing.assert_close(model([], self.x)[1], other([], self.x)[1], rtol=0, atol=0)

    def test_factory_and_no_label_argument(self):
        config = {'dropout':0., 'research':{'family':'propagation', 'width':8, 'hops':2}}
        model, placed = factory(config, self.x, self.graphs, torch.device('cpu'))
        self.assertEqual(placed, [])
        self.assertIsNone(model.metapath_devices)
        with self.assertRaises(TypeError):
            model([], self.x, labels=torch.tensor([0, 1, 0]))
        with self.assertRaises(TypeError):
            factory(config, self.x, self.graphs, torch.device('cpu'), labels=torch.tensor([0, 1, 0]))

    def test_zero_indegree_rejected(self):
        g = dgl.graph(([0], [0]), num_nodes=3)
        with self.assertRaises(ValueError):
            incoming_mean_matrix(g, 3)


if __name__ == '__main__':
    unittest.main()
