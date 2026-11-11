"""Server-side numerical checks; run: python -m pytest -q tests/test_ablation.py."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import dgl
from dgl.nn import GATv2Conv

from ablation_config import EXPERIMENTS
from fusion import SemanticAttention, ViewInteraction
from model import MCDHGN, MemEfficientGATv2Conv
import run_ablation as runner
from summarize_ablations import summarize


def test_global_matches_original_output_and_gradient():
    torch.manual_seed(2)
    layer = SemanticAttention(6).double()
    z = torch.randn(7, 3, 6, dtype=torch.double, requires_grad=True)
    out, beta = layer(z)
    old_beta = torch.softmax(layer.project(z).mean(0), dim=0).expand(7, 3, 1)
    expected = (old_beta * z).sum(1)
    torch.testing.assert_close(out, expected)
    grad = torch.autograd.grad(out.square().sum(), z, retain_graph=True)[0]
    old_grad = torch.autograd.grad(expected.square().sum(), z)[0]
    torch.testing.assert_close(grad, old_grad)
    assert beta.shape == (3, 1)


def test_node_weights_depend_on_gene_and_sum_over_views():
    layer = SemanticAttention(1, hidden_size=1, mode="node")
    with torch.no_grad():
        layer.project[0].weight.fill_(1)
        layer.project[0].bias.zero_()
        layer.project[2].weight.fill_(1)
    z = torch.tensor([[[2.0], [-2.0]], [[-2.0], [2.0]]])
    _, beta = layer(z)
    assert beta[0, 0, 0] > beta[0, 1, 0]
    assert beta[1, 1, 0] > beta[1, 0, 0]
    torch.testing.assert_close(beta.sum(1), torch.ones(2, 1))
    perturbed = z.clone()
    perturbed[1] = 50
    torch.testing.assert_close(layer(perturbed)[1][0], beta[0])


def test_cross_view_never_mixes_genes_and_is_view_permutation_equivariant():
    torch.manual_seed(3)
    module = ViewInteraction(8, "cross", rank=3, alpha=0.6)
    z = torch.randn(5, 4, 8)
    before = module(z)
    changed = z.clone()
    changed[1] += 10
    torch.testing.assert_close(module(changed)[0], before[0])
    perm = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(module(z[:, perm]), before[:, perm])
    changed = z.clone()
    changed[0, 1] += 3
    assert not torch.allclose(module(changed)[0, 0], before[0, 0])


def test_chunking_preserves_output_and_gradient():
    torch.manual_seed(4)
    small = ViewInteraction(8, "cross", rank=3, chunk_size=2).double()
    large = copy.deepcopy(small)
    large.chunk_size = 100
    a = torch.randn(7, 4, 8, dtype=torch.double, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    torch.testing.assert_close(small(a), large(b))
    small(a).square().sum().backward()
    large(b).square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad)
    for p, q in zip(small.parameters(), large.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_retention_endpoint_and_exact_capacity_control():
    module = ViewInteraction(8, "cross", rank=3, alpha=0)
    x = torch.randn(5, 4, 8)
    torch.testing.assert_close(module(x), x)
    module.alpha = 1
    shared = module(x)
    module.alpha = 0.6
    torch.testing.assert_close(module(x), 0.4 * x + 0.6 * shared)
    control = ViewInteraction(8, "pointwise", rank=3)
    assert sum(p.numel() for p in module.parameters()) == sum(p.numel() for p in control.parameters()) == 96


@pytest.mark.parametrize("exp", list(EXPERIMENTS))
def test_all_experiments_forward_and_backward(exp):
    torch.manual_seed(10)
    nodes = torch.arange(6)
    graph = dgl.graph((torch.cat([nodes, nodes]), torch.cat([nodes, (nodes + 1) % 6])), num_nodes=6)
    model = MCDHGN(3, 5, 4, 2, [2], 0, fusion_dim=3, **EXPERIMENTS[exp])
    h, logits, _, beta = model([graph] * 3, torch.randn(6, 5))
    assert h.shape == (6, 8) and logits.shape == (6, 2)
    assert beta.shape == ((3, 1) if EXPERIMENTS[exp]["semantic_mode"] == "global" else (6, 3, 1))
    torch.nn.functional.cross_entropy(logits, nodes % 2).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_optional_fusion_does_not_shift_common_initialization():
    models = []
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6"):
        torch.manual_seed(42)
        models.append(MCDHGN(2, 5, 4, 2, [2], 0, fusion_dim=3, **EXPERIMENTS[exp]))
    base = models[0].state_dict()
    for model in models[1:]:
        for name, value in base.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)


def test_chunked_gatv2_matches_dgl_scores_outputs_and_gradients():
    torch.manual_seed(19)
    graph = dgl.add_self_loop(dgl.graph(([0, 1, 2, 0, 3], [1, 2, 3, 3, 0]), num_nodes=4))
    reference = GATv2Conv(5, 3, 2, feat_drop=0, attn_drop=0)
    actual = MemEfficientGATv2Conv(5, 3, 2, feat_drop=0, attn_drop=0, edge_chunk_size=2)
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(4, 5, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    out, weights = reference(graph, x, get_attention=True)
    other, other_weights = actual(graph, y, get_attention=True)
    torch.testing.assert_close(out, other, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weights, other_weights, atol=1e-5, rtol=1e-5)
    out.square().sum().backward()
    other.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-5)
    for p, q in zip(reference.parameters(), actual.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-5)


def test_block_ids_are_mapped_to_global_gene_ids():
    runner.dependencies()
    graph = dgl.add_self_loop(dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3))
    block = dgl.to_block(graph, torch.tensor([2, 0, 1]))
    restored = runner.global_graph(block, 3)
    original_edges = set(zip(*[v.tolist() for v in graph.edges()]))
    restored_edges = set(zip(*[v.tolist() for v in restored.edges()]))
    assert restored_edges == original_edges


def test_score_is_invariant_to_per_gene_common_logit_offset():
    runner.dependencies()
    logits = torch.tensor([[1., 2.], [3., 1.], [0., 3.], [4., 2.]])
    labels = torch.tensor([1, 0, 1, 0])
    a, _ = runner.classification_metrics(logits, labels)
    b, _ = runner.classification_metrics(logits + torch.tensor([[8.], [-6.], [4.], [-2.]]), labels)
    for metric in ("ap", "auc", "accuracy", "macro_f1"):
        assert a[metric] == b[metric]


def test_summary_pairs_same_seed_and_fold():
    groups = {}
    for exp, offset in (("E1", 0), ("E2", .01)):
        groups[exp] = {}
        for fold in range(2):
            groups[exp][(42, fold)] = {
                "validation": {m: .7 + fold * .1 + offset for m in ("ap", "auc", "accuracy", "macro_f1")},
                "parameters": 100, "train_seconds": 1,
            }
    report = summarize(groups, expected_folds=2)
    assert report["paired_contrasts"]["node_weights_only"]["metrics"]["ap"]["mean_delta_pp"] == pytest.approx(1)
    del groups["E2"][(42, 1)]
    with pytest.raises(ValueError, match="matching"):
        summarize(groups, expected_folds=2)
