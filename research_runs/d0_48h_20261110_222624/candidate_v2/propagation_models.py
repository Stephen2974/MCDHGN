"""APPNP/GPR-inspired multi-metapath logit propagation, not a reproduction.

Only immutable features and frozen, directed metapath graphs enter this module.
One shared MLP produces two logits per node. Every selected path propagates
these logits with an incoming-neighbor mean (including existing duplicate edges
and self loops). One hop coefficient vector is shared across selected paths;
learned global path-softmax weights aggregate their predictions.

For K hops, PPR coefficients are alpha*(1-alpha)**k for k < K and
(1-alpha)**K for k == K, including gamma=[1] when K=0. Signed coefficients
are unconstrained parameters; softmax coefficients are constrained to a simplex.
All three modes start from the same truncated-PPR polynomial. Propagation
matrices are nonpersistent buffers: reconstruct them from the fingerprinted
input graphs before restoring a checkpoint. No input graph is modified.
"""
import torch
from torch import nn


@torch.no_grad()
def incoming_mean_matrix(graph, num_nodes):
    """Return CPU sparse P with P[dst,src] = multiplicity / indegree(dst)."""
    if graph.is_block or graph.num_nodes() != num_nodes:
        raise ValueError('Expected a full globally indexed graph with matching nodes')
    src, dst = graph.edges(order='eid')
    src, dst = src.detach().cpu().long(), dst.detach().cpu().long()
    degree = torch.bincount(dst, minlength=num_nodes).float()
    if (degree == 0).any():
        raise ValueError('Zero indegree: frozen graphs must already contain valid incoming edges')
    return torch.sparse_coo_tensor(
        torch.stack((dst, src)), degree[dst].reciprocal(),
        (num_nodes, num_nodes), dtype=torch.float32).coalesce()


def ppr_coefficients(hops, alpha):
    if type(hops) is not int or hops < 0:
        raise ValueError('hops must be a nonnegative integer')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between 0 and 1')
    values = [alpha * (1 - alpha) ** k for k in range(hops)]
    values.append((1 - alpha) ** hops)
    return torch.tensor(values, dtype=torch.float32)


class MultiPathLogitPropagation(nn.Module):
    metapath_devices = None

    def __init__(self, features, graphs, width=128, dropout=0.4, hops=5,
                 alpha=0.1, gamma_mode='fixed', path_mode='all'):
        super().__init__()
        if features.ndim != 2 or not features.is_floating_point():
            raise ValueError('features must be a floating point node-feature matrix')
        if not graphs or width <= 0 or not 0 <= dropout < 1:
            raise ValueError('Need graphs, a positive width and dropout in [0,1)')
        if gamma_mode not in ('fixed', 'signed', 'softmax'):
            raise ValueError('Unknown gamma_mode')
        if path_mode not in ('all', 'ppi'):
            raise ValueError('Unknown path_mode')
        self.hops, self.gamma_mode = hops, gamma_mode
        self.feature_shape = tuple(features.shape)
        self.num_paths = len(graphs)
        self.selected_paths = tuple(range(len(graphs))) if path_mode == 'all' else (0,)
        initial = ppr_coefficients(hops, alpha)
        self.register_buffer('initial_gamma', initial)
        if gamma_mode == 'fixed':
            self.register_buffer('gamma', initial.clone())
        elif gamma_mode == 'signed':
            self.gamma = nn.Parameter(initial.clone(), requires_grad=hops > 0)
        else:
            self.gamma_logits = nn.Parameter(initial.log(), requires_grad=hops > 0)
        self.path_logits = nn.Parameter(torch.zeros(len(self.selected_paths)),
                                        requires_grad=hops > 0 and len(self.selected_paths) > 1)
        self.encoder = nn.Sequential(nn.Linear(features.shape[1], width), nn.ReLU(), nn.Dropout(dropout))
        self.predict = nn.Linear(width, 2)
        # Keep adjacency sparse and detach topology; no edge-by-hidden-feature tensors.
        for i in self.selected_paths:
            self.register_buffer('transition_' + str(i),
                                 incoming_mean_matrix(graphs[i], len(features)), persistent=False)

    def coefficients(self):
        return self.gamma_logits.softmax(0) if self.gamma_mode == 'softmax' else self.gamma

    def forward(self, graphs, x, return_attention=False):
        if tuple(x.shape) != self.feature_shape:
            raise ValueError('Feature dimensions differ from the graph used for construction')
        h = self.encoder(x)
        raw_logits = self.predict(h)
        gamma, path_weights = self.coefficients(), self.path_logits.softmax(0)
        if self.hops == 0:
            # Exactly the shared MLP in every mode; no rounding from path averaging,
            # no trainable multiplicative gain in the signed-coefficient K=0 control.
            logits = raw_logits
        else:
            path_outputs = []
            for i in self.selected_paths:
                p = getattr(self, 'transition_' + str(i))
                current = raw_logits
                value = gamma[0] * current
                for k in range(1, self.hops + 1):
                    current = torch.sparse.mm(p, current)
                    value = value + gamma[k] * current
                path_outputs.append(value)
            logits = (torch.stack(path_outputs, dim=1) * path_weights[None, :, None]).sum(1)
        # Retain the original path dimension: PPI-only has weight 1 at index 0.
        all_weights = x.new_zeros(self.num_paths)
        selected = torch.tensor(self.selected_paths, device=x.device, dtype=torch.long)
        all_weights = all_weights.index_copy(0, selected, path_weights)
        beta = all_weights[None, :, None].expand(len(x), -1, -1)
        return h, logits, None, beta


def factory(config, features, graphs, primary):
    """Return (model, []); adjacency is compiled once rather than passed each step."""
    settings = dict(config.get('research', {}))
    if settings.pop('family', 'propagation') != 'propagation':
        raise ValueError('This factory only supports research.family=propagation')
    model = MultiPathLogitPropagation(features, graphs, dropout=config['dropout'], **settings)
    return model.to(primary), []


# Both names allow a small explicit dispatch branch in the existing worker.
make_propagation_model = factory
