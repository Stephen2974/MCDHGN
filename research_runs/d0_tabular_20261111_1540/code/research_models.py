"""Isolated, label-free architectural ablations for the frozen D0 inputs.

research.family: raw | mcdhgn | sign (default mcdhgn).
mcdhgn: raw_residual=False, layer_norm=False, head='original'|'linear'.
sign: hops=2, include_zero=True, raw_branch=True, layer_norm=False,
      semantic_mode='node'|'global', width=128 (shared width, separate view MLPs).
raw: width=128, layer_norm=False.

SIGN propagation is incoming-neighbor MEAN on each existing sampled metapath.
Duplicate edges and existing self loops are preserved. No labels are accepted.
The propagation cache is immutable per model and excluded from checkpoints;
reconstruct it from the recorded, fingerprinted graph/features before loading.
"""
import torch
from torch import nn

from fusion import SemanticAttention
from model import MCDHGN, place_metapath_graphs


def dense_encoder(in_size, width, dropout, layer_norm=False):
    return nn.Sequential(
        nn.Linear(in_size, width),
        nn.LayerNorm(width) if layer_norm else nn.Identity(),
        nn.ELU(), nn.Dropout(dropout),
        nn.Linear(width, width), nn.ELU(), nn.Dropout(dropout),
    )


def uniform_beta(x, num_paths):
    return x.new_full((len(x), num_paths, 1), 1.0 / num_paths)


@torch.no_grad()
def incoming_mean_features(graph, features, hops=2, include_zero=True):
    """Return [X, P X, ...] concatenated; P[dst,src]=count/in_degree(dst)."""
    if hops < 0 or (hops == 0 and not include_zero):
        raise ValueError('Need at least one feature channel')
    x = features.detach().float().cpu()
    g = graph.to('cpu')
    if g.num_nodes() != len(x):
        raise ValueError('Feature/graph node count mismatch')
    src, dst = g.edges(order='eid')
    degree = torch.bincount(dst, minlength=len(x)).float()
    if (degree == 0).any():
        raise ValueError('Zero-in-degree node; do not silently change frozen graphs')
    p = torch.sparse_coo_tensor(
        torch.stack((dst.long(), src.long())), 1.0 / degree[dst],
        (len(x), len(x)), dtype=x.dtype).coalesce()
    parts = [x] if include_zero else []
    for _ in range(hops):
        x = torch.sparse.mm(p, x)
        parts.append(x)
    out = torch.cat(parts, dim=1)
    if not torch.isfinite(out).all():
        raise ValueError('Nonfinite propagated features')
    return out


class RawMLP(nn.Module):
    metapath_devices = None

    def __init__(self, in_size, num_paths, width, dropout, layer_norm=False):
        super().__init__()
        self.num_paths = num_paths
        self.encoder = dense_encoder(in_size, width, dropout, layer_norm)
        self.predict = nn.Linear(width, 2)

    def forward(self, graphs, x, return_attention=False):
        h = self.encoder(x)
        # Uniform beta is an interface placeholder, not learned graph importance.
        return h, self.predict(h), None, uniform_beta(x, self.num_paths)


class ResidualMCDHGN(nn.Module):
    metapath_devices = None

    def __init__(self, base, in_size, width, dropout, raw_residual=False,
                 layer_norm=False, head='original'):
        super().__init__()
        if head not in ('original', 'linear'):
            raise ValueError('Unknown head')
        self.base = base
        self.raw = dense_encoder(in_size, width, dropout) if raw_residual else None
        self.norm = nn.LayerNorm(width) if layer_norm else nn.Identity()
        # Replace rather than retain an unused trainable head.
        if head == 'linear':
            self.base.predict = nn.Linear(width, 2)

    def forward(self, graphs, x, return_attention=False):
        # Run the baseline encoder directly so its prediction head is evaluated
        # exactly once, after the explicitly selected residual/normalization.
        h, alpha, beta = x, None, None
        for i, layer in enumerate(self.base.layers):
            h, alpha_i, beta = layer(graphs, h, return_attention=return_attention and i == 0)
            if i == 0:
                alpha = alpha_i
        if self.raw is not None:
            h = h + self.raw(x)
        h = self.norm(h)
        logits = self.base.predict(h)
        if beta.ndim == 2:
            beta = beta.unsqueeze(0).expand(len(x), -1, -1)
        return h, logits, alpha, beta


class MultiPathSIGN(nn.Module):
    metapath_devices = None

    def __init__(self, features, graphs, width=128, dropout=0.4, hops=2,
                 include_zero=True, raw_branch=True, layer_norm=False,
                 semantic_mode='node'):
        super().__init__()
        self.num_paths = len(graphs)
        self.feature_shape = tuple(features.shape)
        self.encoders = nn.ModuleList()
        for i, g in enumerate(graphs):
            cache = incoming_mean_features(g, features, hops, include_zero)
            self.register_buffer('propagated_' + str(i), cache, persistent=False)
            self.encoders.append(dense_encoder(cache.shape[1], width, dropout, layer_norm))
        self.semantic = SemanticAttention(width, hidden_size=min(width, 128), mode=semantic_mode)
        self.raw = dense_encoder(features.shape[1], width, dropout, layer_norm) if raw_branch else None
        self.predict = nn.Linear(width * (2 if raw_branch else 1), 2)

    def forward(self, graphs, x, return_attention=False):
        if tuple(x.shape) != self.feature_shape:
            raise ValueError('This propagation cache belongs to a different feature matrix')
        z = torch.stack([encoder(getattr(self, 'propagated_' + str(i)))
                         for i, encoder in enumerate(self.encoders)], dim=1)
        h, beta = self.semantic(z)
        if self.raw is not None:
            h = torch.cat((h, self.raw(x)), dim=1)
        if beta.ndim == 2:
            beta = beta.unsqueeze(0).expand(len(x), -1, -1)
        return h, self.predict(h), None, beta


def make_research_model(config, features, graphs, primary):
    """Single GPU/CPU model factory matching run_ablation.make_model's result."""
    settings = dict(config.get('research', {}))
    family = settings.pop('family', 'mcdhgn')
    width = settings.pop('width', 128)
    dropout = config['dropout']
    if not graphs:
        raise ValueError('At least one graph is needed for output-path metadata')
    if family == 'raw':
        model = RawMLP(features.shape[1], len(graphs), width, dropout, **settings)
        return model.to(primary), []  # The raw control never reads graph connectivity.
    if family == 'sign':
        model = MultiPathSIGN(features, graphs, width=width, dropout=dropout, **settings)
        return model.to(primary), []  # Propagation is cached once on the CPU.
    if family != 'mcdhgn':
        raise ValueError('Unknown research family: ' + family)
    base = MCDHGN(
        num_meta_paths=len(graphs), in_size=features.shape[1],
        hidden_size=config['hidden_size'], out_size=2, num_heads=[config['heads']],
        dropout=dropout, share_weights=False, metapath_device_ids=None,
        primary_device_id=primary.index or 0,
        edge_counts=[g.num_edges() for g in graphs], parallel_metapaths=False,
        fusion_dim=config['fusion_dim'], fusion_chunk_size=config['fusion_chunk_size'],
        **config['architecture'])
    raw = settings.pop('raw_residual', False)
    norm = settings.pop('layer_norm', False)
    head = settings.pop('head', 'original')
    if settings:
        raise ValueError('Unknown research settings: ' + repr(settings))
    # Identity settings return the actual unmodified baseline, including state keys.
    model = (ResidualMCDHGN(base, features.shape[1], config['hidden_size'] * config['heads'],
                           dropout, raw, norm, head) if raw or norm or head != 'original' else base)
    return model.to(primary), place_metapath_graphs(graphs, None, primary)
