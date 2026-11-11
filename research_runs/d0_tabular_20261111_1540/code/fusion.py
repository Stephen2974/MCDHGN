"""Independent, testable metapath fusion modules (tensors use N, M, D order).

The cross-view module is an adaptation inspired by MODIG, not a reproduction
of its public implementation's within-view, across-gene self-attention.
"""

import math

import torch
from torch import nn


class SemanticAttention(nn.Module):
    """Pool M views globally or separately for each of N genes.

    The global default retains the original state_dict keys and returns (M, 1)
    weights. Node mode returns (N, M, 1). Both modes have identical parameters.
    """

    def __init__(self, in_size, hidden_size=128, mode="global"):
        super().__init__()
        if mode not in ("global", "node"):
            raise ValueError("semantic mode must be 'global' or 'node'")
        self.mode = mode
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        if z.ndim != 3 or z.shape[1] == 0:
            raise ValueError("expected nonempty (genes, metapaths, features)")
        scores = self.project(z)  # (N, M, 1)
        if self.mode == "global":
            beta = torch.softmax(scores.mean(0), dim=0)  # (M, 1)
            return (beta.unsqueeze(0) * z).sum(1), beta
        beta = torch.softmax(scores, dim=1)  # normalize views, NOT genes
        return (beta * z).sum(1), beta


class ViewInteraction(nn.Module):
    """Mix views of the SAME gene, retaining its graph-encoded representation.

    cross: Q,K,V in R^(N,M,r); attention in R^(N,M,M), never (N,N).
    pointwise: an MLP applied independently to each view, with exactly the
      same 4*D*r trainable weights as cross. It is a capacity control.
    alpha=0 is the identity, alpha=1 removes the retained representation.
    There is intentionally no extra dropout, normalization, or raw-feature
    residual here: those would confound the initial ablations.
    """

    def __init__(self, in_size, mode="none", rank=64, alpha=0.6, chunk_size=1024):
        super().__init__()
        if mode not in ("none", "cross", "pointwise"):
            raise ValueError("fusion mode must be none, cross, or pointwise")
        if rank <= 0 or chunk_size <= 0 or not 0 <= alpha <= 1:
            raise ValueError("rank/chunk_size must be positive; alpha in [0, 1]")
        self.mode, self.rank = mode, rank
        self.alpha, self.chunk_size = float(alpha), int(chunk_size)
        if mode == "cross":
            self.query = nn.Linear(in_size, rank, bias=False)
            self.key = nn.Linear(in_size, rank, bias=False)
            self.value = nn.Linear(in_size, rank, bias=False)
            self.output = nn.Linear(rank, in_size, bias=False)
        elif mode == "pointwise":
            self.mlp = nn.Sequential(
                nn.Linear(in_size, 2 * rank, bias=False),
                nn.ELU(),
                nn.Linear(2 * rank, in_size, bias=False),
            )

    def forward(self, z):
        if z.ndim != 3:
            raise ValueError("expected (genes, metapaths, features)")
        if self.mode == "none" or self.alpha == 0:
            return z
        chunks = []
        for part in z.split(self.chunk_size, dim=0):
            if self.mode == "cross":
                q, k, v = self.query(part), self.key(part), self.value(part)
                weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.rank), dim=-1)
                shared = self.output(weights @ v)
            else:
                shared = self.mlp(part)
            chunks.append((1 - self.alpha) * part + self.alpha * shared)
        return torch.cat(chunks, dim=0)
