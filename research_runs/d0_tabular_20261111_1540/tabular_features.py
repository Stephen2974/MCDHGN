"""Label-free features from the existing directed, sampled metapath edges."""
import numpy as np
from scipy import sparse


def feature_views(features, edge_lists):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite N x F features')
    first, second, deviations = [], [], []
    for src, dst in edge_lists:
        src, dst = np.asarray(src), np.asarray(dst)
        degree = np.bincount(dst, minlength=len(x))
        if len(src) != len(dst) or len(degree) != len(x) or np.any(degree == 0):
            raise ValueError('Invalid frozen graph or zero incoming degree')
        # COO -> CSR sums duplicate edges, retaining their multiplicity.
        p = sparse.coo_matrix((1.0 / degree[dst], (dst, src)),
                              shape=(len(x), len(x))).tocsr()
        mean = p @ x
        first.append(mean)
        second.append(p @ mean)
        deviations.append(np.sqrt(np.maximum(p @ (x * x) - mean * mean, 0)))
    if not first:
        raise ValueError('Expected at least one metapath')
    return {name: np.asarray(np.concatenate(parts, axis=1), dtype=np.float32)
            for name, parts in {
                'raw': [x], 'mean1': [x] + first,
                'mean2': [x] + first + second,
                'moments2': [x] + first + second + deviations,
            }.items()}
