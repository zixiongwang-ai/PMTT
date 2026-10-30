"""Fixed positive, hard-negative and uniform candidate sampling."""
import numpy as np
from .io import seed

def soft_labels(xy, spec):
    gh, gw = spec['grid']
    xy = np.asarray(xy) * spec['scale']
    q = np.clip(xy / 16 - 0.5, [0, 0], [gw - 1, gh - 1])
    lo = q.astype(int)
    hi = np.minimum(lo + 1, [gw - 1, gh - 1])
    f = q - lo
    ids = np.stack([lo[:, 1] * gw + lo[:, 0], lo[:, 1] * gw + hi[:, 0], hi[:, 1] * gw + lo[:, 0], hi[:, 1] * gw + hi[:, 0]], 1)
    w = np.stack([(1 - f[:, 0]) * (1 - f[:, 1]), f[:, 0] * (1 - f[:, 1]), (1 - f[:, 0]) * f[:, 1], f[:, 0] * f[:, 1]], 1)
    return (ids, w)

def candidate_bank(cos, xy, spec, key):
    pos, weights = soft_labels(xy, spec)
    n = cos.shape[1]
    banks = []
    ys = []
    for a in range(len(cos)):
        positive = np.unique(pos[a])
        remaining = np.ones(n, bool)
        remaining[positive] = False
        order = np.argsort(-cos[a], kind='stable')
        hard = order[remaining[order]][:min(128, n - len(positive))]
        remaining[hard] = False
        rng = np.random.default_rng(seed('bank', key, a))
        uniform = rng.permutation(np.flatnonzero(remaining))[:min(256, n) - len(positive) - len(hard)]
        ix = np.r_[positive, hard, uniform]
        y = (ix[:, None] == pos[a][None]) @ weights[a]
        if len(ix) != min(256, n) or abs(y.sum() - 1) > 1e-08:
            raise ValueError('Candidate identity')
        banks.append(ix)
        ys.append(y)
    return (np.stack(banks), np.stack(ys))
