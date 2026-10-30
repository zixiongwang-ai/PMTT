"""Minimum-Sufficient Representation-Native Readers."""
import numpy as np
import torch
from .features import primitives
from .assets import load_reader_parameters
KNOTS=np.linspace(-.8,.8,17).tolist()

def design(bank, reader, knots=None):
    """Select the Center, Observed or Manifold feature vector in the fitted parameter order."""
    b = bank['correct']
    cos, ln, tn = (b[..., 0], b[..., 1], b[..., 2])
    if reader == 'Cosine':
        return cos[..., None]
    if reader in ('Observed', 'DonorObserved'):
        return torch.cat([cos[..., None], b[..., 3:10] if reader == 'Observed' else bank['donor']], -1)
    if reader in ('Manifold', 'DestroyedManifold'):
        return torch.cat([cos[..., None], b[..., 10:18] if reader == 'Manifold' else bank['destroyed']], -1)
    if reader == 'Center':
        point = torch.stack([cos, cos.square(), cos.pow(3), cos * (ln + tn).exp(), ln, tn, (ln - tn).abs()], -1)
        knots = torch.as_tensor(knots, device=b.device, dtype=b.dtype)
        return torch.cat([point, (cos[..., None] - knots).clamp_min(0)], -1)
    raise ValueError(reader)

def standardize(x, stats):
    return (x - torch.tensor(stats['mean'], device=x.device, dtype=x.dtype)) / torch.tensor(stats['std'], device=x.device, dtype=x.dtype)

def score(bank, condition, records, scale):
    """Apply the representation-specific frozen reader after scale-specific standardization."""
    rep = {'DonorObserved': 'Observed', 'DestroyedManifold': 'Manifold'}.get(condition, condition)
    rec = records[rep]
    st = rec['scales'][scale]
    x = design(bank, condition, st['knots']).double()
    return standardize(x, st['statistics']) @ torch.tensor(rec['coefficient'], device=x.device, dtype=torch.float64)
