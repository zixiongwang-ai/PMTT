"""Visible-to-Hidden coverage, intrinsic laws, and diagnostic controls."""
import numpy as np
import torch
from .manifold import unit, reference, residuals, error, rotate_object, unbounded, basis, subspace_overlap

@torch.no_grad()
def evaluate(x, frozen_cap=None):
    """Evaluate Hidden states against Visible-only representations and diagnostic controls.
    
    x contains c[N,D], v[N,16,D] and h[N,16,D]. frozen_cap fixes the paper bound."""
    c, v, h = (x['c'], x['v'], x['h'])
    u = unit(c)
    d = h - u[:, None]
    rv = residuals(c, v)[1]
    a = reference(rv)['a']
    full = reference(residuals(c, torch.cat([v, h], 1))[1])['a']
    norm = a.norm(dim=(-2, -1))
    cap = 1.05 * norm.quantile(0.999) if frozen_cap is None else torch.as_tensor(frozen_cap, device=norm.device, dtype=norm.dtype)
    capped = norm > cap
    a = a * (cap / norm.clamp_min(1e-12)).clamp_max(1)[:, None, None]
    torch.manual_seed(20261024)
    random = rotate_object(dict(a=a), c)['a']
    similarity = u @ u.T
    similarity.fill_diagonal_(-torch.inf)
    donor_ids = similarity.argmax(-1)
    donor = a[donor_ids]
    direction = unit(c[donor_ids]) - u
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    donor = donor - 2 * direction[:, :, None] * (direction[:, None, :] @ donor)
    shared = torch.empty_like(a)
    n = len(c)
    half = n // 2
    if half == 0:
        raise ValueError('Shared controls require >=2 anchors')
    for target, source in ((slice(0, half), slice(half, n)), (slice(half, n), slice(0, half))):
        z = rv[source].reshape(-1, rv.shape[-1]).double()
        ev, q = torch.linalg.eigh(z.T @ z / max(len(z), 1))
        q = q[:, -8:].flip(-1)
        r = ev[-8:].flip(-1).clamp_min(0).sqrt()
        coeff = z @ q / r.clamp_min(1e-12)
        bank = (q * (r * coeff.norm(dim=-1).max().clamp_min(1))).float()
        shared[target] = bank[None] - u[target, :, None] * (u[target, None, :] @ bank[None])
    records = dict(
        point=d.square().sum(-1),
        factors=a,
        full_factors=full,
        donor_ids=donor_ids,
        cap_saturation=capped,
        cap=cap,
        radial=residuals(c, h)[2].square().sum(-1),
    )
    for name, f in (('manifold', a), ('full32', full), ('random', random), ('donor', donor), ('shared', shared)):
        err, z = error(f, d)
        records[name] = err
        records[name + '_boundary'] = z.norm(dim=-1) > 0.999
    records['unbounded'] = unbounded(a, d)
    records['full32_unbounded'] = unbounded(full, d)
    observed = torch.cat([u[:, None], v], 1)
    records['observed'] = torch.cdist(h, observed).square().amin(-1)
    _, s = basis(a)
    records['spectrum'] = s.square()
    records['energy'] = s.square().sum(-1)
    records['effective_rank'] = s.square().sum(-1).square() / s.pow(4).sum(-1).clamp_min(1e-12)
    records['normalized_energy'] = records['energy'] / records['point'].mean(-1).clamp_min(1e-12)
    records['overlap'] = torch.stack([subspace_overlap(a, full, r) for r in (1, 2, 4, 8)], -1)
    records['score_scale'] = rv.norm(dim=-1).median()
    records['visible_magnitude'] = rv.norm(dim=-1).mean(-1)
    records['hidden_magnitude'] = d.norm(dim=-1)
    records['normalized_spectrum'] = records['spectrum'] / records['energy'][:, None].clamp_min(1e-12)
    _, fs = basis(full)
    records['full32_spectrum'] = fs.square()
    records['full32_effective_rank'] = fs.square().sum(-1).square() / fs.pow(4).sum(-1).clamp_min(1e-12)
    for k, t in records.items():
        if not torch.isfinite(t).all():
            raise FloatingPointError(k)
    return {k: t.cpu() for k, t in records.items()}

def summary(r):
    """Aggregate intrinsic errors as pooled coverage and separate per-anchor statistics."""
    point = r['point'].double()
    if point.sum() <= 0:
        raise ValueError('Zero point denominator')
    result = {}
    for name in ('manifold', 'unbounded', 'full32', 'full32_unbounded', 'observed', 'random', 'donor', 'shared'):
        err = r[name].double()
        valid = point.sum(-1) > 1e-12
        per = 1 - err[valid].sum(-1) / point[valid].sum(-1)
        if not len(per):
            raise ValueError('No nondegenerate anchors')
        quantiles = torch.tensor(
            [0.05, 0.25, 0.5, 0.75, 0.95],
            dtype=torch.double,
        )
        result[name] = dict(
            pooled=float(1 - err.sum() / point.sum()),
            mean=float(per.mean()),
            quantiles=torch.quantile(per, quantiles).tolist(),
        )
    scalar_fields = (
        'energy',
        'normalized_energy',
        'effective_rank',
        'overlap',
        'spectrum',
        'cap_saturation',
        'manifold_boundary',
        'normalized_spectrum',
        'full32_spectrum',
        'full32_effective_rank',
    )
    for name in scalar_fields:
        result[name] = r[name].float().mean(0).tolist()
    for control in ('observed', 'random', 'donor', 'shared'):
        result['manifold-' + control] = dict(
            pooled=result['manifold']['pooled'] - result[control]['pooled'],
            improved_fraction=float(
                (
                    r['manifold'].sum(-1)
                    < r[control].sum(-1)
                ).float().mean()
            ),
        )
    result['zero_point_anchors'] = int((point.sum(-1) <= 1e-12).sum())
    result['radial_fraction'] = float(r['radial'].sum() / point.sum())
    return result

def paired(values, seed=931):
    a = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, 2000, 100):
        ix = rng.integers(len(a), size=(100, len(a)))
        draws.extend(a[ix].mean(1).tolist())
    return dict(
        mean=float(a.mean()),
        median=float(np.median(a)),
        positive_fraction=float((a > 0).mean()),
        interval=np.quantile(draws, [0.025, 0.975]).tolist(),
    )

def correspondence(source, target, hw=(14, 14), size=(224, 224)):
    h, w = hw
    ih, iw = size
    device = source.device
    source, target = (source.double(), target.double())
    y_coordinates = (
        torch.arange(h, device=device, dtype=torch.float64) + 0.5
    ) * 16
    x_coordinates = (
        torch.arange(w, device=device, dtype=torch.float64) + 0.5
    ) * 16
    yy, xx = torch.meshgrid(
        y_coordinates,
        x_coordinates,
        indexing='ij',
    )
    x = source[..., 0, None] + xx.flatten() / iw * source[..., 2, None]
    y = source[..., 1, None] + yy.flatten() / ih * source[..., 3, None]
    tx = (x - target[..., 0, None]) / target[..., 2, None] * iw
    ty = (y - target[..., 1, None]) / target[..., 3, None] * ih
    gx, gy = (tx / 16 - 0.5, ty / 16 - 0.5)
    tolerance = 1e-10
    valid = (gx >= -tolerance) & (gx <= w - 1 + tolerance) & (gy >= -tolerance) & (gy <= h - 1 + tolerance)
    grid = torch.stack((2 * (gx + 0.5) / w - 1, 2 * (gy + 0.5) / h - 1), -1).float()
    return (grid, valid)

@torch.no_grad()
def encode(backbone, row, chunk=32):
    """Encode one canonical view and 32 orbital views using the supplied canonical patch index."""
    crops = row['crops'].cuda()
    pos = row['positions'].cuda()
    states = []
    anchor = backbone(to_device_images(row['images'][:1]))
    center = anchor['c'][0, pos]
    for start in range(1, 33, chunk):
        r = backbone(to_device_images(row['images'][start:start + chunk]))
        cp = crops[start:start + len(r['c'])]
        grid, valid = correspondence(crops[:1].expand(len(cp), -1), cp)
        if not valid[:, pos].all():
            raise ValueError('Invalid continuous support')
        states.append(teacher_target(r['c'], grid[:, pos], r['hw']).transpose(0, 1))
    t = torch.cat(states, 1)
    return dict(c=center.cpu(), v=t[:, :16].cpu(), h=t[:, 16:].cpu())

from .observations import to_device_images
from .sampling import teacher_target
