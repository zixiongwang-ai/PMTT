"""Centered tangent factors [..., D, rank] with rank at most eight."""
import torch
from torch.nn import functional as F
EPS=1e-12

def unit(x):
    """Normalize the last feature dimension in FP32 or FP64."""
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    return F.normalize(x, dim=-1, eps=EPS)

def tangent(u, v):
    return v - u.unsqueeze(-1) * (u.unsqueeze(-2) @ v)

def factor_energy(a):
    return a.square().sum((-2, -1))

def shape_energy(a):
    return (a.transpose(-2, -1) @ a).square().sum((-2, -1))

def overlap(a, b):
    return (a.transpose(-2, -1) @ b).square().sum((-2, -1))

def shape_distance(a, b):
    ea, eb = (shape_energy(a), shape_energy(b))
    num = ea + eb - 2 * overlap(a, b)
    tol = 32 * torch.finfo(a.dtype).eps * (ea + eb).clamp_min(EPS)
    if torch.any(num < -tol):
        raise FloatingPointError('Negative shape distance beyond roundoff')
    return num.clamp_min(0) / (ea + eb + EPS)

def compatibility(a, b):
    return overlap(a, b) / (torch.sqrt(shape_energy(a) * shape_energy(b)) + EPS)

def transport(a, b, v):
    a, b = (a.detach(), b.detach())
    dot = (a * b).sum(-1)
    valid = dot > -0.95
    denominator = torch.where(valid, 1 + dot, torch.ones_like(dot))
    out = v - (a + b).unsqueeze(-1) * (a.unsqueeze(-2) @ v) / denominator[..., None, None]
    return (out, valid)

@torch.no_grad()
def solve_ball(a, d):
    if a.dtype not in (torch.float32, torch.float64):
        raise TypeError('Geometry requires FP32/64')
    g = a.transpose(-2, -1) @ a
    b = (a.transpose(-2, -1) @ d.unsqueeze(-1)).squeeze(-1)
    ev, q = torch.linalg.eigh(g)
    ev = ev.clamp_min(0)
    bt = (q.transpose(-2, -1) @ b.unsqueeze(-1)).squeeze(-1)
    threshold = ev.amax(-1, keepdim=True) * torch.finfo(a.dtype).eps * 8
    active = ev > threshold
    inv = torch.where(active, 1 / ev.clamp_min(torch.finfo(a.dtype).tiny), torch.zeros_like(ev))
    z0 = bt * inv
    interior = z0.square().sum(-1) <= 1
    lo = torch.zeros_like(ev[..., 0])
    hi = bt.norm(dim=-1).clamp_min(torch.finfo(a.dtype).tiny)
    for _ in range(64 if a.dtype == torch.float64 else 40):
        mid = (lo + hi) / 2
        zn = (bt / (ev + mid.unsqueeze(-1)).clamp_min(torch.finfo(a.dtype).tiny)).norm(dim=-1)
        lo = torch.where(zn > 1, mid, lo)
        hi = torch.where(zn > 1, hi, mid)
    zb = bt / (ev + hi.unsqueeze(-1)).clamp_min(torch.finfo(a.dtype).tiny)
    z = (q @ torch.where(interior.unsqueeze(-1), z0, zb).unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(z).all():
        raise FloatingPointError('Non-finite constrained solution')
    return z

def coverage(u, a, t, s_ref=1.0):
    z = solve_ball(a, t.detach() - u)
    return (t.detach() - u - (a @ z.unsqueeze(-1)).squeeze(-1)).square().sum(-1) / s_ref ** 2

def select(c, a, q):
    h = (a.transpose(-2, -1) @ q.unsqueeze(-1)).squeeze(-1)
    z = h / torch.sqrt(1 + h.square().sum(-1, keepdim=True))
    return c + c.norm(dim=-1, keepdim=True) * (a @ z.unsqueeze(-1)).squeeze(-1)

def spectrum(a):
    e = torch.linalg.eigvalsh(a.transpose(-2, -1) @ a).clamp_min(0).flip(-1)
    total = e.sum(-1)
    valid = total > EPS
    nan = torch.full_like(total, float('nan'))
    return dict(
        energy=total,
        max_singular=e[..., 0].sqrt(),
        first=torch.where(valid, e[..., 0] / total, nan),
        top2=torch.where(valid, e[..., :2].sum(-1) / total, nan),
        rank=torch.where(
            valid,
            total.square() / e.square().sum(-1).clamp_min(EPS),
            nan,
        ),
        valid=valid,
    )

def random_geometry(u, a, generator=None):
    v = torch.randn(a.shape, device=a.device, dtype=a.dtype, generator=generator)
    q = torch.linalg.qr(tangent(u, v), mode='reduced').Q
    e = torch.linalg.eigvalsh(a.transpose(-2, -1) @ a).clamp_min(0).sqrt()
    return q * e.unsqueeze(-2)

def donor_geometry(u, a, donor):
    b = tangent(u, donor)
    norm = b.norm(dim=(-2, -1), keepdim=True)
    if torch.any((norm <= EPS) & (a.norm(dim=(-2, -1), keepdim=True) > EPS)):
        raise ValueError('Degenerate projected donor')
    return b * a.norm(dim=(-2, -1), keepdim=True) / norm.clamp_min(EPS)

def residuals(c, targets):
    """Return full, tangent and radial residuals for Center [...,D] and states [...,K,D]."""
    u = unit(c)
    d = unit(targets) - u.unsqueeze(-2)
    radial = (d * u.unsqueeze(-2)).sum(-1, keepdim=True) * u.unsqueeze(-2)
    return (d, d - radial, radial)

def basis(a):
    u, s, _ = torch.linalg.svd(a, full_matrices=False)
    valid = s > s.amax(-1, keepdim=True) * torch.finfo(s.dtype).eps * max(a.shape[-2:])
    return (u * valid.unsqueeze(-2), s)

def unbounded(a, d):
    q, _ = basis(a)
    projected = d @ q @ q.transpose(-1, -2)
    return (d - projected).square().sum(-1)

def subspace_overlap(a, b, rank=None):
    """Compute normalized, basis-invariant overlap, optionally restricted to a leading rank."""
    qa, _ = basis(a)
    qb, _ = basis(b)
    if rank is not None:
        qa = qa[..., :rank]
        qb = qb[..., :rank]
    denom = torch.minimum(qa.square().sum((-2, -1)), qb.square().sum((-2, -1))).clamp_min(1)
    return (qa.transpose(-1, -2) @ qb).square().sum((-2, -1)) / denom

def reference_svd(d, rank=8, drift=False):
    """Construct a bounded rank-r factor from [...,K,D] tangent residuals using direct SVD."""
    mu = d.mean(-2) if drift else torch.zeros_like(d[..., 0, :])
    x = d - mu.unsqueeze(-2)
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    r = min(rank, s.shape[-1])
    q = vh[..., :r, :].transpose(-1, -2)
    active = s[..., :r] > s[..., :1] * torch.finfo(s.dtype).eps * max(d.shape[-2:])
    q = q * active.unsqueeze(-2)
    radii = s[..., :r] / d.shape[-2] ** 0.5
    z = x @ q / radii.unsqueeze(-2).clamp_min(1e-12)
    scale = z.norm(dim=-1).amax(-1, keepdim=True).clamp_min(1.0)
    a = q * (radii * scale).unsqueeze(-2)
    if r < rank:
        a = torch.nn.functional.pad(a, (0, rank - r))
    return dict(a=a, drift=mu, spectrum=s.square() / d.shape[-2])


def reference(d, rank=8, drift=False):
    """Construct a centered factor [...,D,rank] from Visible tangent residuals.
    
    The FP64 K-by-K Gram construction falls back to direct SVD at the precision floor.
    The factor encloses the build projections; inference coefficients obey ||z||2 <= 1."""
    mu = d.mean(-2) if drift else torch.zeros_like(d[..., 0, :])
    x = d - mu.unsqueeze(-2)
    xd = x.double()
    eigen, u = torch.linalg.eigh(xd @ xd.transpose(-1, -2))
    dimension = min(x.shape[-2:])
    eigen = eigen.flip(-1)[..., :dimension].clamp_min(0)
    u = u.flip(-1)[..., :, :dimension]
    r = min(rank, dimension)
    ev = eigen[..., :r]
    ur = u[..., :, :r]
    floor = max((torch.finfo(d.dtype).eps * max(d.shape[-2:])) ** 2, torch.finfo(torch.float64).eps * x.shape[-2] * 8)
    active = ev > eigen[..., :1] * floor
    ur = ur * active.unsqueeze(-2)
    scale = (ur.square().sum(-1) * x.shape[-2]).sqrt().amax(-1, keepdim=True).clamp_min(1.0)
    a = xd.transpose(-1, -2) @ ur * (scale / x.shape[-2] ** 0.5).unsqueeze(-2)
    if r < rank:
        a = torch.nn.functional.pad(a, (0, rank - r))
    a = a.to(d.dtype)
    spectrum = (eigen / x.shape[-2]).to(d.dtype)
    uncertain = (ev[..., -1] <= eigen[..., 0] * torch.finfo(torch.float64).eps * x.shape[-2] * 128) & (eigen[..., 0] > 0)
    if uncertain.any():
        flat = d.reshape(-1, *d.shape[-2:])
        mask = uncertain.reshape(-1)
        direct = reference_svd(flat[mask], rank, drift)
        a = a.reshape(-1, d.shape[-1], rank)
        a[mask] = direct['a']
        a = a.reshape(*d.shape[:-2], d.shape[-1], rank)
        spectrum = spectrum.reshape(-1, dimension)
        spectrum[mask] = direct['spectrum']
        spectrum = spectrum.reshape(*d.shape[:-2], dimension)
    return dict(a=a, drift=mu, spectrum=spectrum)

def rotate_object(obj, c, donor=None):
    """Rotate complete geometry into a random tangent orientation while preserving its spectrum."""
    columns = obj['a']
    if 'drift' in obj and obj['drift'].norm(dim=-1).max() > 1e-10:
        columns = torch.cat([columns, obj['drift'].unsqueeze(-1)], -1)
    old = torch.linalg.qr(columns, mode='reduced').Q
    raw = torch.randn_like(old)
    if donor is not None:
        raw[..., :, :donor.shape[-1]] = donor
    u = unit(c)
    raw = raw - u.unsqueeze(-1) * (u.unsqueeze(-2) @ raw)
    target = torch.linalg.qr(raw, mode='reduced').Q
    result = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
    for j in range(old.shape[-1]):
        v = old[..., j] - target[..., j]
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        old = old - 2 * v.unsqueeze(-1) * (v.unsqueeze(-2) @ old)
        result['a'] = result['a'] - 2 * v.unsqueeze(-1) * (v.unsqueeze(-2) @ result['a'])
        if 'drift' in result:
            result['drift'] = result['drift'] - 2 * v * (v * result['drift']).sum(-1, keepdim=True)
        if 'vertices' in result:
            result['vertices'] = result['vertices'] - 2 * (result['vertices'] @ v.unsqueeze(-1)) * v.unsqueeze(-2)
    return result

@torch.no_grad()
def coefficients(a, d):
    """Return minimum-error coefficients [...,N,rank] subject to Euclidean norm at most one."""
    g = a.transpose(-1, -2) @ a
    ev, q = torch.linalg.eigh(g)
    ev = ev.clamp_min(0)
    b = d @ a @ q
    threshold = ev.amax(-1, keepdim=True) * torch.finfo(a.dtype).eps * 8
    inv = torch.where(ev > threshold, ev.clamp_min(1e-30).reciprocal(), 0)
    z0 = b * inv.unsqueeze(-2)
    inside = z0.square().sum(-1) <= 1
    lo = torch.zeros_like(b[..., 0])
    hi = b.norm(dim=-1).clamp_min(1e-30)
    for _ in range(64 if a.dtype == torch.float64 else 40):
        mid = (lo + hi) / 2
        outside = (b / (ev.unsqueeze(-2) + mid.unsqueeze(-1)).clamp_min(1e-30)).square().sum(-1) > 1
        lo = torch.where(outside, mid, lo)
        hi = torch.where(outside, hi, mid)
    z = torch.where(inside.unsqueeze(-1), z0, b / (ev.unsqueeze(-2) + hi.unsqueeze(-1)).clamp_min(1e-30))
    return z @ q.transpose(-1, -2)

def error(a, d):
    """Return squared residual errors and bounded coefficients for target displacements [...,N,D]."""
    z = coefficients(a.detach(), d.detach())
    return ((d - z @ a.transpose(-1, -2)).square().sum(-1), z)

@torch.no_grad()
def canonical(a):
    """Return basis, singular values, equal-eigenvalue groups and active directions of [N,D,8] factors."""
    g = a.double().transpose(-1, -2) @ a.double()
    e, v = torch.linalg.eigh(g)
    e = e.flip(-1).clamp_min(0)
    v = v.flip(-1)
    active = e > e.amax(-1, keepdim=True) * 1e-07
    s = e.sqrt()
    q = a.double() @ v / s.clamp_min(1e-15).unsqueeze(-2)
    q = q * active.unsqueeze(-2)
    groups = (e[:, :, None] - e[:, None, :]).abs() <= e.amax(-1)[:, None, None] * 1e-05
    groups = groups & active[:, :, None] & active[:, None, :]
    return (q.float(), s.float(), groups.float(), active.float())
