"""Representation-native scalar features with literal donor Visible replacement."""
import math
import torch
from .manifold import unit, canonical

def observed_stats(v, t, cos):
    """Return seven actual-Visible similarity statistics for source [B,16,D] and targets [B,N,D]."""
    sim = torch.einsum('bvd,bnd->bnv', v, t)
    top = sim.topk(4, dim=-1).values
    statistics = [
        top[..., 0],
        top[..., :2].mean(-1),
        top.mean(-1),
        sim.mean(-1),
        sim.std(-1, unbiased=False),
        (torch.logsumexp(10 * sim, -1) - math.log(16)) / 10,
        (sim > cos[..., None]).float().mean(-1),
    ]
    return torch.stack(statistics, -1)

def geometry_stats(u, t, shape, cos, v):
    """Return eight bounded-Manifold query features, a cross-feature callable, and squared error."""
    q, s, groups, active = shape
    dist = (2 - 2 * cos).clamp_min(0)
    proj = torch.einsum('bnd,bdr->bnr', t, q) - torch.einsum('bd,bdr->br', u, q)[:, None]
    energy = proj.square() * active[:, None]
    e = s.square()
    b = proj * s[:, None]
    z0 = b / e[:, None].clamp_min(1e-15)
    lo = torch.zeros_like(dist)
    hi = b.norm(dim=-1).clamp_min(1e-15)
    for _ in range(32):
        mid = (lo + hi) * 0.5
        outside = (b / (e[:, None] + mid[..., None]).clamp_min(1e-15)).square().sum(-1) > 1
        lo = torch.where(outside, mid, lo)
        hi = torch.where(outside, hi, mid)
    z = torch.where((z0.square().sum(-1) <= 1)[..., None], z0, b / (e[:, None] + hi[..., None]).clamp_min(1e-15))
    bounded = (dist - 2 * (z * b).sum(-1) + (z.square() * e[:, None]).sum(-1)).clamp_min(0)
    sub = (dist - energy.sum(-1)).clamp_min(0)
    frac = e / e.sum(-1, keepdim=True).clamp_min(1e-15)
    demand = z0.norm(dim=-1).clamp_max(10000)
    geometry_features = [
        bounded.log1p(),
        sub.log1p(),
        (dist - bounded) / dist.clamp_min(1e-06),
        z.norm(dim=-1),
        (demand - 1).clamp_min(0).log1p(),
        (energy * frac[:, None]).sum(-1) / dist.clamp_min(1e-06),
        sub / dist.clamp_min(1e-06),
        frac.square().sum(-1)[:, None].expand_as(cos),
    ]
    g = torch.stack(geometry_features, -1)
    zs = z * s[:, None]
    uq = torch.einsum('bd,bdr->br', u, q)
    state_norm = (1 + 2 * (zs * uq[:, None]).sum(-1) + zs.square().sum(-1)).clamp_min(1e-12).sqrt()

    def cross(vv, fs):
        uv = torch.einsum('bd,bvd->bv', u, vv)
        qv = torch.einsum('bdr,bvd->brv', q, vv)
        vs = (uv[:, None] + torch.einsum('bnr,brv->bnv', zs, qv)) / state_norm[..., None]
        return torch.stack([1 - bounded / 2 - fs[..., 0], (2 - 2 * vs.amax(-1)).clamp_min(0)], -1)
    return (g, cross, bounded)

@torch.no_grad()
def primitives(x, target, donor=None, xdestroyed=None, chunk=512):
    """Return native scalar banks [source,target,18], optionally with donor and destroyed controls."""
    u = unit(x['c'])
    n = len(u)
    shapes = canonical(torch.cat([x['manifold'], xdestroyed])) if xdestroyed is not None else canonical(x['manifold'])
    pieces = []
    for begin in range(0, len(target), chunk):
        raw = target[begin:begin + chunk]
        t = unit(raw)
        cos = (u @ t.T).clamp(-1, 1)
        ln = x['c'].norm(dim=-1).clamp_min(1e-12).log().clamp(-20, 20)[:, None].expand_as(cos)
        tn = raw.norm(dim=-1).clamp_min(1e-12).log().clamp(-20, 20)[None].expand_as(cos)
        vv = torch.cat([x['visible'], donor]) if donor is not None else x['visible']
        cc = torch.cat([cos, cos]) if donor is not None else cos
        fs = observed_stats(vv, t[None].expand(len(vv), -1, -1), cc)
        uu = torch.cat([u, u]) if xdestroyed is not None else u
        cc = torch.cat([cos, cos]) if xdestroyed is not None else cos
        gs = geometry_stats(uu, t[None].expand(len(uu), -1, -1), shapes, cc, x['visible'])[0]
        result = dict(correct=torch.cat([torch.stack([cos, ln, tn], -1), fs[:n], gs[:n]], -1))
        if donor is not None:
            result['donor'] = fs[n:]
        if xdestroyed is not None:
            result['destroyed'] = gs[n:]
        if any((not torch.isfinite(v).all() for v in result.values())):
            raise FloatingPointError('Native scalar nonfinite')
        pieces.append(result)
    return {k: torch.cat([r[k] for r in pieces], 1) for k in pieces[0]}
