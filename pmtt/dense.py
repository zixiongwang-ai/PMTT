"""Shared Center task axes and scalar Observed and Manifold mixtures."""

import math
import time

import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn import functional as TF
from .manifold import canonical
from .io import ensure_finite

def valid_depth(y):
    v = np.isfinite(y) & (y > 0.001) & (y <= 10)
    v[:45] = False
    v[471:] = False
    v[:, :41] = False
    v[:, 601:] = False
    return v

def grid(size):
    w, h = size
    xx, yy = np.meshgrid(
        (np.arange(14) + 0.5) / 14 * w,
        (np.arange(14) + 0.5) / 14 * h,
    )
    return np.stack([xx.ravel(), yy.ravel()], 1)

def points(image, label, row, ds, calibration=False):
    w, h = image.size
    if ds == 'nyu':
        ij = np.argwhere(valid_depth(label))
        if not len(ij):
            raise ValueError('No legal training depth ' + row['id'])
        values = label[ij[:, 0], ij[:, 1]]
        order = np.argsort(values, kind='stable')
        quantiles = np.array([0.125, 0.375, 0.625, 0.875])
        indices = (quantiles * len(order)).astype(int)
        ix = order[np.minimum(indices, len(order) - 1)]
        xy = ij[ix, ::-1] + 0.5
    else:
        ids = np.linspace(0, 13, 2 if calibration else 8).round().astype(int)
        xx, yy = np.meshgrid(ids, ids)
        xy = grid(image.size)[(yy * 14 + xx).ravel()]
    labels = label[
        np.minimum(xy[:, 1].astype(int), h - 1),
        np.minimum(xy[:, 0].astype(int), w - 1),
    ]
    if ds == 'nyu':
        valid = (labels > 0.001) & (labels <= 10)
    else:
        valid = (labels >= 0) & (labels < 150)
    return (xy[valid], labels[valid])

def center_axis_fit(x, y, ds):
    """Fit a raw-Center linear log-depth or class axis using the fixed training objective."""
    x = x.cuda()
    y = y.cuda().float()
    N, D = x.shape
    K = 1 if ds == 'nyu' else 150
    target = y.log() if ds == 'nyu' else y.long()
    trajectory = []

    def objective(flat):
        w = torch.tensor(flat.reshape(D + 1, K), device='cuda', dtype=torch.float32)
        loss = torch.zeros((), device='cuda', dtype=torch.float64)
        grad = torch.zeros((D + 1, K), device='cuda', dtype=torch.float64)
        for i in range(0, N, 8192):
            a = x[i:i + 8192]
            z = a @ w[:-1] + w[-1]
            if ds == 'nyu':
                err = z[:, 0] - target[i:i + len(a)]
                delta = err.clamp(-1, 1)[:, None]
                cost = torch.where(err.abs() <= 1, 0.5 * err.square(), err.abs() - 0.5)
            else:
                labs = target[i:i + len(a)]
                lp = z - z.logsumexp(-1, keepdim=True)
                cost = -lp[torch.arange(len(a), device='cuda'), labs]
                delta = lp.exp()
                delta[torch.arange(len(a), device='cuda'), labs] -= 1
            loss += cost.double().sum() / N
            grad[:-1] += (a.T @ delta).double() / N
            grad[-1] += delta.double().sum(0) / N
        loss += 0.0001 * w[:-1].double().square().sum()
        grad[:-1] += 0.0002 * w[:-1].double()
        v = float(loss)
        g = grad.cpu().numpy().ravel()
        if not np.isfinite(v) or not np.isfinite(g).all():
            raise FloatingPointError('Axis fit')
        trajectory.append(v)
        return (v, g)
    options = dict(maxiter=500, ftol=1e-09, gtol=1e-05, maxls=40, maxcor=12)
    result = minimize(
        objective,
        np.zeros((D + 1) * K),
        jac=True,
        method='L-BFGS-B',
        options=options,
    )
    _, gradient = objective(result.x)
    if not (result.success or abs(gradient).max() < 2e-05):
        raise RuntimeError('Center axis unconverged ' + str(result.message))
    parameters = result.x.reshape(D + 1, K)
    fit = dict(
        trajectory=trajectory,
        converged=bool(result.success),
        gradient_inf=float(abs(gradient).max()),
        iterations=int(result.nit),
        examples=N,
        parameters=(D + 1) * K,
        inputs='raw Center only; shared Center task axis',
        l2=0.0001,
    )
    return dict(
        w=torch.tensor(parameters[:-1], dtype=torch.float32),
        b=torch.tensor(parameters[-1], dtype=torch.float32),
        fit=fit,
    )

def scalar_features(x, axis, ds, donor=None, destroyed=None):
    """Apply the shared Center task axis and form per-class, per-patch scalar reader features."""
    w = axis['w'].to(x['c'].device)
    bias = axis['b'].to(x['c'].device)
    c = x['c'] @ w + bias
    v = (x['visible'] if donor is None else donor) @ w + bias
    a = x['manifold'] if destroyed is None else destroyed
    radius = torch.einsum('ndr,dk->nkr', a, w).norm(dim=-1)
    s = canonical(a)[1].square()
    energy = s.sum(-1)
    fraction = s / energy[:, None].clamp_min(1e-12)
    conc = fraction.square().sum(-1)
    rank = 1 / conc.clamp_min(1e-12)
    dom = fraction.max(-1).values

    def repeat(q):
        return q[:, None].expand_as(c)

    if ds == 'nyu':
        sortedv = v.sort(dim=1).values
        observed = [
            c,
            v.mean(1),
            v.median(1).values,
            v.min(1).values,
            v.max(1).values,
            v.std(1, unbiased=False),
            sortedv[:, 2:-2].mean(1),
            v.max(1).values - v.min(1).values,
        ]
        manifold = [
            c,
            radius,
            radius / w.norm(dim=0).clamp_min(1e-12),
            repeat(energy),
            repeat(conc),
            repeat(rank),
            repeat(dom),
            repeat(s.sqrt().max(-1).values),
        ]
    else:
        observed = [
            c,
            v.mean(1),
            v.max(1).values,
            v.topk(2, dim=1).values.mean(1),
            torch.logsumexp(v, dim=1) - math.log(16),
            v.std(1, unbiased=False),
        ]
        manifold = [
            c,
            c + radius,
            c - radius,
            radius / w.norm(dim=0).clamp_min(1e-12),
            repeat(conc),
            repeat(rank),
            repeat(energy),
        ]
    f = torch.stack(observed, -1)
    g = torch.stack(manifold, -1)
    return dict(Center=c, Observed=ensure_finite(f), Manifold=ensure_finite(g))

def mix(x, record, scale):
    st = record['statistics'][scale]
    mean = torch.tensor(st['mean'], device=x.device)
    std = torch.tensor(st['std'], device=x.device)
    coefficient = torch.tensor(
        record['coefficient'],
        device=x.device,
        dtype=torch.float64,
    )
    normalized = (x.double() - mean) / std
    return normalized @ coefficient + record.get('intercept', 0.0)

def confusion(pred, label, mask=None):
    """Return additive 150-by-150 counts with GT rows; labels outside 0..149 are ignored."""
    valid = (label >= 0) & (label < 150)
    if mask is not None:
        valid &= mask
    indices = (label[valid] * 150 + pred[valid]).astype(int)
    return np.bincount(indices, minlength=22500).reshape(150, 150)

def seg_metrics(cm):
    cm = np.asarray(cm, float)
    den = cm.sum(0) + cm.sum(1) - cm.diagonal()
    valid = den > 0
    row = cm.sum(1)
    present = row > 0
    diagonal = cm.diagonal()
    return dict(
        miou=float(np.mean(diagonal[valid] / den[valid])) if valid.any() else None,
        pixel_accuracy=float(cm.trace() / cm.sum()) if cm.sum() else None,
        mean_class_accuracy=(
            float(np.mean(diagonal[present] / row[present]))
            if present.any()
            else None
        ),
        per_class=[
            float(cm[i, i] / den[i]) if den[i] else None
            for i in range(150)
        ],
    )

def depth_metrics(pred, truth, mask):
    """Evaluate positive meter-valued predictions against truth on an explicit boolean mask."""
    p = pred[mask]
    y = truth[mask]
    if not len(y):
        return dict(count=0, AbsRel=None)
    ratio = np.maximum(p / y, y / p)
    squared_error = (p - y) ** 2
    log_squared_error = (np.log(p) - np.log(y)) ** 2
    return dict(
        count=len(y),
        AbsRel=float(np.mean(abs(p - y) / y)),
        SqRel=float(np.mean(squared_error / y)),
        RMSE=float(np.mean(squared_error) ** 0.5),
        log_RMSE=float(np.mean(log_squared_error) ** 0.5),
        delta1=float(np.mean(ratio < 1.25)),
        delta2=float(np.mean(ratio < 1.25 ** 2)),
        delta3=float(np.mean(ratio < 1.25 ** 3)),
    )

def upsample_argmax(z, size, chunk=16):
    """Upsample each class plane from 14-by-14 before argmax; ties use the lower class index."""
    best = None
    pred = None
    for first in range(0, z.shape[1], chunk):
        native = z[:, first:first + chunk].T.reshape(1, -1, 14, 14)
        block = TF.interpolate(
            native,
            size=size,
            mode='bilinear',
            align_corners=False,
        )[0]
        value, ix = block.max(0)
        ix = ix + first
        if best is None:
            best = value
            pred = ix
        else:
            better = value > best
            pred = torch.where(better, ix, pred)
            best = torch.maximum(best, value)
    return pred

def control_features(x, axis, ds, rep, donor=None, destroyed=None):
    w = axis['w'].to(x['c'].device)
    bias = axis['b'].to(x['c'].device)
    c = x['c'] @ w + bias
    if rep == 'Observed':
        v = (x['visible'] if donor is None else donor) @ w + bias
        if ds == 'nyu':
            sortedv = v.sort(dim=1).values
            values = [
                c,
                v.mean(1),
                v.median(1).values,
                v.min(1).values,
                v.max(1).values,
                v.std(1, unbiased=False),
                sortedv[:, 2:-2].mean(1),
                v.max(1).values - v.min(1).values,
            ]
        else:
            values = [
                c,
                v.mean(1),
                v.max(1).values,
                v.topk(2, dim=1).values.mean(1),
                torch.logsumexp(v, dim=1) - math.log(16),
                v.std(1, unbiased=False),
            ]
        f = torch.stack(values, -1)
        return ensure_finite(f)
    if rep != 'Manifold':
        raise ValueError(rep)
    a = x['manifold'] if destroyed is None else destroyed
    radius = torch.einsum('ndr,dk->nkr', a, w).norm(dim=-1)
    s = canonical(a)[1].square()
    energy = s.sum(-1)
    fraction = s / energy[:, None].clamp_min(1e-12)
    conc = fraction.square().sum(-1)
    rank = 1 / conc.clamp_min(1e-12)
    dom = fraction.max(-1).values

    def repeat(q):
        return q[:, None].expand_as(c)

    if ds == 'nyu':
        values = [
            c,
            radius,
            radius / w.norm(dim=0).clamp_min(1e-12),
            repeat(energy),
            repeat(conc),
            repeat(rank),
            repeat(dom),
            repeat(s.sqrt().max(-1).values),
        ]
    else:
        values = [
            c,
            c + radius,
            c - radius,
            radius / w.norm(dim=0).clamp_min(1e-12),
            repeat(conc),
            repeat(rank),
            repeat(energy),
        ]
    g = torch.stack(values, -1)
    return ensure_finite(g)
