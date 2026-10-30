"""Invertible train-only conditioning of the raw Center cross-entropy objective."""
import time
from .io import save
import numpy as np
import torch
from scipy.optimize import minimize


def raw_objective(x, target, flat):
    N, D = x.shape
    K = 150
    w = torch.tensor(np.asarray(flat).reshape(D + 1, K), device=x.device, dtype=torch.float32)
    loss = torch.zeros((), device=x.device, dtype=torch.float64)
    grad = torch.zeros((D + 1, K), device=x.device, dtype=torch.float64)
    for i in range(0, N, 8192):
        a = x[i:i + 8192]
        z = a @ w[:-1] + w[-1]
        labs = target[i:i + len(a)]
        lp = z - z.logsumexp(-1, keepdim=True)
        cost = -lp[torch.arange(len(a), device=x.device), labs]
        delta = lp.exp()
        delta[torch.arange(len(a), device=x.device), labs] -= 1
        loss += cost.double().sum() / N
        grad[:-1] += (a.T @ delta).double() / N
        grad[-1] += delta.double().sum(0) / N
    loss += 0.0001 * w[:-1].double().square().sum()
    grad[:-1] += 0.0002 * w[:-1].double()
    v = float(loss)
    g = grad.cpu().numpy().ravel()
    if not np.isfinite(v) or not np.isfinite(g).all():
        raise FloatingPointError('Axis fit')
    return (v, g)


def precondition(x):
    N, D = x.shape
    mean = torch.zeros(D, device=x.device, dtype=torch.float64)
    second = torch.zeros((D, D), device=x.device, dtype=torch.float64)
    for i in range(0, N, 8192):
        a = x[i:i + 8192].double()
        mean += a.sum(0) / N
        second += a.T @ a / N
    covariance = second - mean[:, None] * mean[None, :]
    eigen, q = torch.linalg.eigh(covariance)
    r = q * (eigen.clamp_min(0) + 0.03).rsqrt()[None, :] @ q.T
    p = torch.zeros((D + 1, D + 1), device=x.device, dtype=torch.float64)
    p[:-1, :-1] = r
    p[-1, :-1] = -mean @ r
    p[-1, -1] = 1
    transfer_bound = float(torch.linalg.inv(p).T.abs().sum(1).max())
    coordinate_scale = max(1.0, transfer_bound * 1.01)
    p *= coordinate_scale
    diagnostics = dict(
        raw_gradient_bound=transfer_bound / coordinate_scale,
        coordinate_scale=coordinate_scale,
        covariance_min=float(eigen.min()),
        covariance_max=float(eigen.max()),
        conditioner_ridge=0.03,
        mean_norm=float(mean.norm()),
        raw_regularization_unchanged=True,
    )
    return (p.cpu().numpy(), diagnostics)


def fit(x, y, ds, diagnostic=None):
    if ds != 'ade':
        raise ValueError('ADE20K classifier required')
    x = x.cuda()
    target = y.cuda().float().long()
    N, D = x.shape
    valid_target = ((target >= 0) & (target < 150)).all()
    if not len(x) or not torch.isfinite(x).all() or not valid_target:
        raise ValueError('Invalid ADE axis inputs')
    start = time.time()
    p, conditioning = precondition(x)
    trajectory = []
    raw_gradient = []
    last = {}

    def objective(u):
        raw = p @ u.reshape(D + 1, 150)
        value, g = raw_objective(x, target, raw.ravel())
        raw_gradient.append(float(abs(g).max()))
        trajectory.append(value)
        last.update(raw=raw, value=value, gradient=raw_gradient[-1])
        if len(trajectory) % 25 == 0:
            print(
                'ADE_AXIS_SOLVER',
                len(trajectory),
                value,
                raw_gradient[-1],
                round(time.time() - start, 2),
                flush=True,
            )
        return (value, (p.T @ g.reshape(D + 1, 150)).ravel())
    options = dict(maxiter=500, ftol=1e-09, gtol=1e-05, maxls=40, maxcor=12)
    result = minimize(
        objective,
        np.zeros((D + 1) * 150),
        jac=True,
        method='L-BFGS-B',
        options=options,
    )
    objective(result.x)
    raw = last['raw']
    gradient = last['gradient']
    accepted = bool(result.success or gradient < 2e-05)
    record = dict(
        trajectory=trajectory,
        converged=bool(result.success),
        gradient_inf=gradient,
        iterations=int(result.nit),
        examples=N,
        parameters=(D + 1) * 150,
        inputs='raw Center only; shared Center task axis',
        l2=0.0001,
        solver='L-BFGS-B with invertible train-only parameter conditioning',
        message=str(result.message),
        conditioning=conditioning,
        maxiter=500,
        ftol=1e-09,
        gtol=1e-05,
        zero_initialization=True,
        raw_gradient_trajectory=raw_gradient,
        seconds=time.time() - start,
        accepted=accepted,
    )
    if diagnostic:
        save(record, diagnostic)
    if not accepted:
        raise RuntimeError('Center axis unconverged ' + str(result.message))
    return dict(
        w=torch.tensor(raw[:-1], dtype=torch.float32),
        b=torch.tensor(raw[-1], dtype=torch.float32),
        fit=record,
    )
