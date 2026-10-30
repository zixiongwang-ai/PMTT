"""Deterministic FP64 multinomial fitting and feature standardization."""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from .settings import SCIENCE


def statistics(x, mask):
    vals = x[mask].double()
    std, mean = torch.std_mean(vals, dim=0, unbiased=False)
    return dict(
        mean=mean.cpu().tolist(),
        std=std.clamp_min(
            SCIENCE['standardization_eps']
        ).cpu().tolist(),
        raw_std=std.cpu().tolist(),
        min=vals.amin(0).cpu().tolist(),
        max=vals.amax(0).cpu().tolist(),
        count=len(vals),
    )


def optimize(x, y, mask, rw, lam, maxiter=None):
    x = x.double()
    y = y.to(x.device, dtype=x.dtype)
    mask = mask.to(x.device)
    rw = rw.to(x.device, dtype=x.dtype)
    rw = rw / rw.sum()
    flat = x.reshape(-1, x.shape[-1])
    trajectory = []
    started = time.time()

    def objective(w):
        ww = torch.as_tensor(w, device=x.device, dtype=x.dtype)
        scores = (flat @ ww).reshape(x.shape[:2]).masked_fill(~mask, -1e+30)
        logp = scores - torch.logsumexp(scores, -1, keepdim=True)
        loss = (-(logp * y).sum(-1) * rw).sum() + lam * (ww @ ww)
        delta = (logp.exp() - y) * rw[:, None]
        delta = delta.masked_fill(~mask, 0)
        gradient = flat.T @ delta.reshape(-1) + 2 * lam * ww
        value = float(loss)
        grad = gradient.cpu().numpy()
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError('Non-finite convex objective')
        trajectory.append(value)
        return (value, grad)
    options = dict(
        maxiter=maxiter or SCIENCE['solver_maxiter'],
        gtol=SCIENCE['solver_tolerance'],
        ftol=SCIENCE['solver_ftol'],
        maxls=40,
        maxcor=24,
    )
    result = minimize(
        objective,
        np.zeros(x.shape[-1]),
        jac=True,
        method='L-BFGS-B',
        options=options,
    )
    value, grad = objective(result.x)
    converged = bool(result.success or np.abs(grad).max() <= SCIENCE['solver_tolerance'] * 2)
    if not converged:
        raise RuntimeError(
            'Solver did not converge: '
            + str(result.message)
            + ' gradient='
            + str(np.abs(grad).max())
        )
    return dict(
        coefficient=result.x.tolist(),
        objective=value,
        trajectory=trajectory,
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        converged=converged,
        message=str(result.message),
        gradient_inf=float(np.abs(grad).max()),
        seconds=time.time() - started,
    )
