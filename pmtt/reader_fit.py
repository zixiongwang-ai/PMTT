"""Shared-scale convex reader fitting and validation-based L2 selection."""
import time
from pathlib import Path
import numpy as np
import torch
from .io import save, load
from .settings import SCALES, REPS, SCIENCE
from scipy.optimize import minimize
from .readers import design, standardize


def tables(base, task, scale, all_train=False):
    index_name = 'all_index.json' if all_train else 'index.json'
    return load(Path(base) / task / 'train' / scale / index_name)['paths']


def read_table(p):
    return torch.load(p, map_location='cpu', weights_only=False, mmap=True)


def blocks(base, task, scale, chunk=2048, cache=None, all_train=False):
    entries = cache[scale] if cache is not None else tables(base, task, scale, all_train)
    for p in entries:
        r = p if isinstance(p, dict) else read_table(p)
        for j in range(0, len(r['x']), chunk):
            weights = r.get('rw', torch.ones(len(r['x'])))
            yield (
                r['x'][j:j + chunk].cuda(),
                r['y'][j:j + chunk].cuda(),
                weights[j:j + chunk].cuda(),
            )


def stats(base, task, rep, scale, cache=None, all_train=False):
    count = 0
    total = None
    squares = None
    queries = 0
    samples = []
    for b, y, rw in blocks(base, task, scale, cache=cache, all_train=all_train):
        vals = b[..., 0].flatten()
        samples.append(vals[::max(1, len(vals) // 1024)].detach())
    values = torch.cat(samples)
    if len(values) > 2000000:
        values = values[::int(np.ceil(len(values) / 2000000))]
    quantiles = torch.tensor(
        [i / 18 for i in range(1, 18)],
        device='cuda',
        dtype=torch.float64,
    )
    knots = torch.quantile(values.double(), quantiles).cpu().tolist()
    del samples, values
    for b, y, rw in blocks(base, task, scale, cache=cache, all_train=all_train):
        x = design({'correct': b}, rep, knots).double().flatten(0, 1)
        t = x.sum(0)
        ss = x.square().sum(0)
        total = t if total is None else total + t
        squares = ss if squares is None else squares + ss
        count += len(x)
        queries += len(b)
    mean = total / count
    var = (squares / count - mean.square()).clamp_min(0)
    statistics = dict(
        mean=mean.cpu().tolist(),
        std=var.sqrt().clamp_min(1e-06).cpu().tolist(),
        count=count,
    )
    return dict(knots=knots, statistics=statistics, queries=queries)


def fit(base, task, all_train=False):
    directory = Path(base) / task / 'coefficients'
    out = {}
    cache = None
    for rep in REPS:
        suffix = '-refit-statistics.json' if all_train else '-statistics.json'
        sp = directory / (rep + suffix)
        st = (
            load(sp)
            if sp.exists()
            else {
                scale: stats(base, task, rep, scale, cache, all_train)
                for scale in SCALES
            }
        )
        save(st, sp)
        lambdas = (
            [load(directory / (rep + '.json'))['lambda_value']]
            if all_train
            else SCIENCE['l2']
        )
        for lam in lambdas:
            name = rep + ('-refit' if all_train else '-' + str(lam)) + '.json'
            p = directory / name
            if p.exists():
                out[rep + '/' + str(lam)] = load(p)
                if all_train:
                    save(load(p), directory / (rep + '.json'))
                continue
            trajectory = []
            started = time.time()
            d = len(st['S']['statistics']['mean'])

            def objective(ww):
                w = torch.tensor(ww, device='cuda', dtype=torch.float64)
                loss = torch.zeros((), device='cuda', dtype=torch.float64)
                gradient = torch.zeros_like(w)
                for s in SCALES:
                    for b, y, rw in blocks(base, task, s, cache=cache, all_train=all_train):
                        designed = design(
                            {'correct': b},
                            rep,
                            st[s]['knots'],
                        ).double()
                        x = standardize(designed, st[s]['statistics'])
                        y = y.double()
                        weights = rw.double() / (4 * st[s]['queries'])
                        logits = x @ w
                        logp = logits - torch.logsumexp(logits, -1, keepdim=True)
                        loss += (-(logp * y).sum(-1) * weights).sum()
                        gradient += torch.einsum(
                            'bkd,bk->d',
                            x,
                            (logp.exp() - y) * weights[:, None],
                        )
                loss += lam * w.square().sum()
                gradient += 2 * lam * w
                v = float(loss)
                g = gradient.cpu().numpy()
                if not np.isfinite(v) or not np.isfinite(g).all():
                    raise FloatingPointError('Non-finite fit')
                trajectory.append(v)
                return (v, g)
            options = dict(
                maxiter=600,
                gtol=2e-06,
                ftol=1e-11,
                maxls=40,
                maxcor=24,
            )
            result = minimize(
                objective,
                np.zeros(d),
                jac=True,
                method='L-BFGS-B',
                options=options,
            )
            _, g = objective(result.x)
            if not (result.success or abs(g).max() <= 4e-06):
                raise RuntimeError('Unconverged scalar fit ' + str(result.message))
            rec = dict(
                task=task,
                reader=rep,
                lambda_value=lam,
                coefficient=result.x.tolist(),
                scales=st,
                shared_across_scales=True,
                all_official_train_refit=all_train,
                capacity=d,
                seconds=time.time() - started,
                iterations=int(result.nit),
                converged=bool(result.success),
                gradient_inf=float(abs(g).max()),
                trajectory=trajectory,
            )
            save(rec, p)
            out[rep + '/' + str(lam)] = rec
            if all_train:
                save(rec, directory / (rep + '.json'))
    return dict(candidates=len(out))

def select(base, task, validation):
    result = {}
    for rep in REPS:
        vals = {
            lam: float(
                np.mean(
                    [validation[scale][rep + '/' + str(lam)] for scale in SCALES]
                )
            )
            for lam in SCIENCE['l2']
        }
        chosen = max(vals, key=lambda lam: (vals[lam], lam))
        rec = load(Path(base) / task / 'coefficients' / (rep + '-' + str(chosen) + '.json'))
        rec['validation_scores'] = {str(k): v for k, v in vals.items()}
        save(rec, Path(base) / task / 'coefficients' / (rep + '.json'))
        result[rep] = rec
    return result

def selected(base, task):
    return {rep: load(Path(base) / task / 'coefficients' / (rep + '.json')) for rep in REPS}

def candidates(base, task):
    result = {}
    for rep in REPS:
        for lam in SCIENCE['l2']:
            path = Path(base) / task / 'coefficients' / (
                rep + '-' + str(lam) + '.json'
            )
            result[rep + '/' + str(lam)] = {rep: load(path)}
    return result
