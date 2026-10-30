"""Center-only axis fitting and class-independent scalar mixture selection."""

import gc
import math

from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from .io import identity, save, load
from .settings import SCALES, ARCH, SCIENCE
from .backbone import Backbone
from .representation import anchors, structured, gpu
from .datasets.preparation import prepared_manifest
from .dense import points, scalar_features, center_axis_fit, mix, confusion, seg_metrics

def axis(base, ds, scale):
    m = manifest(base, ds)
    directory = Path(base) / ds / 'axis' / scale
    final = directory / 'axis.pt'
    if final.exists():
        return load(final)['fit']
    bb = Backbone(scale)
    xs = []
    ys = []
    counts = []
    for j, row in enumerate(m['splits']['train']):
        p = directory / 'features' / (identity(row['id']) + '.pt')
        if not p.exists():
            image, label = image_label(row, ds)
            xy, y = points(image, label, row, ds)
            aa = anchors(image, xy, [(ds, row['id'], 'axis', i) for i in range(len(xy))])
            c = structured(bb, image, aa, 17, 4, centers_only=True) if len(aa) else torch.empty((0, ARCH[scale][1]))
            class_counts = (
                np.bincount(label[label >= 0], minlength=150).tolist()
                if ds == 'ade'
                else None
            )
            save(
                dict(
                    c=c,
                    y=torch.tensor(y),
                    id=row['id'],
                    xy=xy.tolist(),
                    class_counts=class_counts,
                ),
                p,
            )
        r = load(p)
        xs.append(r['c'])
        ys.append(r['y'])
        counts.append(dict(id=row['id'], count=len(r['c'])))
        if j % 512 == 0:
            print('AXIS EXTRACT', ds, scale, j, flush=True)
    del bb
    gc.collect()
    torch.cuda.empty_cache()
    try:
        fit = center_axis_fit(torch.cat(xs), torch.cat(ys), ds)
    except RuntimeError:
        if ds != 'ade':
            raise
        from .axis_fit import fit as conditioned_fit
        fit = conditioned_fit(torch.cat(xs), torch.cat(ys), ds)
    fit['images'] = counts
    if ds == 'ade':
        class_counts = [
            load(directory / 'features' / (identity(row['id']) + '.pt'))[
                'class_counts'
            ]
            for row in m['splits']['train']
        ]
        fit['class_frequency'] = np.sum(class_counts, axis=0).tolist()
    save(fit, final)
    return fit['fit']

def calibrate(base, ds, scale):
    final = Path(base) / ds / 'calibration' / scale / 'index.json'
    if final.exists():
        return load(final)
    bb = Backbone(scale)
    ax = load(Path(base) / ds / 'axis' / scale / 'axis.pt')
    paths = []
    for j, row in enumerate(manifest(base, ds)['splits']['train']):
        split = row['calibration_split']
        if split is None:
            continue
        p = Path(base) / ds / 'calibration' / scale / (identity(row['id']) + '.pt')
        if not p.exists():
            image, label = image_label(row, ds)
            xy, y = points(image, label, row, ds, True)
            if not len(xy):
                save(dict(id=row['id'], split=split, invalid='all_ignore_patches', count=0), p)
                paths.append(str(p))
                continue
            aa = anchors(image, xy, [(ds, row['id'], 'calibration', i) for i in range(len(xy))])
            x = gpu(structured(bb, image, aa, 17, 4))
            features = scalar_features(x, ax, ds)
            dp = Path(base) / ds / 'donors' / scale / (identity(row['id']) + '.pt')
            donor_directory = Path(base) / ds / 'donors' / scale
            donor_count = (
                len(list(donor_directory.glob('*.pt')))
                if donor_directory.exists()
                else 0
            )
            if donor_count < 2:
                save(dict(id=row['id'], x={k: v.cpu() for k, v in x.items()}), dp)
            save(dict(id=row['id'], split=split, count=len(y), features={k: v.cpu() for k, v in features.items()}, y=torch.tensor(y)), p)
        paths.append(str(p))
    save(dict(paths=paths), final)
    del bb
    gc.collect()
    torch.cuda.empty_cache()
    return dict(count=len(paths))

def calibration_data(base, ds, scale, split):
    rs = [load(p) for p in load(Path(base) / ds / 'calibration' / scale / 'index.json')['paths']]
    rs = [r for r in rs if r['split'] == split and r['count']]
    if not rs:
        raise ValueError('Empty dense calibration ' + split)
    features = {
        key: torch.cat([record['features'][key] for record in rs])
        for key in ('Center', 'Observed', 'Manifold')
    }
    labels = torch.cat([record['y'] for record in rs])
    return (features, labels)

def mixture_fit(base, ds):
    from .readers import standardize
    records = {}
    for rep in ('Observed', 'Manifold'):
        st = {}
        xx = []
        yy = []
        weights = []
        for s in SCALES:
            f, y = calibration_data(base, ds, s, 'fit')
            x = f[rep].double().cuda()
            flat = x.reshape(-1, x.shape[-1])
            mean = flat.mean(0)
            std = flat.std(0, unbiased=False).clamp_min(1e-06)
            st[s] = dict(mean=mean.cpu().tolist(), std=std.cpu().tolist())
            xx.append((x - mean) / std)
            yy.append(y.cuda())
            weights.append(torch.full((len(y),), 1 / (4 * len(y)), device='cuda', dtype=torch.float64))
        x = torch.cat(xx)
        y = torch.cat(yy)
        rw = torch.cat(weights)
        d = x.shape[-1] + int(ds == 'nyu')
        fits = []
        for lam in SCIENCE['l2']:

            def objective(ww):
                w = torch.tensor(ww, device='cuda', dtype=torch.float64)
                z = x @ w[:-1] + w[-1] if ds == 'nyu' else x @ w
                if ds == 'nyu':
                    err = z[:, 0] - y.log()
                    cost = err.square()
                    weighted_error = 2 * err * rw
                    gradient = torch.cat(
                        [
                            torch.einsum('nd,n->d', x[:, 0], weighted_error),
                            weighted_error.sum()[None],
                        ]
                    )
                else:
                    lp = z - z.logsumexp(-1, keepdim=True)
                    labs = y.long()
                    cost = -lp[torch.arange(len(y), device='cuda'), labs]
                    delta = lp.exp()
                    delta[torch.arange(len(y), device='cuda'), labs] -= 1
                    gradient = torch.einsum('nkd,nk->d', x, delta * rw[:, None])
                loss = (cost * rw).sum() + lam * w.square().sum()
                gradient += 2 * lam * w
                return (float(loss), gradient.cpu().numpy())
            result = minimize(
                objective,
                np.zeros(d),
                jac=True,
                method='L-BFGS-B',
                options=dict(maxiter=600, gtol=2e-06, ftol=1e-11),
            )
            _, grad = objective(result.x)
            if not (result.success or abs(grad).max() < 4e-06):
                raise RuntimeError('Dense mixture unconverged')
            rec = dict(
                coefficient=(
                    result.x[:-1].tolist()
                    if ds == 'nyu'
                    else result.x.tolist()
                ),
                intercept=float(result.x[-1]) if ds == 'nyu' else 0.0,
                statistics=st,
                lambda_value=lam,
                shared_scales=list(SCALES),
                class_independent=True,
                parameters=d,
                train_split='train calibration fit',
                gradient_inf=float(abs(grad).max()),
            )
            scores = []
            for s in SCALES:
                f, vy = calibration_data(base, ds, s, 'validation')
                z = mix(f[rep].cuda(), rec, s)
                if ds == 'nyu':
                    prediction = z[:, 0].clamp(
                        math.log(0.001),
                        math.log(10),
                    ).exp()
                    truth = vy.cuda()
                    scores.append(
                        -float(((prediction - truth).abs() / truth).mean())
                    )
                else:
                    cm = confusion(z.argmax(-1).cpu().numpy(), vy.numpy())
                    scores.append(seg_metrics(cm)['miou'])
            rec['validation'] = scores
            fits.append(rec)
        chosen = max(fits, key=lambda r: (np.mean(r['validation']), r['lambda_value']))
        records[rep] = chosen
        save(dict(selected=chosen, candidates=fits), Path(base) / ds / 'mixtures' / (rep + '.json'))
        del x, y, xx, yy
        gc.collect()
        torch.cuda.empty_cache()
    return dict(readers=list(records))

def manifest(base, ds):
    return prepared_manifest(base, ds)

def image_label(row, ds):
    from .datasets.dense import image_label as read
    return read(row, ds)
