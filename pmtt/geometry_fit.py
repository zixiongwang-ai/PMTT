"""B/L reader selection, 128-pair refit and held-out pose calibration."""
import time
from pathlib import Path
import numpy as np
import torch
from .io import load, save
from .readers import design
from .convex_fit import optimize, statistics
from .readers import standardize
from .evaluation.geometry import assignment
from .datasets.preparation import prepared_manifest

SCALES = ('B', 'L')
SCIENCE = dict(
    hinge_quantiles=[i / 18 for i in range(1, 18)],
    lambda_grid=[0.0001, 0.001, 0.01],
    pose_thresholds=[0.5, 1.0, 2.0, 4.0],
)


def manifest(base, ds):
    return prepared_manifest(base, ds)


def scalarpath(base, ds, scale, pair):
    return Path(base) / ds / 'calibration' / scale / (pair + '.pt')


def training(base, ds, scale, splits):
    pieces = []
    weights = []
    rows = [r for r in manifest(base, ds)['rows'] if r['split'] in splits]
    for row in rows:
        b = load(scalarpath(base, ds, scale, row['id']))['bank']['correct']
        cr = load(Path(base) / ds / 'candidates' / (row['id'] + '.pt'))
        valid = torch.tensor(row['gt']['valid'])
        indices = cr['indices']
        selected = torch.gather(
            b,
            1,
            indices[..., None].expand(-1, -1, b.shape[-1]),
        )
        pieces.append(selected[valid])
        weights.append(cr['weights'][valid])
    return ({'correct': torch.cat(pieces)}, torch.cat(weights), [r['id'] for r in rows])

def score(bank, condition, record, scale):
    if condition == 'Cosine':
        return bank['correct'][..., 0].numpy().astype(np.float64)
    reader = {'DonorObserved': 'Observed', 'DestroyedManifold': 'Manifold'}.get(condition, condition)
    rec = record[reader]
    st = rec['scales'][scale]
    x = design(bank, condition, st['knots']).numpy().astype(np.float64)
    normalized = (
        x - np.asarray(st['statistics']['mean'])
    ) / np.asarray(st['statistics']['std'])
    return normalized @ np.asarray(rec['coefficient'])

def validation(base, ds, reader, rec):
    result = []
    for scale in SCALES:
        correct = count = 0
        for row in manifest(base, ds)['rows']:
            if row['split'] != 'validation':
                continue
            r = load(scalarpath(base, ds, scale, row['id']))
            scores = score(r['bank'], reader, {reader: rec}, scale)
            m = assignment(scores, r['spec'])
            valid = np.asarray(row['gt']['valid'])[m['source']]
            truth = np.asarray(row['gt']['truth'])[m['source']] * r['spec']['scale']
            err = np.linalg.norm(m['refined'] - truth, axis=1)
            correct += int(((err <= 10) & valid).sum())
            count += int(valid.sum())
        result.append(correct / max(count, 1))
    return float(np.mean(result))

def prepare_training(base, ds, splits, reader):
    xx = []
    yy = []
    rw = []
    scales = {}
    ids = None
    for scale in SCALES:
        bank, y, ids = training(base, ds, scale, splits)
        knots = torch.quantile(
            bank['correct'][..., 0].double().flatten(),
            torch.tensor(SCIENCE['hinge_quantiles'], dtype=torch.float64),
        ).tolist()
        x = design(bank, reader, knots).double()
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        st = statistics(x, mask)
        x = standardize(x, st)
        xx.append(x)
        yy.append(y)
        rw.append(torch.full((len(x),), 0.5 / len(x), dtype=torch.float64))
        scales[scale] = dict(knots=knots, statistics=st, examples=len(x), pairs=ids)
    return (torch.cat(xx), torch.cat(yy), torch.cat(rw), scales)

def fit_dataset(base, ds):
    torch.set_num_threads(4)
    directory = Path(base) / ds / 'fit_parameters'
    records = {}
    ident = 'paper_calibration'
    for reader in ('Center', 'Observed', 'Manifold'):
        final = directory / (reader + '.json')
        if final.exists():
            rec = load(final)
            if rec['identity'] != ident:
                raise ValueError('Fit identity')
            records[reader] = rec
            continue
        x, y, rw, scales = prepare_training(base, ds, ['fit'], reader)
        candidates = []
        for lam in SCIENCE['lambda_grid']:
            p = directory / (reader + '-' + str(lam) + '.json')
            if p.exists():
                rec = load(p)
            else:
                fit = optimize(x, y, torch.ones(y.shape, dtype=torch.bool), rw, lam)
                rec = dict(identity=ident, reader=reader, lambda_value=lam, scales=scales, **fit)
                rec['validation_precision10'] = validation(base, ds, reader, rec)
                save(rec, p)
            if rec['identity'] != ident:
                raise ValueError('Candidate fit identity')
            candidates.append(rec)
        selected = max(candidates, key=lambda r: (r['validation_precision10'], r['lambda_value']))
        del x, y, rw
        x, y, rw, scales = prepare_training(base, ds, ['fit', 'validation'], reader)
        fit = optimize(
            x,
            y,
            torch.ones(y.shape, dtype=torch.bool),
            rw,
            selected['lambda_value'],
        )
        selection = [
            dict(
                lambda_value=record['lambda_value'],
                validation_precision10=record['validation_precision10'],
            )
            for record in candidates
        ]
        selection_pairs = [
            row['id']
            for row in manifest(base, ds)['rows']
            if row['split'] == 'validation'
        ]
        finalrec = dict(
            identity=ident,
            reader=reader,
            lambda_value=selected['lambda_value'],
            scales=scales,
            selection=selection,
            selection_pairs=selection_pairs,
            **fit,
        )
        save(finalrec, final)
        records[reader] = finalrec
        del x, y, rw
        print('FIT', ds, reader, finalrec['lambda_value'], finalrec['iterations'], flush=True)
    from .evaluation.geometry import evaluate_scores, auc
    threshold_file = directory / 'pose_threshold.json'
    if not threshold_file.exists():
        lam = records['Center']['lambda_value']
        valrec = load(directory / ('Center-' + str(lam) + '.json'))
        predictions = {th: [] for th in SCIENCE['pose_thresholds']}
        for scale in SCALES:
            for row in manifest(base, ds)['rows']:
                if row['split'] != 'validation':
                    continue
                r = load(scalarpath(base, ds, scale, row['id']))
                scores = score(r['bank'], 'Center', {'Center': valrec}, scale)
                for th in SCIENCE['pose_thresholds']:
                    v = evaluate_scores(scores, row, r['spec'], th, 'Center', scale)
                    predictions[th].append(v['metrics']['pose_error'])
        vals = {str(th): auc(e)['auc10'] for th, e in predictions.items()}
        chosen = max(SCIENCE['pose_thresholds'], key=lambda th: (vals[str(th)], -th))
        validation_pairs = [
            row['id']
            for row in manifest(base, ds)['rows']
            if row['split'] == 'validation'
        ]
        save(
            dict(
                identity=ident,
                threshold=chosen,
                selection_auc10=vals,
                reader='Center on 96 fitting pairs',
                shared_scales=list(SCALES),
                pairs=validation_pairs,
            ),
            threshold_file,
        )
    return dict(readers=records, pose=load(threshold_file))
