"""All-pair geometry correspondence and separate 64-anchor supplementary pose."""

import gc
import time

from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from .io import identity, save, load, pack, unpack
from .settings import REPS, conditions, ASSETS
from .backbone import Backbone
from .representation import anchors, structured, gpu, destroy
from .sampling import normalize
from .datasets.preparation import prepared_manifest
from .readers import score
from .features import primitives
from .datasets.geometry import image_path, anchor_grid, resize_spec, reproject
from .evaluation.geometry import evaluate_scores
from .assets import load_reader_parameters


@lru_cache(32)
def read_source(p):
    return unpack(p)

@lru_cache(16)
def read_target(p):
    return unpack(p)

def path(base, ds, scale, kind, key):
    return Path(base) / ds / 'cache' / scale / kind / (identity(key) + '.ptz')

def source(base, bb, ds, name, n=32):
    p = path(base, ds, bb.scale, 'source', (name, n))
    if not p.exists():
        with Image.open(image_path(ds, name)) as im:
            image = im.convert('RGB')
            if n == 32:
                xy = anchor_grid(image.size)
            else:
                w, h = image.size
                xx, yy = np.meshgrid(
                    np.linspace(0.08 * w, 0.92 * w, 8),
                    np.linspace(0.08 * h, 0.92 * h, 8),
                )
                xy = np.stack([xx.ravel(), yy.ravel()], 1)
            keys = [
                (ds, name, index)
                if n == 32
                else (ds, name, 'pose64', index)
                for index in range(n)
            ]
            aa = anchors(image, xy, keys)
            x = structured(bb, image, aa, 17, 4)
            pack(dict(x=x, xy=xy.tolist(), observation_identity=identity(aa)), p)
    return read_source(p)

def target(base, bb, ds, name):
    p = path(base, ds, bb.scale, 'target', name)
    if not p.exists():
        with Image.open(image_path(ds, name)) as im:
            image = im.convert('RGB')
            spec = resize_spec(image.size)
            resized = image.resize(
                tuple(spec['resized']),
                Image.Resampling.BICUBIC,
            )
            r = bb(normalize(resized)[None].cuda())
            pack(dict(features=r['c'][0].cpu(), spec=spec), p)
    return read_target(p)

def run(base, ds, scale, mode='paper'):
    start = time.time()
    bb = Backbone(scale)
    m = manifest(base, ds)
    rows = m['rows']
    fitted = Path(base) / ds / 'coefficients' / (scale + '.json')
    records = load(fitted) if mode == 'refit' else load_reader_parameters(ds, scale)
    names = sorted({row['source'] for row in rows})
    threshold_path = Path(base) / ds / 'pose_threshold.json'
    threshold = (
        load(threshold_path)['threshold']
        if mode == 'refit'
        else load(ASSETS / 'calibration/geometry.json')[ds]['threshold']
    )
    for j, row in enumerate(rows):
        p = Path(base) / ds / 'records' / scale / (row['id'] + '.json')
        need64 = scale in ('B', 'H+')
        p64 = Path(base) / ds / 'pose64' / scale / (row['id'] + '.json')
        if p.exists() and (not need64 or p64.exists()):
            continue
        t = target(base, bb, ds, row['target'])
        x = gpu(source(base, bb, ds, row['source'])['x'])
        if not p.exists():
            donorname = names[(names.index(row['source']) + 1) % len(names)]
            vd = source(base, bb, ds, donorname)['x']['visible'].cuda()
            destroyed_manifold = destroy(x, (ds, row['id'], scale))
            b = primitives(
                x,
                t['features'].cuda(),
                vd,
                destroyed_manifold,
                512,
            )
            values = {}
            for cond in conditions(ds, scale):
                scores = score(b, cond, records, scale).cpu().numpy()
                values[cond] = evaluate_scores(
                    scores,
                    row,
                    t['spec'],
                    threshold,
                    cond,
                    scale,
                )
            save(
                dict(
                    id=row['id'],
                    scale=scale,
                    scene=row['scene'],
                    calibration_overlap=row['calibration_overlap'],
                    difficulty=row['difficulty'],
                    anchors=32,
                    donor_source=donorname,
                    conditions=values,
                ),
                p,
            )
        if need64 and (not p64.exists()):
            src = source(base, bb, ds, row['source'], 64)
            sx = gpu(src['x'])
            rr = dict(row)
            rr['gt'] = reproject(row, np.asarray(src['xy']))
            b = primitives(sx, t['features'].cuda(), chunk=512)
            values = {}
            for cond in REPS:
                scores = score(b, cond, records, scale).cpu().numpy()
                val = evaluate_scores(
                    scores,
                    rr,
                    t['spec'],
                    threshold,
                    cond,
                    scale,
                )
                metric_names = (
                    'pose_error',
                    'failure',
                    'success',
                    'inlier_count',
                    'inlier_ratio',
                    'assigned_matches',
                )
                values[cond] = dict(
                    pose=val['pose'],
                    metrics={key: val['metrics'][key] for key in metric_names},
                )
            save(
                dict(
                    id=row['id'],
                    scale=scale,
                    anchors=64,
                    role='pose_only_supplementary',
                    conditions=values,
                ),
                p64,
            )
        if j % 50 == 0:
            print('WIDE', ds, scale, j, len(rows), flush=True)
    del bb
    gc.collect()
    torch.cuda.empty_cache()
    return dict(dataset=ds, scale=scale, pairs=len(rows), seconds=time.time() - start)

def manifest(base, ds):
    return prepared_manifest(base, ds)
