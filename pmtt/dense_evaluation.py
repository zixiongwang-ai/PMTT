"""Frozen or refitted task axes evaluated at all 196 native patch locations."""
import math
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as TF
from .settings import ASSETS, conditions, controls
from .assets import csv_rows, load_task_axis, load_mixtures
from .backbone import Backbone
from .datasets.preparation import prepared_manifest
from .datasets.dense import image_label
from .representation import anchors, structured, destroy
from .dense import grid, points, scalar_features, control_features, mix, valid_depth, depth_metrics, confusion, seg_metrics, upsample_argmax
from .io import identity, load, pack

def parameters(output, task, scale, mode):
    if mode == 'paper':
        return load_task_axis(task, scale), load_mixtures(task)
    root = Path(output) / task
    axis = load(root / 'axis' / scale / 'axis.pt')
    mixtures = {
        representation: load(
            root / 'mixtures' / (representation + '.json')
        )['selected']
        for representation in ('Observed', 'Manifold')
    }
    return axis, mixtures

def donors(backbone, output, task, scale):
    identities = [r for r in csv_rows(ASSETS / 'manifests/fitting/dense_donors.csv') if r['task'] == task and r['scale'] == scale]
    rows = {r['id']: r for r in prepared_manifest(output, task)['splits']['train']}
    result = []
    for identity_row in identities:
        row = rows[identity_row['image_id']]
        image, label = image_label(row, task)
        xy, _ = points(image, label, row, task, True)
        aa = anchors(image, xy, [(task, row['id'], 'calibration', i) for i in range(len(xy))])
        result.append(dict(id=row['id'], x=structured(backbone, image, aa, 17, 4, keep_device=True)))
    return result

def evaluate(output, task, scale, mode='paper'):
    backbone = Backbone(scale)
    axis, mixtures = parameters(output, task, scale, mode)
    donor_rows = donors(backbone, output, task, scale) if controls(task, scale) else []
    rows = prepared_manifest(output, task)['splits']['test']
    for row in rows:
        path = Path(output) / task / 'records' / scale / (identity(row['id']) + '.ptz')
        image, label = image_label(row, task)
        xy = grid(image.size)
        aa = anchors(image, xy, [(task, row['id'], 'dense', i) for i in range(len(xy))])
        x = structured(backbone, image, aa, 17, 4, keep_device=True)
        features = scalar_features(x, axis, task)
        if donor_rows:
            donor = next(r for r in donor_rows if r['id'] != row['id'])
            visible = donor['x']['visible']
            visible = visible[torch.arange(len(xy), device=visible.device) % len(visible)]
            donor_features = control_features(x, axis, task, 'Observed', donor=visible)
            destroyed_features = control_features(x, axis, task, 'Manifold', destroyed=destroy(x, (task, row['id'], scale)))
        result = dict(id=row['id'], scene=row['scene'], scale=scale, patches=196, conditions={})
        if task == 'ade':
            edge = ndimage.maximum_filter(label, size=3) != ndimage.minimum_filter(label, size=3)
            edge = ndimage.binary_dilation(edge, iterations=2)
        else:
            valid = valid_depth(label)
            edge = np.hypot(*np.gradient(label)) > .1
        for condition in conditions(task, scale):
            if condition == 'Center':
                z = features['Center']
            else:
                rep = {'DonorObserved': 'Observed', 'DestroyedManifold': 'Manifold'}.get(condition, condition)
                if condition == 'DonorObserved':
                    values = donor_features
                elif condition == 'DestroyedManifold':
                    values = destroyed_features
                else:
                    values = features[rep]
                z = mix(values, mixtures[rep], scale)
            if task == 'nyu':
                native = z[:, 0].clamp(math.log(.001), math.log(10)).exp().reshape(1, 1, 14, 14)
                pred = TF.interpolate(native, size=label.shape, mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                metrics = depth_metrics(pred, label, valid)
                strata_masks = [
                    ('near', label < 2),
                    ('mid', (label >= 2) & (label < 5)),
                    ('far', label >= 5),
                    ('boundary', edge),
                    ('smooth', ~edge),
                ]
                metrics['strata'] = {
                    name: depth_metrics(pred, label, valid & mask)
                    for name, mask in strata_masks
                }
                result['conditions'][condition] = dict(metrics=metrics, score=-metrics['AbsRel'])
            else:
                pred = upsample_argmax(z, label.shape).cpu().numpy()
                cm, cb, ci = confusion(pred, label), confusion(pred, label, edge), confusion(pred, label, ~edge)
                metrics = seg_metrics(cm)
                result['conditions'][condition] = dict(
                    cm=torch.from_numpy(cm),
                    boundary_cm=torch.from_numpy(cb),
                    interior_cm=torch.from_numpy(ci),
                    metrics=metrics,
                    score=metrics['miou'],
                )
        pack(result, path)
