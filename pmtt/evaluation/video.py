"""DAVIS region overlap and boundary F, including empty-mask conventions."""
import math
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as TF

def boundary(mask):
    e = np.zeros_like(mask)
    s = np.zeros_like(mask)
    se = np.zeros_like(mask)
    e[:, :-1] = mask[:, 1:]
    s[:-1] = mask[1:]
    se[:-1, :-1] = mask[1:, 1:]
    b = (mask != e) | (mask != s) | (mask != se)
    b[-1, :] = mask[-1, :] != e[-1, :]
    b[:, -1] = mask[:, -1] != s[:, -1]
    b[-1, -1] = False
    return b

def jf(pred, truth, obj):
    valid = truth != 255
    p = (pred == obj) & valid
    t = (truth == obj) & valid
    union = (p | t).sum()
    j = float((p & t).sum() / union) if union else 1.0
    bp = boundary(p)
    bt = boundary(t)
    radius = int(math.ceil(0.008 * np.linalg.norm(truth.shape)))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    pd = ndimage.binary_dilation(bp, structure=disk)
    td = ndimage.binary_dilation(bt, structure=disk)
    precision = float((bp & td).sum() / bp.sum()) if bp.any() else float(not bt.any())
    recall = float((bt & pd).sum() / bt.sum()) if bt.any() else float(not bp.any())
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(J=j, F=f, JF=(j + f) / 2)

def prepare(truth, objects):
    valid = truth != 255
    radius = int(math.ceil(0.008 * np.linalg.norm(truth.shape)))
    out = {}
    for obj in objects:
        t = (truth == obj) & valid
        bt = boundary(t)
        out[obj] = (t, bt, ndimage.distance_transform_edt(~bt) <= radius if bt.any() else np.zeros_like(bt))
    return (valid, radius, out)

def jf_cached(pred, obj, context):
    valid, radius, targets = context
    t, bt, td = targets[obj]
    p = (pred == obj) & valid
    union = (p | t).sum()
    j = float((p & t).sum() / union) if union else 1.0
    bp = boundary(p)
    npred = int(bp.sum())
    ntrue = int(bt.sum())
    if npred:
        pd = ndimage.distance_transform_edt(~bp) <= radius
        precision = float((bp & td).sum() / npred)
    else:
        pd = np.zeros_like(bp)
        precision = float(ntrue == 0)
    recall = float((bt & pd).sum() / ntrue) if ntrue else float(npred == 0)
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(J=j, F=f, JF=(j + f) / 2)

def object_scores(scores, membership, topk=8, temperature=1.0):
    result = []
    for k in range(membership.shape[1]):
        valid = membership[:, k] > 0
        if not valid.any():
            raise ValueError('Missing source object')
        s = scores[valid] + membership[valid, k, None].double().log()
        top = s.topk(min(topk, len(s)), dim=0).values
        result.append((torch.logsumexp(top / temperature, 0) - math.log(len(top))) * temperature)
    return torch.stack(result)
