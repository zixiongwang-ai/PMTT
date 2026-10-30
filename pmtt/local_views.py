"""Sixteen directionally complete local views about a canonical point."""
import math
import numpy as np
from .assets import observation_seed

def layout(image, xy, key=None, bbox=None, seed=None):
    """Return 17 crops and orbital metadata for a fixed image point.

    Coordinates and crop extents use original-image pixel edges. Supply an
    explicit numeric seed for a new point, or a key from the paper identities.
    """
    W, H = image.size
    x, y = map(float, xy)
    margin = min(x, W - x, y, H - y)
    if margin <= 1e-05:
        raise ValueError('coverage_infeasible: coordinate on image boundary')
    rng = np.random.default_rng(observation_seed(key) if seed is None else seed)
    context = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox is not None else min(W, H)
    side = min(W, H, 2 * margin, context) * 0.8
    anchor = [x - side / 2, y - side / 2, side, side]
    crops = [anchor]
    meta = []
    for j in range(16):
        sector = j % 8
        angle = sector * math.pi / 4 + rng.uniform(-0.1, 0.1)
        ratio = math.exp(rng.uniform(math.log(0.75), math.log(4 / 3)))
        scale = rng.uniform(0.7, 1.15)
        cw = side * scale * math.sqrt(ratio)
        ch = side * scale / math.sqrt(ratio)
        shift = rng.uniform(0.02, 0.08)
        qx = 0.5 + shift * math.cos(angle)
        qy = 0.5 + shift * math.sin(angle)
        shrink = min(1, x / (qx * cw), (W - x) / ((1 - qx) * cw), y / (qy * ch), (H - y) / ((1 - qy) * ch)) * 0.999
        cw *= shrink
        ch *= shrink
        crops.append([x - qx * cw, y - qy * ch, cw, ch])
        meta.append([qx, qy, qx - 0.5, qy - 0.5, math.log(cw * ch / side ** 2), math.log(cw / ch), math.sin(angle), math.cos(angle)])
    return (crops, meta)
