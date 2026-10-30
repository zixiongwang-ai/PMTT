"""Pixel-edge coordinates and continuous patch-feature sampling."""
import hashlib, random, math
from contextlib import contextmanager
import numpy as np
import torch
from torchvision.transforms import functional as TF
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)

def seed_for(*parts):
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), 'little') % 2 ** 32

def normalize(img):
    return TF.normalize(TF.to_tensor(img), MEAN, STD)

def crop_params(img):
    w, h = img.size
    aspect = w / h
    rmin, rmax = (0.75, 4 / 3)
    max_area = min(1.0, rmax / aspect, aspect / rmin)
    if max_area <= 0.5:
        r = min(max(aspect, rmin), rmax)
        area = max_area * w * h
        cw, ch = (math.sqrt(area * r), math.sqrt(area / r))
    else:
        low = max(math.log(rmin), math.log(0.5 * aspect))
        high = min(math.log(rmax), math.log(2 * aspect))
        for _ in range(10000):
            r = math.exp(random.uniform(low, high))
            cap = min(1.0, r / aspect, aspect / r)
            if random.random() * (max_area - 0.5) <= cap - 0.5:
                area = random.uniform(0.5, cap) * w * h
                break
        else:
            raise RuntimeError('Crop rejection failed within feasible support')
        cw, ch = (math.sqrt(area * r), math.sqrt(area / r))
    cw, ch = (min(cw, w), min(ch, h))
    return (random.uniform(0, max(0, w - cw)), random.uniform(0, max(0, h - ch)), cw, ch)

def correspondence(source, target, hw=(14, 14), size=(224, 224)):
    """Map source patch centers into a target crop and return normalized grid plus support mask."""
    device = source.device
    h, w = hw
    ih, iw = size
    yy, xx = torch.meshgrid((torch.arange(h, device=device) + 0.5) * 16, (torch.arange(w, device=device) + 0.5) * 16, indexing='ij')
    x = source[..., 0, None] + xx.flatten() / iw * source[..., 2, None]
    y = source[..., 1, None] + yy.flatten() / ih * source[..., 3, None]
    tx = (x - target[..., 0, None]) / target[..., 2, None] * iw
    ty = (y - target[..., 1, None]) / target[..., 3, None] * ih
    gx, gy = (tx / 16 - 0.5, ty / 16 - 0.5)
    valid = (gx >= 0) & (gx <= w - 1) & (gy >= 0) & (gy <= h - 1) & (tx >= 0) & (ty >= 0) & (tx < iw) & (ty < ih)
    grid = torch.stack((2 * (gx + 0.5) / w - 1, 2 * (gy + 0.5) / h - 1), -1)
    return (grid, valid)

def sample(values, grid, hw, border=False):
    """Bilinearly sample [B,H*W,D] states at normalized grid [B,N,2], with align_corners=False."""
    b, n, d = values.shape
    image = values.transpose(1, 2).reshape(b, d, *hw)
    sampled = torch.nn.functional.grid_sample(
        image,
        grid.unsqueeze(2),
        align_corners=False,
        padding_mode='border' if border else 'zeros',
    )
    return sampled.squeeze(-1).transpose(1, 2)

def teacher_target(c, grid, hw):
    from .manifold import unit
    return unit(sample(unit(c), grid, hw))

@contextmanager
def random_stream_seed(seed):
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.random.default_generator.manual_seed(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
