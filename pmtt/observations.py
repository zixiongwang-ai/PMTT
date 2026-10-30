"""Dense Orbital Views retain one image point across context and aspect changes."""

import math
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from .sampling import crop_params, normalize, correspondence, random_stream_seed

SECTORS = (0, 2, 4, 6, 1, 3, 5, 7)

def proposed(img, exception):
    p = crop_params(img)
    if not exception:
        return p
    W, H = img.size
    ratio = math.exp(random.uniform(math.log(0.75), math.log(4 / 3)))
    scale = random.uniform(0.65, 0.85)
    h = min(H, W / ratio) * scale
    w = h * ratio
    return (random.uniform(0, W - w), random.uniform(0, H - h), w, h)

def group(
    img,
    key,
    epoch=0,
    visible_count=16,
    hidden_count=16,
    size=224,
    purpose='intrinsic',
    packed=False,
    observation_seed=None,
):
    """Construct canonical and orbital observations from one image and a numeric random seed.
    
    Paper calls use visible_count=16 and hidden_count=16. Crops retain one supported image point;
    returned uint8 pixels include the group-shared photometric transform when packed=True."""
    if visible_count not in (1, 4, 8, 16, 32):
        raise ValueError('Unsupported observation K')
    W, H = img.size
    max_area = min(1.0, 4 / 3 / (W / H), W / H / 0.75)
    exception = max_area <= 0.5
    reason = 'zero translation support' if exception else None
    with random_stream_seed(observation_seed):
        if random.random() < 0.8:
            img = T.ColorJitter(0.4, 0.4, 0.2, 0.1)(img)
        if random.random() < 0.2:
            img = TF.to_grayscale(img, num_output_channels=3)
        for trial in range(0 if exception else 128):
            anchor = proposed(img, exception)
            crops = [anchor]
            labels = [-1]
            valid = torch.ones((size // 16) ** 2, dtype=torch.bool)
            ok = True
            for j in range(visible_count + hidden_count):
                wanted = SECTORS[(j if j < visible_count else j - visible_count) % 8]
                h = size // 16
                coordinates = (torch.arange(h) + 0.5) / h
                yy, xx = torch.meshgrid(
                    coordinates,
                    coordinates,
                    indexing='ij',
                )
                xx = xx.flatten()
                yy = yy.flatten()
                px = anchor[0] + xx * anchor[2]
                py = anchor[1] + yy * anchor[3]
                found = False
                for _ in range(16):
                    proposals = proposal_batch(img, 256, exception)
                    cp = torch.from_numpy(proposals)
                    remaining = valid.nonzero().flatten()
                    tx = (px[remaining][None] - cp[:, 0, None]) / cp[:, 2, None]
                    ty = (py[remaining][None] - cp[:, 1, None]) / cp[:, 3, None]
                    dx = tx - xx[remaining]
                    dy = ty - yy[remaining]
                    labels_here = torch.floor(
                        (torch.atan2(dy, dx) + math.pi / 8)
                        / (math.pi / 4)
                    ).long() % 8
                    masks = (
                        (tx >= 0.5 / h)
                        & (tx <= 1 - 0.5 / h)
                        & (ty >= 0.5 / h)
                        & (ty <= 1 - 0.5 / h)
                    )
                    masks &= (labels_here == wanted) & (dx.abs() + dy.abs() > 1e-07)
                    eligible = masks.any(-1).nonzero()
                    if len(eligible):
                        at = int(eligible[0])
                        candidate = proposals[at].tolist()
                        mask = torch.zeros_like(valid)
                        mask[remaining] = masks[at]
                        found = True
                        break
                if not found:
                    ok = False
                    break
                crops.append(candidate)
                labels.append(wanted)
                valid &= mask
            if ok:
                coords = torch.tensor(crops[1:])
                areas = coords[:, 2] * coords[:, 3] / (W * H)
                aspects = (coords[:, 2] / coords[:, 3]).log()
                rlo = (
                    max(math.log(0.75), math.log(0.5 * W / H))
                    if not exception
                    else math.log(0.75)
                )
                rhi = (
                    min(math.log(4 / 3), math.log(2 * W / H))
                    if not exception
                    else math.log(4 / 3)
                )
                area_span = (
                    max_area - 0.5
                    if not exception
                    else float(areas.max() - areas.min())
                )
                for lo in range(0, len(areas), 16):
                    aa = areas[lo:lo + 8]
                    rr = aspects[lo:lo + 8]
                    limited_area = aa.max() - aa.min() < 0.15 * area_span
                    limited_aspect = rr.max() - rr.min() < 0.15 * (rhi - rlo)
                    if len(aa) >= 8 and (limited_area or limited_aspect):
                        ok = False
                        break
            if ok:
                anchor_area = anchor[2] * anchor[3] / (W * H)
                for lo in (0, 16):
                    aa = areas[lo:lo + 16]
                    if not aa.min() < anchor_area < aa.max():
                        ok = False
                        break
            if ok:
                bounds = torch.tensor(crops, dtype=torch.float64)
                inside_start = (bounds[:, :2] >= 0).all()
                inside_end = (
                    bounds[:, :2] + bounds[:, 2:]
                    <= torch.tensor([W, H])
                ).all()
                ok = bool(inside_start and inside_end)
            if ok:
                valid &= supported_positions(torch.tensor(crops, dtype=torch.float32), labels, size)
                ok = bool(valid.any())
            if ok:
                break
        else:
            exception = True
            reason = reason or 'bounded coordinate coverage search exhausted'
            crops, labels, valid = construct_covered(
                img,
                visible_count,
                hidden_count,
                size,
            )
        valid &= supported_positions(torch.tensor(crops, dtype=torch.float32), labels, size)
        if not valid.any():
            raise ValueError(
                f'No supported anchor: {key}, epoch={epoch}, purpose={purpose}'
            )
        positions = torch.where(valid)[0]
        selection = torch.linspace(
            0,
            len(positions) - 1,
            min(8, len(positions)),
        ).long()
        positions = positions[selection]
        views = [
            img.resize(
                (size, size),
                Image.Resampling.BICUBIC,
                box=(x, y, x + width, y + height),
            )
            for x, y, width, height in crops
        ]
        images = torch.stack(
            [TF.pil_to_tensor(view) if packed else normalize(view) for view in views]
        )
    crop_tensor = torch.tensor(crops, dtype=torch.float32)
    if exception:
        area_fraction = crop_tensor[:, 2:].prod(-1) / (W * H)
        scale_range = [
            float(area_fraction.min()),
            float(area_fraction.max()),
        ]
    else:
        scale_range = None
    exception_record = dict(
        applied=exception,
        reason=reason,
        scale_range=scale_range,
        original_size=[W, H],
    )
    return dict(
        images=images,
        crops=crop_tensor,
        positions=positions,
        sectors=labels,
        key=key,
        visible_count=visible_count,
        hidden_count=hidden_count,
        epoch=epoch,
        purpose=purpose,
        exception=exception_record,
    )

def proposal_batch(img, count, exception):
    W, H = img.size
    aspect = W / H
    if exception:
        ratio = np.exp(np.random.uniform(math.log(0.75), math.log(4 / 3), count))
        scale = np.random.uniform(0.65, 0.85, count)
        ch = np.minimum(H, W / ratio) * scale
        cw = ch * ratio
    else:
        maximum = min(1.0, 4 / 3 / aspect, aspect / 0.75)
        low = max(math.log(0.75), math.log(0.5 * aspect))
        high = min(math.log(4 / 3), math.log(2 * aspect))
        ratios = []
        areas = []
        remaining = count
        for _ in range(128):
            proposed_ratio = np.exp(
                np.random.uniform(low, high, max(remaining * 3, 32))
            )
            cap = np.minimum(1.0, np.minimum(proposed_ratio / aspect, aspect / proposed_ratio))
            accepted = np.random.random(len(cap)) * (maximum - 0.5) <= cap - 0.5
            rr = proposed_ratio[accepted][:remaining]
            cc = cap[accepted][:remaining]
            ratios.append(rr)
            areas.append(np.random.uniform(0.5, cc))
            remaining -= len(rr)
            if remaining == 0:
                break
        if remaining:
            raise ValueError('Vector crop rejection exhausted')
        ratio = np.concatenate(ratios)
        area = np.concatenate(areas) * W * H
        cw = np.sqrt(area * ratio)
        ch = np.sqrt(area / ratio)
    x = np.random.random(count) * np.maximum(0, W - cw)
    y = np.random.random(count) * np.maximum(0, H - ch)
    return np.stack([x, y, cw, ch], -1).astype(np.float32)

def to_device_images(images, device='cuda'):
    """Transfer packed [N,3,H,W] uint8 pixels and apply the exact FP32 normalization lookup."""
    value = images.to(device, non_blocking=True)
    if value.dtype == torch.uint8:
        from .sampling import MEAN, STD
        table = (
            torch.arange(256, dtype=torch.float32)
            .expand(3, -1)
            .clone()
            .div_(255)
        )
        table.sub_(torch.tensor(MEAN)[:, None]).div_(torch.tensor(STD)[:, None])
        table = table.to(device)
        channels = torch.arange(3, device=device)[None, :, None, None]
        value = table[channels, value.long()]
    return value

def construct_covered(img, visible_count, hidden_count, size):
    W, H = img.size
    h = size // 16
    position = h // 2 * h + h // 2
    q = (h // 2 + 0.5) / h
    for shrink in np.arange(0.99, 0.09, -0.01):
        side = float(shrink) * min(W, H)
        anchor = [(W - side) / 2, (H - side) / 2, side, side]
        px = anchor[0] + q * side
        py = anchor[1] + q * side
        crops = [anchor]
        labels = [-1]
        ok = True
        for j in range(visible_count + hidden_count):
            k = j if j < visible_count else j - visible_count
            wanted = SECTORS[k % 8]
            if k % 8 == 0:
                aspect_order = random.sample(range(8), 8)
                scale_order = random.sample(range(8), 8)
            ratio = math.exp(
                math.log(0.75)
                + (math.log(4 / 3) - math.log(0.75))
                * ((aspect_order[k % 8] + 0.5) / 8)
            )
            scale = 0.9 + 0.2 * ((scale_order[k % 8] + 0.5) / 8)
            cw = side * scale * math.sqrt(ratio)
            ch = side * scale / math.sqrt(ratio)
            angle = wanted * math.pi / 4 + random.uniform(-0.08, 0.08)
            shift = random.uniform(0.004, 0.008)
            tx = q + shift * math.cos(angle)
            ty = q + shift * math.sin(angle)
            crop = [px - tx * cw, py - ty * ch, cw, ch]
            if (
                crop[0] < 0
                or crop[1] < 0
                or crop[0] + cw > W
                or crop[1] + ch > H
            ):
                ok = False
                break
            crops.append(crop)
            labels.append(wanted)
        if ok:
            valid = torch.zeros(h * h, dtype=torch.bool)
            valid[position] = True
            valid &= supported_positions(torch.tensor(crops, dtype=torch.float32), labels, size)
            if valid.any():
                return (crops, labels, valid)
    raise ValueError('coverage_infeasible: no legal complete stencil')

def supported_positions(crops, labels, size=224):
    """Return the legal canonical patch mask after FP32 crop serialization.
    
    Crops are [views,4] in x,y,width,height image coordinates; labels encode eight directions."""
    h = size // 16
    count = len(crops) - 1
    grid, mask = correspondence(
        crops[:1].expand(count, -1),
        crops[1:],
        (h, h),
        (size, size),
    )
    cp = crops.double()
    coordinates = (torch.arange(h, dtype=torch.float64) + 0.5) / h
    yy, xx = torch.meshgrid(
        coordinates,
        coordinates,
        indexing='ij',
    )
    q = torch.stack([xx.flatten(), yy.flatten()], -1)
    xy = cp[0, :2] + q * cp[0, 2:]
    crop_start = cp[1:, :2, None].transpose(1, 2)
    crop_extent = cp[1:, 2:, None].transpose(1, 2)
    t = (xy[None] - crop_start) / crop_extent
    g = t * h - 0.5
    eps = torch.finfo(torch.float32).eps
    uncertainty = 2 * eps * (
        (2 * (q * cp[0, 2:]).abs() + xy.abs())[None]
        / cp[1:, 2:][:, None]
        + 8 * t.abs()
        + 2
    )
    safe = ((g > uncertainty * h) & (g < h - 1 - uncertainty * h)).all(-1)
    delta = t - q
    actual = torch.floor(
        (torch.atan2(delta[..., 1], delta[..., 0]) + math.pi / 8)
        / (math.pi / 4)
    ).long() % 8
    safe &= actual == torch.as_tensor(labels[1:])[:, None]
    angles = torch.as_tensor(labels[1:], dtype=torch.float64) * math.pi / 4
    along = (
        delta[..., 0] * angles.cos()[:, None]
        + delta[..., 1] * angles.sin()[:, None]
    )
    across = (
        -delta[..., 0] * angles.sin()[:, None]
        + delta[..., 1] * angles.cos()[:, None]
    )
    safe &= along * math.tan(math.pi / 8) - across.abs() > 2 * uncertainty.sum(-1)
    safe &= delta.abs().sum(-1) > 1e-07
    return (safe & mask).all(0)
