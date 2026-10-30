"""Center, actual Visible states, and their analytic Manifold factor."""
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
from .sampling import normalize, sample
from .manifold import unit, reference, residuals, canonical
from .local_views import layout
from .assets import destroyed_seed
from .io import ensure_finite


def anchors(image, xy, keys, bbox=None, seeds=None):
    """Build image-point observations from paper keys or explicit numeric seeds."""
    if len(xy) != len(keys) or (seeds is not None and len(seeds) != len(xy)):
        raise ValueError('Each point requires one observation identity')
    result = []
    for index, (point, key) in enumerate(zip(xy, keys)):
        seed = None if seeds is None else seeds[index]
        crops = layout(image, point, key, bbox, seed)[0]
        result.append(
            dict(
                id=index,
                xy=list(map(float, point)),
                crops=crops,
                key=key,
            )
        )
    return result


def gpu(x):
    return {k: v.cuda(non_blocking=True) for k, v in x.items()}


def crops_cpu(image, anchors):
    crops = []
    for anchor in anchors:
        for x, y, width, height in anchor['crops']:
            crop = image.resize(
                (224, 224),
                Image.Resampling.BICUBIC,
                box=(x, y, x + width, y + height),
            )
            crops.append(normalize(crop))
    return torch.stack(crops)

@torch.no_grad()
def encode_crops(backbone, pixels, anchors, chunk=17):
    """Encode 17-view groups into raw Center [N,D], Visible [N,16,D] and Manifold [N,D,8].
    
    Center samples raw final tokens; Visible samples normalized tokens and normalizes again."""
    states = []
    n = len(anchors)
    cp = torch.tensor(
        [crop for anchor in anchors for crop in anchor['crops']],
        device='cuda',
        dtype=torch.float32,
    )
    xy = torch.tensor(
        [anchor['xy'] for anchor in anchors for _ in range(17)],
        device='cuda',
        dtype=torch.float32,
    )
    grid = ((xy - cp[:, :2]) / cp[:, 2:] * 2 - 1)[:, None]
    if bool((grid.abs() > 1 - 1 / 14 + 1e-06).any()):
        raise ValueError('Observation coverage')
    for begin in range(0, len(pixels), chunk):
        batch_pixels = pixels[begin:begin + chunk].to(
            'cuda',
            non_blocking=True,
        )
        r = backbone(batch_pixels)
        g = grid[begin:begin + len(r['c'])]
        raw = sample(r['c'], g, r['hw'])[:, 0]
        vis = unit(sample(unit(r['c']), g, r['hw'])[:, 0])
        indices = torch.arange(begin, begin + len(raw), device='cuda')
        is_center = indices % 17 == 0
        states.append(torch.where(is_center[:, None], raw, vis))
    state = torch.cat(states).reshape(n, 17, -1)
    c = state[:, 0]
    v = state[:, 1:]
    a = reference(residuals(c, v)[1])['a']
    return dict(c=c, visible=v, manifold=a)

@torch.no_grad()
def destroyed(c, a, keys):
    """Generate spectrum-preserving tangent controls using one numeric seed per anchor."""
    u = unit(c)
    s = canonical(a)[1]
    rows = []
    for j, key in enumerate(keys):
        g = torch.Generator(device=c.device).manual_seed(int(key))
        z = torch.randn(a[j].shape, device=c.device, generator=g, dtype=torch.float32)
        z = z - u[j, :, None] * (u[j, :, None] * z).sum(0, keepdim=True)
        q = torch.linalg.qr(z, mode='reduced').Q
        rows.append(q * s[j][None])
    return torch.stack(rows)

def _prefetch(image, groups, prefetch_threads):
    if prefetch_threads == 0:
        for group in groups:
            yield (crops_cpu(image, group), group)
    else:
        with ThreadPoolExecutor(max_workers=prefetch_threads) as ex:
            futures = {}
            ahead = min(2 * prefetch_threads, len(groups))
            for i in range(ahead):
                futures[i] = ex.submit(crops_cpu, image, groups[i])
            for i, group in enumerate(groups):
                pixels = futures.pop(i).result()
                if i + ahead < len(groups):
                    futures[i + ahead] = ex.submit(crops_cpu, image, groups[i + ahead])
                yield (pixels, group)

def center_batches(image, aa, batch=17):
    result = []
    for begin in range(0, len(aa), batch):
        group = aa[begin:begin + batch]
        pixels = []
        for anchor in group:
            x, y, width, height = anchor['crops'][0]
            crop = image.resize(
                (224, 224),
                Image.Resampling.BICUBIC,
                box=(x, y, x + width, y + height),
            )
            pixels.append(normalize(crop))
        pixels = torch.stack(pixels)
        if len(pixels) < 17:
            pixels = torch.cat([pixels, pixels[-1:].expand(17 - len(pixels), -1, -1, -1)])
        result.append(pixels)
    return result

@torch.inference_mode()
def structured(
    backbone,
    image,
    aa,
    batch=17,
    prefetch_threads=4,
    centers_only=False,
    prepared_centers=None,
    keep_device=False,
):
    """Build patch objects at the supplied image-point anchors with fixed 17-view forwards.
    
    centers_only returns [N,D]; otherwise return Center, Visible and Manifold tensors."""
    if not aa:
        raise ValueError('Empty anchors')
    if batch != 17:
        raise ValueError('Structured observations require batch=17')
    outputs = []
    if centers_only:
        pixels_list = (
            prepared_centers
            if prepared_centers is not None
            else center_batches(image, aa, batch)
        )
        if len(pixels_list) != (len(aa) + batch - 1) // batch:
            raise ValueError('Prepared batch count')
        for pixels, begin in zip(pixels_list, range(0, len(aa), batch)):
            group = aa[begin:begin + batch]
            r = backbone(pixels.cuda())
            r['c'] = r['c'][:len(group)]
            cp = torch.tensor(
                [anchor['crops'][0] for anchor in group],
                device='cuda',
                dtype=torch.float32,
            )
            xy = torch.tensor(
                [anchor['xy'] for anchor in group],
                device='cuda',
                dtype=torch.float32,
            )
            grid = ((xy - cp[:, :2]) / cp[:, 2:] * 2 - 1)[:, None]
            outputs.append(sample(r['c'], grid, r['hw'])[:, 0].cpu())
        if len(outputs) != len(pixels_list):
            raise ValueError('Prepared batch count')
        return torch.cat(outputs)
    groups = [aa[i:i + 1] for i in range(len(aa))]
    for pix, group in _prefetch(image, groups, prefetch_threads):
        outputs.append(encode_crops(backbone, pix, group, batch))
    merged = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    return {k: ensure_finite(v if keep_device else v.cpu()) for k, v in merged.items()}

def destroy(x, key):
    seeds = [destroyed_seed(key, i) for i in range(len(x['c']))]
    return destroyed(x['c'], x['manifold'], seeds)

def observed_states(center, visible):
    """Return normalized Center plus the sixteen actual Visible states."""
    return torch.cat([unit(center)[:, None], visible], dim=1)
