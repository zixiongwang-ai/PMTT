"""SPair-71k source observations, shared candidates and full-grid evaluation."""
import gc, time, collections
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from .io import identity, save, load, pack, unpack
from .settings import controls, conditions, data_root
from .backbone import Backbone
from .representation import anchors, structured, gpu, destroy
from .sampling import normalize
from .manifold import unit
from .datasets.preparation import prepared_manifest
from .readers import primitives, score
from .reader_fit import candidates, selected
from .candidates import candidate_bank
from .evaluation.correspondence import pck

@lru_cache(32)
def read_target(p):
    return unpack(p)

@lru_cache(256)
def read_source(p):
    return unpack(p)

def cachepath(base, scale, kind, key):
    return Path(base) / 'spair/cache' / scale / kind / (identity(key) + '.ptz')

def target(base, bb, cat, name):
    p = cachepath(base, bb.scale, 'target', (cat, name))
    if not p.exists():
        with Image.open(data_root('spair') / 'JPEGImages' / cat / name) as im:
            image = im.convert('RGB')
            r = bb(normalize(image.resize((1024, 1024), Image.Resampling.BICUBIC))[None].cuda())
            pack(dict(features=r['c'][0].cpu(), size=image.size), p)
    return read_target(p)

def source(base, bb, a):
    keys = [(a['category'], a['src_imname'], tuple(xy), tuple(a['src_bndbox'])) for xy in a['src_kps']]
    paths = [cachepath(base, bb.scale, 'source', k) for k in keys]
    missing = [i for i, p in enumerate(paths) if not p.exists()]
    if missing:
        with Image.open(data_root('spair') / 'JPEGImages' / a['category'] / a['src_imname']) as im:
            image = im.convert('RGB')
            coordinates = [
                (np.asarray(a['src_kps'][index], float) + 0.5).tolist()
                for index in missing
            ]
            missing_keys = [keys[index] for index in missing]
            aa = anchors(
                image,
                coordinates,
                missing_keys,
                a['src_bndbox'],
            )
            x = structured(bb, image, aa, 17, 4)
        for j, i in enumerate(missing):
            pack({k: v[j] for k, v in x.items()}, paths[i])
    rs = [read_source(p) for p in paths]
    return {k: torch.stack([r[k] for r in rs]) for k in rs[0]}

def extract(base, scale):
    bb = Backbone(scale)
    m = manifest(base)
    start = time.time()
    out = Path(base) / 'spair/train' / scale
    paths = []
    buffers = []
    weights = []
    rows = 0
    meta = []
    shard = 0
    for split, items in m['splits'].items():
        for j, item in enumerate(items):
            if split == 'trn':
                shardpath = out / f'{j // 128:05}.pt'
                if shardpath.exists():
                    if str(shardpath) not in paths:
                        paths.append(str(shardpath))
                    continue
            a = item['annotation']
            t = target(base, bb, a['category'], a['trg_imname'])
            x = source(base, bb, a)
            if split == 'trn':
                shardpath = out / f'{j // 128:05}.pt'
                if shardpath.exists():
                    if str(shardpath) not in paths:
                        paths.append(str(shardpath))
                    continue
                xx = gpu(x)
                tar = t['features'].cuda()
                bp = Path(base) / 'spair/banks' / (identity(item['id']) + '.pt')
                if not bp.exists():
                    if scale != 'S':
                        raise ValueError('S spatial bank missing')
                    cos = (unit(xx['c']) @ unit(tar).T).cpu().numpy()
                    spec = dict(grid=[64, 64], scale=[1024 / t['size'][0], 1024 / t['size'][1]])
                    ix, y = candidate_bank(cos, np.asarray(a['trg_kps'], float) + 0.5, spec, ('spair', item['id']))
                    save(dict(ix=torch.tensor(ix), y=torch.tensor(y, dtype=torch.float32)), bp)
                b = load(bp)
                parts = []
                for q in range(len(xx['c'])):
                    v = primitives({k: z[q:q + 1] for k, z in xx.items()}, tar[b['ix'][q]], chunk=256)['correct']
                    parts.append(v.cpu())
                buffers.append(torch.cat(parts))
                weights.append(b['y'])
                meta.append(item['id'])
                rows += len(x['c'])
                if j % 128 == 127 or j == len(items) - 1:
                    save(dict(x=torch.cat(buffers), y=torch.cat(weights), ids=meta), shardpath)
                    paths.append(str(shardpath))
                    buffers = []
                    weights = []
                    meta = []
            if j % 512 == 0:
                print('SPAIR', scale, split, j, len(items), flush=True)
    keep = {(it['annotation']['category'], it['annotation']['trg_imname']) for split in ('val', 'test') for it in m['splits'][split]}
    removed = 0
    for cat, name in {(it['annotation']['category'], it['annotation']['trg_imname']) for it in m['splits']['trn']} - keep:
        p = cachepath(base, scale, 'target', (cat, name))
        if p.exists():
            if not p.resolve().is_relative_to((Path(base) / 'spair/cache').resolve()):
                raise ValueError('Cache cleanup escape')
            removed += p.stat().st_size
            p.unlink()
    count = sum(
        len(
            torch.load(
                path,
                map_location='cpu',
                weights_only=False,
                mmap=True,
            )['x']
        )
        for path in paths
    )
    save(
        dict(paths=paths, count=count, scale=scale),
        out / 'index.json',
    )
    del bb
    gc.collect()
    torch.cuda.empty_cache()
    return dict(scale=scale, seconds=time.time() - start, train_shards=len(paths))

def _cached_source(base, scale, a):
    keys = [(a['category'], a['src_imname'], tuple(xy), tuple(a['src_bndbox'])) for xy in a['src_kps']]
    rs = [read_source(cachepath(base, scale, 'source', k)) for k in keys]
    return {k: torch.stack([r[k] for r in rs]) for k in rs[0]}

def evaluate(base, scale, validation=False):
    m = manifest(base)
    split = 'val' if validation else 'test'
    start = time.time()
    from .reader_fit import selected
    recs = candidates(base, 'spair') if validation else selected(base, 'spair')
    totals = collections.defaultdict(lambda: [0, 0])
    results = []
    donor_items = m['splits']['trn'][:2]
    pools = [(a['annotation']['src_imname'], _cached_source(base, scale, a['annotation'])['visible']) for a in donor_items]
    for j, item in enumerate(m['splits'][split]):
        a = item['annotation']
        p = Path(base) / 'spair' / ('validation' if validation else 'records') / scale / (identity(item['id']) + '.json')
        if p.exists():
            rr = load(p)
        else:
            x = gpu(_cached_source(base, scale, a))
            t = read_target(cachepath(base, scale, 'target', (a['category'], a['trg_imname'])))
            donor = next((v for name, v in pools if name != a['src_imname']))
            vd = donor[torch.arange(len(x['c'])) % len(donor)].cuda()
            use_controls = controls('spair', scale) and not validation
            donor_visible = vd if use_controls else None
            destroyed_manifold = (
                destroy(x, ('spair', item['id'], scale))
                if use_controls
                else None
            )
            b = primitives(
                x,
                t['features'].cuda(),
                donor_visible,
                destroyed_manifold,
                512,
            )
            strata_fields = (
                'viewpoint_variation',
                'scale_variation',
                'occlusion',
                'truncation',
            )
            rr = dict(
                pixel_convention=(
                    'zero-index annotation pixel centers mapped to PIL edge '
                    'coordinates with +0.5'
                ),
                id=item['id'],
                category=a['category'],
                keypoints=len(x['c']),
                strata={key: a.get(key) for key in strata_fields},
                conditions={},
            )
            for cond in list(recs) if validation else conditions('spair', scale):
                rep = cond.split('/')[0]
                records = recs[cond] if validation else recs
                scores = score(b, rep, records, scale)
                v = pck(scores, torch.tensor(a['trg_kps'], device='cuda') + 0.5, t['size'], a['trg_bndbox'])
                rr['conditions'][cond] = dict(
                    good=int(v['ok'].sum()),
                    count=len(v['ok']),
                    pck=float(v['ok'].float().mean()),
                    ranks=v['rank'].tolist(),
                    mean_rank=float(v['rank'].float().mean()),
                    mrr=float(v['rank'].double().reciprocal().mean()),
                )
            save(rr, p)
        results.append(rr)
        for c, v in rr['conditions'].items():
            totals[c][0] += v['good']
            totals[c][1] += v['count']
        if j % 512 == 0:
            print('SPAIR_EVAL', scale, split, j, flush=True)
    metrics = {k: v[0] / v[1] for k, v in totals.items()}
    directory = 'validation' if validation else 'records'
    save(
        dict(
            scale=scale,
            split=split,
            pairs=len(results),
            metrics=metrics,
            seconds=time.time() - start,
        ),
        Path(base) / 'spair' / directory / (scale + '.json'),
    )
    return metrics

def manifest(base):
    return prepared_manifest(base, 'spair')
