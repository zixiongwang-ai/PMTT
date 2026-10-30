"""First-frame DAVIS object propagation and train-side reader fitting."""
import gc, time, collections
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as TF
from .io import identity, seed, save, load, pack, unpack
from .settings import controls, conditions, data_root
from .backbone import Backbone
from .representation import anchors, structured, gpu, destroy
from .sampling import normalize
from .datasets.preparation import prepared_manifest
from .readers import primitives, score
from .reader_fit import candidates, selected
from .evaluation.video import prepare, jf_cached, object_scores

def path(base, scale, kind, key):
    return Path(base) / 'davis/cache' / scale / kind / (identity(key) + '.ptz')

def get_source(base, scale, row):
    return unpack(path(base, scale, 'source', row['sequence']))

def get_target(base, scale, row, frame):
    return unpack(path(base, scale, 'target', (row['sequence'], frame)))

def memberships(row, device='cuda'):
    objs = [0] + row['objects']
    return torch.tensor([[a['membership'].get(str(obj), 0.0) for obj in objs] for a in row['anchors']], device=device)

def prediction(scores, row, t):
    membership = memberships(row)
    s = object_scores(scores, membership)
    h, w = t['hw']
    rh, rw = t['resized']
    s = s.reshape(1, -1, h, w)
    pix = TF.interpolate(s, size=(h * 16, w * 16), mode='bilinear', align_corners=False)[..., :rh, :rw]
    pix = TF.interpolate(pix, size=tuple(t['size'][::-1]), mode='bilinear', align_corners=False)[0]
    ix = pix.argmax(0).cpu().numpy()
    objs = np.array([0] + row['objects'])
    return objs[ix]

def extract(base, scale, paper=False):
    bb = Backbone(scale)
    m = manifest(base)
    start = time.time()
    paths = []
    allpaths = []
    train = m['splits']['train']
    nval = max(1, len(train) // 5) if len(train) > 1 else 0
    validation = {r['sequence'] for r in sorted(train, key=lambda r: seed('davis', r['sequence']))[:nval]}
    protocol = dict(
        train_validation=sorted(validation),
        sampling=(
            'all train frames, 32 evenly spaced target patches/frame, '
            'all source patches'
        ),
    )
    save(protocol, Path(base) / 'davis/train_protocol.json')
    for split, rows in m['splits'].items():
        for row in rows:
            if paper and split == 'train' and (row not in m['splits']['train'][:2]):
                continue
            name = row['sequence']
            p = path(base, scale, 'source', name)
            if not p.exists():
                with Image.open(data_root('davis') / 'JPEGImages/480p' / name / (row['frames'][0] + '.jpg')) as im:
                    image = im.convert('RGB')
                    aa = anchors(image, [a['xy'] for a in row['anchors']], [(name, a['xy']) for a in row['anchors']])
                    x = structured(bb, image, aa, 17, 4)
                    pack(x, p)
            src = gpu(unpack(p))
            mem = memberships(row)
            table = Path(base) / 'davis/train' / scale / (name + '.pt')
            bs = []
            ys = []
            for pos, frame in enumerate(row['frames'][1:], 1):
                tp = path(base, scale, 'target', (name, frame))
                if not tp.exists():
                    with Image.open(data_root('davis') / 'JPEGImages/480p' / name / (frame + '.jpg')) as im:
                        image = im.convert('RGB')
                        w, h = image.size
                        r = 448 / max(w, h)
                        rw, rh = (round(w * r), round(h * r))
                        pix = TF.pad(normalize(image.resize((rw, rh), Image.Resampling.BICUBIC)), (0, -rw % 16, 0, -rh % 16))
                        z = bb(pix[None].cuda())
                    pack(dict(features=z['c'][0].cpu(), hw=z['hw'], size=[w, h], resized=[rh, rw]), tp)
                if split == 'train' and (not paper) and (not table.exists()):
                    t = unpack(tp)
                    h, w = t['hw']
                    rh, rw = t['resized']
                    W, H = t['size']
                    yy, xx = np.meshgrid((np.arange(h) + 0.5) * 16, (np.arange(w) + 0.5) * 16, indexing='ij')
                    valid = (yy < rh) & (xx < rw)
                    ids = np.flatnonzero(valid.ravel())
                    ids = ids[np.linspace(0, len(ids) - 1, min(32, len(ids))).astype(int)]
                    mask = np.array(Image.open(data_root('davis') / 'Annotations/480p' / name / (frame + '.png')))
                    label_y = np.minimum(
                        (yy.ravel()[ids] / rh * H).astype(int),
                        H - 1,
                    )
                    label_x = np.minimum(
                        (xx.ravel()[ids] / rw * W).astype(int),
                        W - 1,
                    )
                    labels = mask[label_y, label_x]
                    b = primitives(src, t['features'][ids].cuda(), chunk=512)['correct'].transpose(0, 1)
                    objs = [0] + row['objects']
                    y = torch.stack(
                        [
                            mem[:, objs.index(int(label))]
                            if int(label) in objs
                            else torch.zeros(len(mem), device='cuda')
                            for label in labels
                        ]
                    )
                    legal = y.sum(-1) > 0
                    y = y[legal]
                    y = y / y.sum(-1, keepdim=True)
                    bs.append(b[legal].cpu())
                    ys.append(y.cpu())
            if split == 'train' and (not paper):
                if not table.exists():
                    save(dict(x=torch.cat(bs), y=torch.cat(ys), sequence=name), table)
                allpaths.append(str(table))
                if name not in validation:
                    paths.append(str(table))
            print('DAVIS cache', scale, split, name, flush=True)
    save(dict(paths=paths, scale=scale), Path(base) / 'davis/train' / scale / 'index.json')
    save(dict(paths=allpaths, scale=scale), Path(base) / 'davis/train' / scale / 'all_index.json')
    del bb
    gc.collect()
    torch.cuda.empty_cache()
    return dict(seconds=time.time() - start, scale=scale, train_sequences=len(paths))

def evaluate(base, scale, validation=False):
    m = manifest(base)
    protocol = load(Path(base) / 'davis/train_protocol.json')
    rows = [r for r in m['splits']['train'] if r['sequence'] in protocol['train_validation']] if validation else m['splits']['val']
    if validation and (not rows):
        raise ValueError('Empty train-side validation split')
    recs = candidates(base, 'davis') if validation else selected(base, 'davis')
    donor_rows = m['splits']['train']
    allmetrics = collections.defaultdict(list)
    for row in rows:
        name = row['sequence']
        src = gpu(get_source(base, scale, row))
        donorrow = next((r for r in donor_rows if r['sequence'] != name), None)
        if donorrow is None:
            raise ValueError('No different-video donor')
        else:
            d = get_source(base, scale, donorrow)['visible']
            donor = d[torch.arange(len(src['c'])) % len(d)].cuda()
        xd = destroy(src, ('davis', name, scale))
        seqmetrics = collections.defaultdict(list)
        for pos, frame in enumerate(row['frames'][1:], 1):
            p = Path(base) / 'davis' / ('validation' if validation else 'records') / scale / name / (frame + '.json')
            old = load(p) if p.exists() else None
            if old is not None and all(('boundary_J' in v for os in old['conditions'].values() for v in os.values())):
                rr = old
            else:
                t = get_target(base, scale, row, frame)
                target = t['features'].cuda()
                chunks = []
                for start in range(0, len(src['c']), 32):
                    xx = {k: v[start:start + 32] for k, v in src.items()}
                    use_controls = controls('davis', scale) and not validation
                    donor_chunk = donor[start:start + 32] if use_controls else None
                    destroyed_chunk = xd[start:start + 32] if use_controls else None
                    chunks.append(
                        primitives(
                            xx,
                            target,
                            donor_chunk,
                            destroyed_chunk,
                            512,
                        )
                    )
                b = {k: torch.cat([r[k] for r in chunks]) for k in chunks[0]}
                truth = np.array(Image.open(data_root('davis') / 'Annotations/480p' / name / (frame + '.png')))
                ctx = prepare(truth, row['objects'])
                rr = dict(sequence=name, frame=frame, position=pos, temporal_fraction=pos / (len(row['frames']) - 1), conditions={})
                for cond in list(recs) if validation else conditions('davis', scale):
                    rep = cond.split('/')[0]
                    pred = prediction(score(b, rep, recs[cond] if validation else recs, scale), row, t)
                    rr['conditions'][cond] = {}
                    for obj in row['objects']:
                        metric = jf_cached(pred, obj, ctx)
                        valid = ctx[0]
                        truth_obj = truth == obj
                        pred_obj = pred == obj
                        band = ctx[2][obj][2]
                        for region, mask in [('boundary_J', band), ('interior_J', ~band)]:
                            truth_region = truth_obj & mask & valid
                            pred_region = pred_obj & mask & valid
                            union = (truth_region | pred_region).sum()
                            metric[region] = float((truth_region & pred_region).sum() / union) if union else 1.0
                        if old is not None:
                            for k in ('J', 'F', 'JF'):
                                if abs(old['conditions'][cond][str(obj)][k] - metric[k]) > 1e-12:
                                    raise ValueError('DAVIS metric mismatch in existing output')
                        rr['conditions'][cond][str(obj)] = metric
                save(rr, p)
            for cond, objects in rr['conditions'].items():
                seqmetrics[cond].append(float(np.mean([v['JF'] for v in objects.values()])))
        for c, vals in seqmetrics.items():
            allmetrics[c].append((float(np.mean(vals)) * len(row['objects']), len(row['objects'])))
    metrics = {c: float(sum((a for a, b in v)) / sum((b for a, b in v))) for c, v in allmetrics.items()}
    report_path = (
        Path(base)
        / 'davis'
        / ('validation' if validation else 'records')
        / (scale + '.json')
    )
    save(dict(scale=scale, sequences=len(rows), metrics=metrics), report_path)
    return metrics

def manifest(base):
    return prepared_manifest(base, 'davis')
