"""Read official files and construct lightweight local indexes in paper order."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from pmtt.assets import csv_rows, task_name
from pmtt.settings import ASSETS, data_root
from pmtt.io import save, load
from pmtt.evaluation.video import boundary

def _rows(task):
    return csv_rows(ASSETS / 'manifests' / (task_name(task) + '.csv'))

def spair_manifest():
    root = data_root('spair')
    splits = {}
    for row in _rows('spair'):
        split, pair = row['split'], row['pair_id']
        directory = root / 'PairAnnotation' / split
        path = directory / (pair + '.json')
        if not path.is_file():
            matches = list(directory.glob(pair.split(':')[0] + '*.json'))
            if len(matches) != 1:
                raise FileNotFoundError(f'Expected one official annotation for {pair}')
            path = matches[0]
        annotation = load(path)
        for field, expected in [('src_imname', row['source']), ('trg_imname', row['target']), ('category', row['category'])]:
            if annotation[field] != expected:
                raise ValueError(f'SPair identity mismatch: {pair}/{field}')
        for name in (row['source'], row['target']):
            if not (root / 'JPEGImages' / row['category'] / name).is_file():
                raise FileNotFoundError(name)
        splits.setdefault(split, []).append(dict(id=pair, annotation=annotation))
    for split, rows in splits.items():
        official = (root / 'Layout/large' / (split + '.txt')).read_text().split()
        if official != [row['id'] for row in rows]:
            raise ValueError('SPair split ordering differs from paper identities')
    return dict(splits=splits)

def davis_manifest():
    root = data_root('davis')
    splits = {}
    for row in _rows('davis'):
        name, split = row['sequence'], row['split']
        frames = sorted((root / 'JPEGImages/480p' / name).glob('*.jpg'))
        if [p.stem for p in frames] != row['frames'].split():
            raise ValueError('DAVIS frame identities differ: ' + name)
        with Image.open(frames[0]) as image:
            width, height = image.size
        ratio = 448 / max(width, height)
        rw, rh = round(width * ratio), round(height * ratio)
        gw, gh = (rw + 15) // 16, (rh + 15) // 16
        with Image.open(root / 'Annotations/480p' / name / (frames[0].stem + '.png')) as image:
            mask = np.array(image)
        edge = binary_dilation(boundary(mask), iterations=3)
        anchors = []
        for iy in range(gh):
            for ix in range(gw):
                if (iy + .5) * 16 >= rh or (ix + .5) * 16 >= rw:
                    continue
                xy = [(ix + .5) * 16 / rw * width, (iy + .5) * 16 / rh * height]
                x, y = map(int, xy)
                label = int(mask[y, x])
                if label == 255:
                    continue
                x0, x1 = int(ix * 16 / rw * width), min(width, int((ix + 1) * 16 / rw * width) + 1)
                y0, y1 = int(iy * 16 / rh * height), min(height, int((iy + 1) * 16 / rh * height) + 1)
                patch = mask[y0:y1, x0:x1]
                values, counts = np.unique(patch[patch != 255], return_counts=True)
                membership = {str(int(v)): float(n / max(1, counts.sum())) for v, n in zip(values, counts)}
                anchors.append(dict(id=len(anchors), xy=xy, label=label, membership=membership, boundary=bool(edge[y, x])))
        objects = [int(v) for v in np.unique(mask) if v not in (0, 255)]
        if objects != [int(v) for v in row['objects'].split()]:
            raise ValueError('DAVIS object identities differ: ' + name)
        if set(objects) - {int(k) for a in anchors for k in a['membership']}:
            raise ValueError('Source grid misses a DAVIS object')
        splits.setdefault(split, []).append(
            dict(
                sequence=name,
                frames=[path.stem for path in frames],
                anchors=anchors,
                objects=objects,
                size=[width, height],
            )
        )
    for split, rows in splits.items():
        if (root / 'ImageSets/2017' / (split + '.txt')).read_text().split() != [r['sequence'] for r in rows]:
            raise ValueError('DAVIS split ordering differs')
    return dict(splits=splits)

def geometry_manifest(task):
    from .geometry import parse_pairs, inspect_pair
    official = parse_pairs(task)
    identities = _rows(task)
    rows = []
    for raw, saved in zip(official, identities):
        if any(raw[k] != saved[k] for k in ('id', 'source', 'target')):
            raise ValueError('Calibrated pair ordering differs')
        row = inspect_pair(raw)
        if row['error'] is not None:
            raise ValueError(row['error'])
        row.update(
            scene=saved['scene'],
            split=saved['split'],
            difficulty=saved['difficulty'],
            calibration_overlap=saved['calibration_overlap'] == 'True',
        )
        rows.append(row)
    if len(rows) != 1500:
        raise ValueError('Expected 1500 official pairs')
    return dict(rows=rows)

def dense_manifest(task):
    root = data_root(task)
    splits = {}
    for row in _rows(task):
        for key in ('image', 'label'):
            if not (root / row[key]).is_file():
                raise FileNotFoundError(root / row[key])
        record = {key: row[key] for key in ('id', 'image', 'label', 'scene')}
        if row['split'] == 'train':
            record['calibration_split'] = row['calibration_split'] or None
        splits.setdefault(row['split'], []).append(record)
    return dict(splits=splits)

def prepare_dataset(task, output):
    if task == 'spair':
        manifest = spair_manifest()
    elif task == 'davis':
        manifest = davis_manifest()
    elif task in ('megadepth', 'scannet'):
        manifest = geometry_manifest(task)
    else:
        manifest = dense_manifest(task)
    path = Path(output) / 'manifests' / (task + '.json')
    save(manifest, path)
    return path

@lru_cache(6)
def prepared_manifest(output, task):
    path = Path(output) / 'manifests' / (task + '.json')
    if not path.is_file():
        raise FileNotFoundError(f'Prepare {task} into {output} before evaluation')
    return load(path)
