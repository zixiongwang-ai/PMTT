"""Frozen identities and fitted parameters, independent of evaluation outputs."""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .settings import ASSETS, REPS

def task_name(task):
    return {'nyu': 'nyuv2', 'ade': 'ade20k'}.get(task, task)

def csv_rows(path):
    with Path(path).open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))

def load_cohort_manifest(cohort):
    rows = csv_rows(ASSETS / 'cohorts' / (cohort + '_cohort.csv'))
    with np.load(ASSETS / 'cohorts' / (cohort + '_observations.npz'), allow_pickle=False) as data:
        observations = {key: data[key].copy() for key in data.files}
    return rows, observations

def load_reader_parameters(task, scale=None):
    root = ASSETS / 'readers' / task
    if task in ('megadepth', 'scannet'):
        return json.loads((root / (scale + '.json')).read_text())
    return {rep: json.loads((root / (rep + '.json')).read_text()) for rep in REPS}

def load_task_axis(task, scale):
    with np.load(ASSETS / 'task_axes' / task_name(task) / (scale + '.npz'), allow_pickle=False) as data:
        result = {'w': torch.from_numpy(data['weight'].copy()), 'b': torch.from_numpy(data['bias'].copy())}
        if 'class_frequency' in data:
            result['class_frequency'] = data['class_frequency'].tolist()
    return result

def load_mixtures(task):
    root = ASSETS / 'task_axes' / task_name(task)
    return {rep: json.loads((root / (rep + '.json')).read_text()) for rep in ('Observed', 'Manifold')}

@lru_cache(3)
def _seed_arrays(kind):
    with np.load(ASSETS / 'calibration' / (kind + '_seeds.npz'), allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}

def _seed(kind, key, anchor=None):
    raw = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(',', ':')).encode()).digest()[:16]
    needle = np.frombuffer(raw, dtype='V16')[0]
    data = _seed_arrays(kind)
    keys, seeds = data['keys'], data['seeds']
    index = int(np.searchsorted(keys, needle))
    if index >= len(keys) or keys[index] != needle:
        raise KeyError('Observation identity is outside the paper manifests')
    if anchor is not None:
        first, last = data['offsets'][index:index + 2]
        if not 0 <= anchor < last - first:
            raise IndexError('Anchor is outside the observation identity')
        index = int(first) + anchor
    return int(seeds[index])

def observation_seed(key):
    if len(key) == 4 and key[0] in ('nyu', 'ade'):
        return _seed('dense_observation', key[:3], key[3])
    return _seed('observation', key)

def destroyed_seed(key, anchor):
    return _seed('destroyed', key, anchor)
