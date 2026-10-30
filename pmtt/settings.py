"""Explicit paths and the numerical conventions of the paper."""
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ASSETS = REPOSITORY / 'assets'
SCALES = ('S', 'B', 'L', 'H+')
REPS = ('Center', 'Observed', 'Manifold')
DATASETS = ('spair', 'davis', 'megadepth', 'scannet', 'nyu', 'ade')
ARCH = {'S': ('vits16', 384, 12, 6), 'B': ('vitb16', 768, 12, 12),
        'L': ('vitl16', 1024, 24, 16), 'H+': ('vith16plus', 1280, 32, 20)}
SCIENCE = dict(l2=[.0001, .001, .01], solver_maxiter=600,
               solver_tolerance=2e-6, solver_ftol=1e-11,
               standardization_eps=1e-6)
_data = {}
_source = None
_weights = {}
_native = None

def configure(data=None, dinov3_source=None, weights=None, native=None):
    global _source, _native
    if data:
        _data.update({k: Path(v).expanduser().resolve() for k, v in data.items() if v})
    if dinov3_source:
        _source = Path(dinov3_source).expanduser().resolve()
    if weights:
        _weights.update({k: Path(v).expanduser().resolve() for k, v in weights.items() if v})
    if native:
        _native = Path(native).expanduser().resolve()

def data_root(task):
    if task not in _data:
        raise ValueError(f'Dataset root required for {task}')
    return _data[task]

def backbone_paths(scale):
    if _source is None or scale not in _weights:
        raise ValueError(f'DINOv3 source and {scale} checkpoint paths are required')
    return _source, _weights[scale]

def native_library():
    if _native:
        return _native
    name = 'pmtt_pose.dll' if sys.platform == 'win32' else ('libpmtt_pose.dylib' if sys.platform == 'darwin' else 'libpmtt_pose.so')
    path = REPOSITORY / 'native' / 'build' / name
    if not path.is_file():
        raise FileNotFoundError('Build the pose bridge and set --native-library')
    return path

def controls(task, scale):
    return scale in ('B', 'H+') or task in ('megadepth', 'scannet')

def conditions(task, scale):
    return list(REPS) + (['DonorObserved', 'DestroyedManifold'] if controls(task, scale) else [])
