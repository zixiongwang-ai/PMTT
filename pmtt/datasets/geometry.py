"""Camera-aware MegaDepth and ScanNet depth and correspondence adapters."""
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image
from pmtt.settings import data_root
SCIENCE={"depth_consistency":0.05}

def anchor_grid(size):
    w, h = size
    nx, ny = (8, 4) if w >= h else (4, 8)
    x = np.linspace(0.08 * w, 0.92 * w, nx)
    y = np.linspace(0.08 * h, 0.92 * h, ny)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx.ravel(), yy.ravel()], 1)

def resize_spec(size):
    w, h = size
    r = 640 / max(w, h)
    rw = max(16, round(w * r / 16) * 16)
    rh = max(16, round(h * r / 16) * 16)
    return dict(original=[w, h], resized=[rw, rh], scale=[rw / w, rh / h], grid=[rh // 16, rw // 16])

def grid_xy(spec):
    gh, gw = spec['grid']
    x, y = np.meshgrid((np.arange(gw) + 0.5) * 16, (np.arange(gh) + 0.5) * 16)
    return np.stack([x.ravel(), y.ravel()], 1)

def normalize_points(xy, K):
    return (np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(K).T)[:, :2]

def rotation_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

@lru_cache(32)
def image_size(ds, name):
    with Image.open(image_path(ds, name)) as im:
        im.load()
        if im.getexif().get(274, 1) != 1:
            raise ValueError('Unresolved EXIF orientation')
        return im.size

def image_path(ds, name):
    return data_root(ds) / ('images' if ds == 'megadepth' else '') / name

@lru_cache(24)
def depth(ds, name):
    root = data_root(ds)
    if ds == 'megadepth':
        a = hdf_depth(root / 'depths' / Path(name).with_suffix('.h5'))
    else:
        a = np.asarray(Image.open(root / name.replace('/color/', '/depth/').replace('.jpg', '.png')), dtype=np.float64) / 1000.0
    return a

@lru_cache(256)
def scene_camera(scene):
    root = data_root('scannet') / scene / 'intrinsic'
    kd = np.loadtxt(root / 'intrinsic_depth.txt')[:3, :3]
    kc = np.loadtxt(root / 'intrinsic_color.txt')[:3, :3]
    ed = np.loadtxt(root / 'extrinsic_depth.txt')
    ec = np.loadtxt(root / 'extrinsic_color.txt')
    if not np.allclose(ed, np.eye(4), atol=1e-07) or not np.allclose(ec, np.eye(4), atol=1e-07):
        raise ValueError('Nonidentity ScanNet extrinsic requires explicit registration')
    return (kd, kc)

def depth_intrinsics(ds, name, K):
    if ds == 'scannet':
        return scene_camera(name.split('/')[0])[0]
    d = depth(ds, name)
    w, h = image_size(ds, name)
    return np.diag([d.shape[1] / w, d.shape[0] / h, 1]) @ K

def sample_depth(ds, name, K, xy):
    d = depth(ds, name)
    kd = depth_intrinsics(ds, name, K)
    uv = np.c_[normalize_points(xy, K), np.ones(len(xy))] @ kd.T
    ix = np.rint(uv[:, :2]).astype(np.int64)
    valid = (ix[:, 0] >= 0) & (ix[:, 0] < d.shape[1]) & (ix[:, 1] >= 0) & (ix[:, 1] < d.shape[0])
    out = np.zeros(len(xy))
    out[valid] = d[ix[valid, 1], ix[valid, 0]]
    valid &= np.isfinite(out) & (out > 0)
    return (out, valid)

def reproject(row, xy=None):
    ds = row['dataset']
    a, b = (row['source'], row['target'])
    xy = anchor_grid(image_size(ds, a)) if xy is None else np.asarray(xy)
    K1, K2, R, t = [np.array(row[k], float) for k in ('K1', 'K2', 'R', 't')]
    z, v = sample_depth(ds, a, K1, xy)
    xyz = np.c_[normalize_points(xy, K1), np.ones(len(xy))] * z[:, None]
    q = xyz @ R.T + t
    front = q[:, 2] > 1e-08
    uv = q @ K2.T
    uv = uv[:, :2] / np.where(front, q[:, 2], 1)[:, None]
    w, h = image_size(ds, b)
    v &= front & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    tz, tv = sample_depth(ds, b, K2, uv)
    v &= tv & (np.abs(tz - q[:, 2]) / np.maximum(tz, 1e-08) < SCIENCE['depth_consistency'])
    return dict(anchors=xy.tolist(), truth=uv.tolist(), valid=v.tolist(), projected_depth=q[:, 2].tolist(), target_depth=tz.tolist())

def parse_pairs(ds):
    rows = []
    root = data_root(ds)
    for j, line in enumerate((root / 'pairs_calibrated.txt').read_text().splitlines()):
        tokens = line.split()
        a, b = tokens[:2]
        v = np.asarray(tokens[2:], float)
        if len(v) != (30 if ds == 'megadepth' else 34):
            raise ValueError('Unknown calibrated pair schema')
        K1 = v[:9].reshape(3, 3)
        K2 = v[9:18].reshape(3, 3)
        if ds == 'megadepth':
            R = v[18:27].reshape(3, 3)
            t = v[27:30]
        else:
            T = v[18:].reshape(4, 4)
            R = T[:3, :3]
            t = T[:3, 3]
            if not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError('Invalid homogeneous pose')
        rows.append(dict(dataset=ds, id=f'{j:04d}', source=a, target=b, K1=K1.tolist(), K2=K2.tolist(), R=R.tolist(), t=t.tolist()))
    if len(rows) != 1500:
        raise ValueError('Expected 1500 official pair rows')
    if len({(r['source'], r['target']) for r in rows}) != len(rows):
        raise ValueError('Duplicate pairs')
    return rows

@lru_cache(1)
def md_views():
    out = {}
    for line in (data_root('megadepth') / 'views.txt').read_text().splitlines():
        v = line.split()
        if v[13] != 'PINHOLE':
            raise ValueError('Unsupported distortion model')
        T = np.eye(4)
        T[:3, :3] = np.array(v[1:10], float).reshape(3, 3)
        T[:3, 3] = np.array(v[10:13], float)
        out[v[0]] = (T, [int(v[14]), int(v[15])], np.array([[float(v[16]), 0, float(v[18])], [0, float(v[17]), float(v[19])], [0, 0, 1]]))
    return out

def inspect_pair(row):
    ds = row['dataset']
    a, b = (row['source'], row['target'])
    try:
        K1, K2, R, t = [np.array(row[k], float) for k in ('K1', 'K2', 'R', 't')]
        if not all((np.isfinite(x).all() for x in (K1, K2, R, t))):
            raise ValueError('Non-finite calibration')
        if min(K1[0, 0], K1[1, 1], K2[0, 0], K2[1, 1]) <= 0 or np.linalg.norm(t) < 1e-08:
            raise ValueError('Degenerate camera')
        if not np.allclose(R.T @ R, np.eye(3), atol=5e-05) or np.linalg.det(R) < 0.999:
            raise ValueError('Invalid rotation')
        sizes = [image_size(ds, n) for n in (a, b)]
        if ds == 'megadepth':
            views = md_views()
            T1, s1, k1 = views[a]
            T2, s2, k2 = views[b]
            T = T2 @ np.linalg.inv(T1)
            if tuple(s1) != sizes[0] or tuple(s2) != sizes[1]:
                raise ValueError('Camera/image dimension mismatch')
        else:
            root = data_root(ds)
            T1 = np.loadtxt(root / a.replace('/color/', '/pose/').replace('.jpg', '.txt'))
            T2 = np.loadtxt(root / b.replace('/color/', '/pose/').replace('.jpg', '.txt'))
            T = np.linalg.inv(T2) @ T1
            k1 = scene_camera(a.split('/')[0])[1]
            k2 = scene_camera(b.split('/')[0])[1]
        if not np.allclose(R, T[:3, :3], atol=0.0003) or not np.allclose(t, T[:3, 3], atol=0.0003):
            raise ValueError('Pair/absolute pose convention mismatch')
        if not np.allclose(K1, k1, atol=0.02) or not np.allclose(K2, k2, atol=0.02):
            raise ValueError('Pair/intrinsic mismatch')
        gt = reproject(row)
        count = sum(gt['valid'])
        return dict(
            **row,
            gt=gt,
            size1=list(sizes[0]),
            size2=list(sizes[1]),
            rotation=rotation_angle(R),
            covisibility=count / 32,
            valid_count=count,
            eligible=count >= 8,
            error=None,
        )
    except Exception as e:
        return dict(**row, eligible=False, error=repr(e), valid_count=0)

def hdf_depth(path):
    import h5py
    with h5py.File(path, 'r') as handle:
        dataset = handle['depth']
        if dataset.ndim != 2:
            raise ValueError('Expected depth[H,W]')
        return np.asarray(dataset, dtype=np.float64)
