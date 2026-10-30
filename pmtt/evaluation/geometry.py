"""Deterministic assignment, local refinement and resized-pixel correspondence."""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from pmtt.io import seed
from pmtt.datasets.geometry import grid_xy, normalize_points, resize_spec, rotation_angle
from .pose import estimate

SCIENCE = {"refinement_temperature": 1.0}

def assignment(scores, spec):
    """Assign one target cell per source and refine within its fixed 3-by-3 resized-pixel neighborhood."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite score matrix')
    ri, ci = linear_sum_assignment(-scores)
    gh, gw = spec['grid']
    coords = grid_xy(spec)
    refined = []
    for a, c in zip(ri, ci):
        y, x = divmod(int(c), gw)
        neighborhood = [
            yy * gw + xx
            for yy in range(max(0, y - 1), min(gh, y + 2))
            for xx in range(max(0, x - 1), min(gw, x + 2))
        ]
        nb = np.array(neighborhood)
        logits = scores[a, nb] / SCIENCE['refinement_temperature']
        w = np.exp(logits - logits.max())
        w /= w.sum()
        refined.append(w @ coords[nb])
    sorted_scores = np.sort(scores, axis=1)
    chosen = scores[ri, ci]
    rank = (scores[ri] > chosen[:, None]).sum(1) + 1
    return dict(
        source=ri,
        target=ci,
        refined=np.asarray(refined),
        raw_score=chosen,
        best=sorted_scores[ri, -1],
        second=sorted_scores[ri, -2],
        margin=sorted_scores[ri, -1] - sorted_scores[ri, -2],
        assignment_rank=rank,
    )

def pose_errors(pred, row):
    if pred['status'] != 'success':
        return dict(rotation_error=180.0, translation_error=180.0, pose_error=180.0)
    R = np.asarray(row['R'])
    t = np.asarray(row['t'])
    pr = np.asarray(pred['R'])
    pt = np.asarray(pred['t'])
    re = rotation_angle(pr @ R.T)
    cosine = abs(float(pt @ t)) / (np.linalg.norm(pt) * np.linalg.norm(t))
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return dict(rotation_error=re, translation_error=float(angle), pose_error=max(re, float(angle)))

def relative_pose(xy1, xy2, row, spec, threshold):
    """Estimate relative pose from original source and resized target pixel coordinates."""
    K1, K2 = (np.asarray(row['K1']), np.asarray(row['K2']))
    p1 = normalize_points(xy1, K1)
    p2 = normalize_points(np.asarray(xy2) / spec['scale'], K2)
    ss = resize_spec(row['size1'])['scale']
    ts = spec['scale']
    focal = np.mean(
        [
            K1[0, 0] * ss[0],
            K1[1, 1] * ss[1],
            K2[0, 0] * ts[0],
            K2[1, 1] * ts[1],
        ]
    )
    pose_seed = seed(row['dataset'], row['id'], 'pose')
    pred = estimate(p1, p2, threshold / focal, pose_seed)
    return dict(
        **pred,
        **pose_errors(pred, row),
        threshold_pixels=threshold,
        threshold_normalized=threshold / focal,
    )

def epipolar_error(xy1, xy2, row, spec):
    R, t = (np.asarray(row['R']), np.asarray(row['t']))
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    x = np.c_[normalize_points(xy1, np.asarray(row['K1'])), np.ones(len(xy1))]
    y = np.c_[normalize_points(np.asarray(xy2) / spec['scale'], np.asarray(row['K2'])), np.ones(len(xy2))]
    ex = x @ E.T
    ety = y @ E
    val = np.sum(y * ex, 1) ** 2
    first = 1 / np.maximum(np.sum(ex[:, :2] ** 2, 1), 1e-20)
    second = 1 / np.maximum(np.sum(ety[:, :2] ** 2, 1), 1e-20)
    return val * (first + second)

def evaluate_scores(scores, row, spec, threshold, condition, scale):
    """Evaluate one source-target score matrix with correspondence, epipolar and pose metrics."""
    start = time.perf_counter()
    m = assignment(scores, spec)
    assignment_seconds = time.perf_counter() - start
    xy = np.asarray(row['gt']['anchors'])[m['source']]
    valid = np.asarray(row['gt']['valid'])[m['source']]
    truth = np.asarray(row['gt']['truth'])[m['source']] * spec['scale']
    errors = np.linalg.norm(m['refined'] - truth, axis=1)
    good5 = (errors <= 5) & valid
    good10 = (errors <= 10) & valid
    nvalid = int(valid.sum())
    t = time.perf_counter()
    pose = relative_pose(xy, m['refined'], row, spec, threshold)
    pose_seconds = time.perf_counter() - t
    epi = epipolar_error(xy, m['refined'], row, spec)
    gt_valid_anchors = int(sum(row['gt']['valid']))
    metrics = dict(
        source_anchors=len(scores),
        gt_valid_anchors=gt_valid_anchors,
        assigned_matches=len(xy),
        evaluable_matches=nvalid,
        precision5=float(good5.sum() / max(nvalid, 1)),
        precision10=float(good10.sum() / max(nvalid, 1)),
        recall5=float(good5.sum() / max(gt_valid_anchors, 1)),
        recall10=float(good10.sum() / max(gt_valid_anchors, 1)),
        mean_pixel_error=(
            float(errors[valid].mean()) if nvalid else None
        ),
        median_pixel_error=(
            float(np.median(errors[valid])) if nvalid else None
        ),
        epipolar_error_median=float(np.median(epi)),
        inlier_count=sum(pose['inliers']),
        inlier_ratio=float(np.mean(pose['inliers'])),
        margin_mean=float(m['margin'].mean()),
        margin_median=float(np.median(m['margin'])),
        pose_error=pose['pose_error'],
        rotation_error=pose['rotation_error'],
        translation_error=pose['translation_error'],
        success=int(pose['status'] == 'success'),
        failure=int(pose['status'] != 'success'),
    )
    match = {k: v.tolist() for k, v in m.items()}
    match.update(
        gt_valid=valid.tolist(),
        pixel_errors=[
            float(error) if is_valid else None
            for error, is_valid in zip(errors, valid)
        ],
        epipolar_errors=epi.tolist(),
        source_coordinates=xy.tolist(),
    )
    return dict(
        dataset=row['dataset'],
        pair_id=row['id'],
        scale=scale,
        condition=condition,
        difficulty=row['difficulty'],
        split=row['split'],
        metrics=metrics,
        matches=match,
        pose=pose,
        assignment_seconds=assignment_seconds,
        pose_seconds=pose_seconds,
    )

def auc(errors, thresholds=(5, 10, 20)):
    errors = np.sort(np.asarray(errors, float))
    recall = np.arange(1, len(errors) + 1) / max(len(errors), 1)
    x = np.r_[0, errors]
    y = np.r_[0, recall]
    result = {}
    for th in thresholds:
        i = np.searchsorted(x, th, side='right')
        xx = np.r_[x[:i], th]
        yy = np.r_[y[:i], y[i - 1]]
        result['auc' + str(th)] = float(np.trapz(yy, xx) / th)
    return result

def aggregate(records):
    if not records:
        raise ValueError('Empty scientific aggregation')
    keys = records[0]['metrics'].keys()
    out = {}
    for k in keys:
        v = [r['metrics'][k] for r in records if r['metrics'][k] is not None]
        out[k] = float(np.mean(v)) if v else None
    pe = [r['metrics']['pose_error'] for r in records]
    out.update(auc(pe))
    out['median_pose_error'] = float(np.median(pe))
    out['pairs'] = len(records)
    return out

def oracle(row, spec, threshold=1.0):
    valid = np.asarray(row['gt']['valid'])
    xy = np.asarray(row['gt']['anchors'])[valid]
    truth = np.asarray(row['gt']['truth'])[valid] * spec['scale']
    return dict(pair_id=row['id'], dataset=row['dataset'], valid=int(valid.sum()), **relative_pose(xy, truth, row, spec, threshold))
