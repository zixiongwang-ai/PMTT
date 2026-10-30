"""Natural-unit aggregation and paired cluster bootstrap."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from .io import load, save, unpack
from .settings import SCALES, REPS, conditions
from .dense import seg_metrics


def summary_value(a, kind):
    a = np.asarray(a, float)
    if kind == 'ratio':
        return float(a[:, 0].sum() / max(a[:, 1].sum(), 1))
    if kind == 'iou':
        t = a.sum(0)
        diag, row, col = np.split(t, 3)
        den = row + col - diag
        v = den > 0
        return float(np.mean(diag[v] / den[v]))
    return float(a[:, 0].mean())

def resample_value(a, weights, kind):
    v = weights @ a
    if kind == 'ratio':
        return v[:, 0] / np.maximum(v[:, 1], 1)
    if kind == 'iou':
        diag, row, col = np.split(v, 3, axis=1)
        den = row + col - diag
        valid = den > 0
        ratios = np.divide(
            diag,
            den,
            out=np.zeros_like(diag),
            where=valid,
        )
        return ratios.sum(1) / valid.sum(1).clip(1)
    return v[:, 0] / weights.sum(1)

def paired(a, b, kind, clusters=None, draws=2000):
    """Bootstrap paired natural units, preserving ratios or additive confusion estimands.
    
    Optional clusters aggregate pairs or images before resampling; draws default to 2000."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape or not len(a):
        raise ValueError('Unpaired observations')
    if clusters is not None:
        names = sorted(set(clusters))
        ix = {k: j for j, k in enumerate(names)}
        aa = np.zeros((len(names), a.shape[1]))
        bb = np.zeros_like(aa)
        for j, name in enumerate(clusters):
            aa[ix[name]] += a[j]
            bb[ix[name]] += b[j]
        if kind == 'mean':
            counts = np.bincount([ix[c] for c in clusters])
            aa = np.c_[aa[:, 0], counts]
            bb = np.c_[bb[:, 0], counts]
            kind = 'ratio'
        a, b = (aa, bb)
    rng = np.random.default_rng(20261025)
    samples = []
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        weights = rng.multinomial(
            len(a),
            np.full(len(a), 1 / len(a)),
            size=size,
        )
        sampled_delta = (
            resample_value(a, weights, kind)
            - resample_value(b, weights, kind)
        )
        samples.extend(sampled_delta.tolist())
    per = np.array(
        [
            summary_value(a[index:index + 1], kind)
            - summary_value(b[index:index + 1], kind)
            for index in range(len(a))
        ]
    )
    return dict(
        delta=summary_value(a, kind) - summary_value(b, kind),
        interval=np.quantile(samples, [0.025, 0.975]).tolist(),
        unit_delta_median=float(np.median(per)),
        improved_fraction=float(np.mean(per > 0)),
        standardized_effect=float(per.mean() / max(per.std(), 1e-12)),
        units=len(a),
        draws=draws,
    )

def units(base, ds, scale):
    root = Path(base) / ds / 'records' / scale
    out = {}
    kind = 'mean'
    metadata = {}
    if ds == 'spair':
        kind = 'ratio'
        for p in sorted(root.glob('*.json')):
            r = load(p)
            out[r['id']] = {c: [v['good'], v['count']] for c, v in r['conditions'].items()}
            metadata[r['id']] = dict(category=r['category'], **r.get('strata', {}))
    elif ds == 'davis':
        kind = 'ratio'
        object_counts = {}
        seq = defaultdict(lambda: defaultdict(list))
        for p in sorted(root.glob('*/*.json')):
            r = load(p)
            for c, objs in r['conditions'].items():
                seq[r['sequence']][c].append(np.mean([v['JF'] for v in objs.values()]))
                object_counts[r['sequence']] = len(objs)
        for name, vals in seq.items():
            out[name] = {
                condition: [
                    float(np.mean(values)) * object_counts[name],
                    object_counts[name],
                ]
                for condition, values in vals.items()
            }
            metadata[name] = {}
    elif ds in ('megadepth', 'scannet'):
        for p in sorted(root.glob('*.json')):
            r = load(p)
            out[r['id']] = {c: [v['metrics']['precision10']] for c, v in r['conditions'].items()}
            metadata[r['id']] = {k: r[k] for k in ('scene', 'calibration_overlap', 'difficulty')}
    elif ds == 'nyu':
        for p in sorted(root.glob('*.ptz')):
            r = unpack(p)
            out[r['id']] = {c: [v['score']] for c, v in r['conditions'].items()}
            metadata[r['id']] = dict(scene=r['scene'])
    else:
        kind = 'iou'
        for p in sorted(root.glob('*.ptz')):
            r = unpack(p)
            vals = {}
            for c, v in r['conditions'].items():
                cm = v['cm'].numpy()
                vals[c] = np.r_[cm.diagonal(), cm.sum(1), cm.sum(0)].tolist()
            out[r['id']] = vals
            metadata[r['id']] = dict(scene=r['scene'])
    return (out, kind, metadata)

def interaction(a, b, c, d, kind, draws):
    rng = np.random.default_rng(20261025)
    samples = []
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        weights = rng.multinomial(
            len(a),
            np.full(len(a), 1 / len(a)),
            size=size,
        )
        sampled = (
            resample_value(a, weights, kind)
            - resample_value(b, weights, kind)
            - resample_value(c, weights, kind)
            + resample_value(d, weights, kind)
        )
        samples.extend(sampled.tolist())
    delta = (
        summary_value(a, kind)
        - summary_value(b, kind)
        - summary_value(c, kind)
        + summary_value(d, kind)
    )
    return dict(
        delta=delta,
        interval=np.quantile(samples, [0.025, 0.975]).tolist(),
    )

def augment(base, ds, report):
    from .statistics import units
    arrays = {}
    ids0 = None
    kind = None
    meta = None
    for scale in SCALES:
        u, kind, meta = units(base, ds, scale)
        ids = sorted(u)
        if ids0 is None:
            ids0 = ids
        elif ids != ids0:
            raise ValueError('Scale identities not paired')
        valid = [k for k in ids if not meta[k].get('calibration_overlap', False)]
        arrays[scale] = {rep: np.asarray([u[k][rep] for k in valid], float) for rep in REPS}
        root = Path(base) / ds / 'records' / scale
        if ds in ('megadepth', 'scannet'):
            from .evaluation.geometry import aggregate, auc
            rows = [load(p) for p in sorted(root.glob('*.json'))]
            for row in rows:
                for value in row['conditions'].values():
                    epipolar_errors = np.asarray(
                        value['matches']['epipolar_errors']
                    )
                    value['metrics']['epipolar_precision_1e4'] = float(
                        np.mean(epipolar_errors < 0.0001)
                    )
            secondary = {
                condition: aggregate(
                    [row['conditions'][condition] for row in rows]
                )
                for condition in conditions(ds, scale)
            }
            by_difficulty = {}
            difficulties = sorted({row['difficulty'] for row in rows})
            for difficulty in difficulties:
                selected = [
                    row for row in rows if row['difficulty'] == difficulty
                ]
                by_difficulty[difficulty] = {
                    condition: aggregate(
                        [row['conditions'][condition] for row in selected]
                    )
                    for condition in conditions(ds, scale)
                }
            pose = {}
            if scale in ('B', 'H+'):
                prs = [load(p) for p in sorted((Path(base) / ds / 'pose64' / scale).glob('*.json'))]
                for cond in REPS:
                    err = [r['conditions'][cond]['metrics']['pose_error'] for r in prs]
                    failures = [
                        row['conditions'][cond]['metrics']['failure']
                        for row in prs
                    ]
                    pose[cond] = dict(
                        **auc(err),
                        median_pose_error=float(np.median(err)),
                        failure_rate=float(np.mean(failures)),
                        pairs=len(prs),
                        role='pose_only_supplementary',
                    )
            invalid_anchor_pairs = sum(
                row['conditions']['Center']['metrics']['gt_valid_anchors'] == 0
                for row in rows
            )
            report['scales'][scale].update(
                secondary=secondary,
                by_difficulty=by_difficulty,
                pose64=pose,
                invalid_anchor_pairs=invalid_anchor_pairs,
            )
        elif ds == 'spair':
            rows = [load(p) for p in sorted(root.glob('*.json'))]
            secondary = {}
            strata = {}
            for cond in conditions(ds, scale):
                ranks = np.concatenate([r['conditions'][cond]['ranks'] for r in rows])
                secondary[cond] = dict(mean_rank=float(ranks.mean()), median_rank=float(np.median(ranks)), MRR=float((1 / ranks).mean()))
            for field in ('category', 'viewpoint_variation', 'scale_variation', 'occlusion', 'truncation'):
                groups = defaultdict(list)
                for r in rows:
                    groups[str(r['category'] if field == 'category' else r['strata'][field])].append(r)
                strata[field] = {}
                for key, selected_rows in groups.items():
                    strata[field][key] = {}
                    for condition in conditions(ds, scale):
                        good = sum(
                            row['conditions'][condition]['good']
                            for row in selected_rows
                        )
                        count = sum(
                            row['conditions'][condition]['count']
                            for row in selected_rows
                        )
                        strata[field][key][condition] = good / count
            categories = sorted(strata['category'])
            macro = {}
            for hi, lo in [('Observed', 'Center'), ('Manifold', 'Observed'), ('Manifold', 'Center')]:
                a = np.array([[strata['category'][c][hi]] for c in categories])
                b = np.array([[strata['category'][c][lo]] for c in categories])
                macro[hi + '-' + lo] = paired(a, b, 'mean', draws=2000)
            report['scales'][scale].update(secondary=secondary, strata=strata, category_macro_paired=macro)
        elif ds == 'davis':
            rows = [load(p) for p in sorted(root.glob('*/*.json'))]
            objects = defaultdict(lambda: defaultdict(list))
            temporal = defaultdict(lambda: defaultdict(list))
            for r in rows:
                bucket = 'short' if r['temporal_fraction'] <= 1 / 3 else 'medium' if r['temporal_fraction'] <= 2 / 3 else 'long'
                for cond, os in r['conditions'].items():
                    for obj, v in os.items():
                        objects[cond][r['sequence'] + '/' + obj].append(v)
                        temporal[cond][bucket].append(v)
            secondary = {}
            for cond, obs in objects.items():
                metric_names = ('J', 'F', 'JF', 'boundary_J', 'interior_J')
                secondary[cond] = {
                    metric: float(
                        np.mean(
                            [
                                np.mean([value[metric] for value in values])
                                for values in obs.values()
                            ]
                        )
                    )
                    for metric in metric_names
                }
                secondary[cond]['objects'] = {
                    name: {
                        metric: float(
                            np.mean([value[metric] for value in values])
                        )
                        for metric in metric_names
                    }
                    for name, values in obs.items()
                }
                secondary[cond]['temporal'] = {
                    bucket: {
                        metric: float(
                            np.mean([value[metric] for value in values])
                        )
                        for metric in metric_names
                    }
                    for bucket, values in temporal[cond].items()
                }
            report['scales'][scale]['secondary'] = secondary
        elif ds == 'nyu':
            rows = [unpack(p) for p in sorted(root.glob('*.ptz'))]
            secondary = {}
            for cond in conditions(ds, scale):
                metrics = [r['conditions'][cond]['metrics'] for r in rows]
                metric_names = (
                    'AbsRel',
                    'SqRel',
                    'RMSE',
                    'log_RMSE',
                    'delta1',
                    'delta2',
                    'delta3',
                )
                secondary[cond] = {
                    key: float(np.mean([value[key] for value in metrics]))
                    for key in metric_names
                }
                secondary[cond]['strata'] = {}
                for stratum in ('near', 'mid', 'far', 'boundary', 'smooth'):
                    valid_metrics = [
                        value
                        for value in metrics
                        if value['strata'][stratum]['count']
                    ]
                    secondary[cond]['strata'][stratum] = {
                        key: (
                            float(
                                np.mean(
                                    [
                                        value['strata'][stratum][key]
                                        for value in valid_metrics
                                    ]
                                )
                            )
                            if valid_metrics
                            else None
                        )
                        for key in metric_names
                    }
            report['scales'][scale]['secondary'] = secondary
        elif ds == 'ade':
            sums = {}
            axis_path = Path(base) / ds / 'axis/S/axis.pt'
            from .assets import load_task_axis
            freq = (load(axis_path) if axis_path.exists() else load_task_axis(ds, 'S')).get('class_frequency')
            for p in sorted(root.glob('*.ptz')):
                r = unpack(p)
                for cond, v in r['conditions'].items():
                    if cond not in sums:
                        sums[cond] = {key: np.zeros((150, 150), np.int64) for key in ('cm', 'boundary_cm', 'interior_cm')}
                    for key in sums[cond]:
                        sums[cond][key] += v[key].numpy()
            secondary = {
                condition: {
                    key: seg_metrics(confusion_matrix)
                    for key, confusion_matrix in matrices.items()
                }
                for condition, matrices in sums.items()
            }
            if freq is not None:
                order = np.argsort(freq, kind='stable')
                rare = set(order[:75])
                common = set(order[75:])
                for cond, s in secondary.items():
                    pc = s['cm']['per_class']
                    s['rare_common'] = {}
                    for name, indices in [('rare', rare), ('common', common)]:
                        values = [pc[index] for index in indices if pc[index] is not None]
                        s['rare_common'][name] = (
                            float(np.mean(values)) if values else None
                        )
            report['scales'][scale]['secondary'] = secondary
    report['scale_interactions'] = {}
    for lo, hi in zip(SCALES[:-1], SCALES[1:]):
        comparisons = [
            ('Observed', 'Center'),
            ('Manifold', 'Observed'),
            ('Manifold', 'Center'),
        ]
        report['scale_interactions'][hi + '-' + lo] = {
            high + '-' + low: interaction(
                arrays[hi][high],
                arrays[hi][low],
                arrays[lo][high],
                arrays[lo][low],
                kind,
                2000,
            )
            for high, low in comparisons
        }
    return report
