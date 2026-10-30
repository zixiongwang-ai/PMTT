"""Aggregate newly evaluated natural units or copy the saved reference tables."""
import csv
import shutil
from pathlib import Path
import numpy as np
from .settings import ASSETS, SCALES, REPS, conditions
from .io import load, save
from .statistics import units, summary_value, paired, augment
from .datasets.preparation import prepared_manifest

METRICS = {'spair': 'PCK@0.1_bbox', 'davis': 'J&F', 'megadepth': 'P@10', 'scannet': 'P@10', 'nyu': '-AbsRel', 'ade': 'mIoU'}

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def dataset_report(output, task, scales=SCALES):
    manifest = prepared_manifest(output, task)
    expected = len(manifest['rows']) if task in ('megadepth', 'scannet') else len(manifest['splits']['val' if task == 'davis' else 'test'])
    result = dict(dataset=task, expected_units=expected, scales={})
    for scale in scales:
        observations, kind, metadata = units(output, task, scale)
        if len(observations) != expected:
            raise ValueError(f'{task}/{scale}: expected {expected} natural units, found {len(observations)}')
        ids = sorted(observations)
        for key in ids:
            if set(observations[key]) != set(conditions(task, scale)):
                raise ValueError('Missing or extra representation conditions')
        values = {c: np.asarray([observations[k][c] for k in ids]) for c in conditions(task, scale)}
        confirm = [i for i, key in enumerate(ids) if not metadata[key].get('calibration_overlap', False)]
        comparisons = {}
        pairs = [('Observed', 'Center'), ('Manifold', 'Observed'), ('Manifold', 'Center')]
        if 'DonorObserved' in values:
            pairs += [('Observed', 'DonorObserved'), ('Manifold', 'DestroyedManifold')]
        for high, low in pairs:
            a, b = values[high][confirm], values[low][confirm]
            comparison = paired(a, b, kind, draws=2000)
            if task in ('megadepth', 'scannet', 'nyu'):
                comparison['scene_sensitivity'] = paired(a, b, kind, [metadata[ids[i]]['scene'] for i in confirm], draws=2000)
            comparison['scope'] = 'untouched' if task in ('megadepth', 'scannet') else 'official evaluation'
            comparisons[high + '-' + low] = comparison
        result['scales'][scale] = dict(
            scores={condition: summary_value(array, kind) for condition, array in values.items()},
            paired=comparisons,
            kind=kind,
            unit_ids=ids,
            valid_units=len(ids),
            calibration_overlap=len(ids) - len(confirm),
        )
        if task == 'spair':
            categories = sorted({metadata[key]['category'] for key in ids})
            result['scales'][scale]['category_macro'] = {
                condition: {
                    category: summary_value(
                        values[condition][
                            [
                                index
                                for index, key in enumerate(ids)
                                if metadata[key]['category'] == category
                            ]
                        ],
                        kind,
                    )
                    for category in categories
                }
                for condition in conditions(task, scale)
            }
    if tuple(scales) == SCALES:
        result = augment(output, task, result)
    save(result, Path(output) / task / 'summary.json')
    return result

def reproduce(output, destination, reference=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if reference:
        for path in (ASSETS / 'reference_results').glob('*.csv'):
            shutil.copyfile(path, destination / path.name)
        return
    primary, effects, control_rows, pose = [], [], [], []
    for task in METRICS:
        report = dataset_report(output, task)
        for scale, result in report['scales'].items():
            for rep in REPS:
                primary.append(
                    dict(
                        dataset=task,
                        scale=scale,
                        representation=rep,
                        score=result['scores'][rep],
                        units=result['valid_units'],
                        metric=METRICS[task],
                    )
                )
            for name, comp in result['paired'].items():
                effects.append(
                    dict(
                        dataset=task,
                        scale=scale,
                        comparison=name,
                        delta=comp['delta'],
                        ci_low=comp['interval'][0],
                        ci_high=comp['interval'][1],
                        scope=comp['scope'],
                        units=comp['units'],
                    )
                )
            for name, score in result['scores'].items():
                if name not in REPS:
                    control_rows.append(dict(dataset=task, scale=scale, condition=name, score=score))
            for rep, value in result.get('pose64', {}).items():
                pose.append(dict(dataset=task, scale=scale, representation=rep, anchors=64, **value))
    write_csv(destination / 'primary_72_cells.csv', primary)
    write_csv(destination / 'paired_effects.csv', effects)
    write_csv(destination / 'control_summary.csv', control_rows)
    write_csv(destination / 'supplementary_pose.csv', pose)
    intrinsic_path = Path(output) / 'intrinsic/statistics.json'
    if intrinsic_path.is_file():
        intrinsic = load(intrinsic_path)
        rows = []
        for cohort in ('original', 'independent_holdout'):
            for scale, value in intrinsic[cohort]['scales'].items():
                rows.append(
                    dict(
                        cohort=cohort,
                        scale=scale,
                        observed_coverage=value['observed']['pooled'],
                        manifold_coverage=value['manifold']['pooled'],
                        full32_coverage=value['full32']['pooled'],
                        effective_rank=value['effective_rank'],
                    )
                )
        write_csv(destination / 'intrinsic_summary.csv', rows)
