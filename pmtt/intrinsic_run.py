"""Separate Original and Independent Holdout Cohorts with paired scale statistics."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .assets import load_cohort_manifest
from .settings import ASSETS, SCALES, data_root
from .backbone import Backbone
from .observations import group
from .intrinsic import encode, evaluate, summary, paired
from .io import load, save

COHORTS = ('original', 'independent_holdout')

def run(output, scales=SCALES, mode='paper'):
    frozen = load(ASSETS / 'calibration/intrinsic.json')
    for cohort in COHORTS:
        rows, observations = load_cohort_manifest(cohort)
        for scale in scales:
            backbone = Backbone(scale)
            states = []
            for index, row in enumerate(rows):
                with Image.open(data_root('imagenet') / row['image_id']) as source:
                    image = source.convert('RGB')
                views = group(
                    image,
                    row['image_id'],
                    visible_count=16,
                    hidden_count=16,
                    purpose='intrinsic',
                    packed=True,
                    observation_seed=int(row['observation_seed']),
                )
                expected = observations['crops'][index]
                if not np.array_equal(views['crops'].numpy(), expected):
                    raise ValueError('Orbital crop identity differs for ' + row['image_id'])
                views['positions'] = torch.tensor(observations['positions'][index])
                states.append(encode(backbone, views, 32))
            tensors = {key: torch.cat([s[key] for s in states]).cuda() for key in ('c', 'v', 'h')}
            records = evaluate(tensors, frozen_cap=frozen[cohort][scale]['cap'] if mode == 'paper' else None)
            if mode == 'paper':
                records['score_scale'] = torch.tensor(frozen[cohort][scale]['score_scale'])
            save(dict(records=records), Path(output) / 'intrinsic' / cohort / scale / 'records.pt')
            del backbone, states, tensors, records
    aggregate(output, scales)

def aggregate(output, scales=SCALES):
    results, all_records = {}, {}
    for cohort in COHORTS:
        rows, observations = load_cohort_manifest(cohort)
        records_by_scale, details = {}, {}
        for scale in scales:
            records = load(Path(output) / 'intrinsic' / cohort / scale / 'records.pt')['records']
            records_by_scale[scale] = records
            details[scale] = summary(records)
            point = records['point'].sum(-1).double()
            valid = point > 1e-12
            for name in ('observed', 'random', 'donor', 'shared'):
                delta = (
                    (
                        records[name].sum(-1)
                        - records['manifold'].sum(-1)
                    ).double()
                    / point.clamp_min(1e-12)
                )[valid]
                details[scale]['manifold-' + name].update(paired(delta.numpy()))
            metadata = torch.tensor(observations['metadata'][:, 16:])
            direction_sequence = [0, 2, 4, 6, 1, 3, 5, 7] * 2
            masks = {
                f'sector_{sector}': torch.tensor(
                    [sector == value for value in direction_sequence]
                )[None].expand(len(rows), -1)
                for sector in range(8)
            }
            for field, column in [('context', 4), ('aspect', 5)]:
                masks[field + '_smaller'] = metadata[..., column] < 0
                masks[field + '_larger'] = metadata[..., column] >= 0
            magnitudes = [
                (
                    'visible',
                    records['visible_magnitude'][:, None].expand_as(records['point']),
                ),
                ('hidden', records['hidden_magnitude']),
            ]
            for field, magnitude in magnitudes:
                quantiles = magnitude.quantile(torch.tensor([.25, .5, .75]))
                labels = torch.bucketize(magnitude.contiguous(), quantiles)
                for q in range(4):
                    masks[field + f'_quartile{q+1}'] = labels == q
            extreme = torch.tensor(
                [row['exception_applied'] == 'True' for row in rows]
            )
            masks['extreme'] = extreme[:, None].expand(len(rows), 16)
            details[scale]['strata'] = {}
            for key, mask in masks.items():
                denominator = records['point'][mask].double().sum()
                details[scale]['strata'][key] = {
                    condition: (
                        float(
                            1
                            - records[condition][mask].double().sum()
                            / denominator
                        )
                        if denominator > 0
                        else None
                    )
                    for condition in (
                        'manifold',
                        'observed',
                        'random',
                        'donor',
                        'shared',
                        'full32',
                    )
                }
        comparisons = {}
        for low, high in zip(scales[:-1], scales[1:]):
            a, b = records_by_scale[high], records_by_scale[low]
            valid = (a['point'].sum(-1) > 1e-12) & (b['point'].sum(-1) > 1e-12)
            def coverage(record, condition):
                return 1 - record[condition].sum(-1) / record['point'].sum(-1).clamp_min(1e-12)
            comparison = {
                condition: paired(
                    (coverage(a, condition) - coverage(b, condition))[valid].numpy()
                )
                for condition in (
                    'manifold',
                    'observed',
                    'random',
                    'donor',
                    'shared',
                )
            }
            organization_delta = (
                coverage(a, 'manifold')
                - coverage(a, 'observed')
                - coverage(b, 'manifold')
                + coverage(b, 'observed')
            )
            comparison['organization_difference_of_differences'] = paired(
                organization_delta[valid].numpy()
            )
            comparisons[high + '-' + low] = comparison
        results[cohort] = dict(scales=details, paired_scales=comparisons)
        all_records[cohort] = records_by_scale
    conditions = ('manifold', 'observed', 'random', 'donor', 'shared')
    results['replication'] = {
        scale: {
            condition: (
                results['independent_holdout']['scales'][scale][condition]['pooled']
                - results['original']['scales'][scale][condition]['pooled']
            )
            for condition in conditions
        }
        for scale in scales
    }
    results['pooled_secondary'] = {}
    for scale in scales:
        point_error = sum(
            all_records[cohort][scale]['point'].double().sum()
            for cohort in COHORTS
        )
        results['pooled_secondary'][scale] = {
            condition: float(
                1
                - sum(
                    all_records[cohort][scale][condition].double().sum()
                    for cohort in COHORTS
                )
                / point_error
            )
            for condition in conditions
        }
    save(results, Path(output) / 'intrinsic/statistics.json')
