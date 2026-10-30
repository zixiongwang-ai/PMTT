"""Paper evaluation and complete train-side fitting from official datasets."""
import copy
from pathlib import Path
import numpy as np
import torch
from .settings import ASSETS, SCALES, REPS
from .assets import load_reader_parameters, csv_rows
from .io import save, load
from .datasets.preparation import prepared_manifest

def install_readers(output, task):
    for rep, record in load_reader_parameters(task).items():
        save(record, Path(output) / task / 'coefficients' / (rep + '.json'))

def paper_spair_features(output, scale):
    from .backbone import Backbone
    from . import spair
    backbone = Backbone(scale)
    manifest = prepared_manifest(output, 'spair')
    for item in manifest['splits']['trn'][:2] + manifest['splits']['test']:
        annotation = item['annotation']
        spair.source(output, backbone, annotation)
        spair.target(output, backbone, annotation['category'], annotation['trg_imname'])

def fit_geometry(output, task):
    from .backbone import Backbone
    from .wide import source, target
    from .representation import gpu
    from .features import primitives
    from .candidates import candidate_bank
    from .geometry_fit import fit_dataset
    from .readers import design
    from .convex_fit import statistics
    rows = [r for r in prepared_manifest(output, task)['rows'] if r['calibration_overlap']]
    for scale in ('B', 'L', 'S', 'H+'):
        backbone = Backbone(scale)
        for row in rows:
            x = gpu(source(output, backbone, task, row['source'])['x'])
            t = target(output, backbone, task, row['target'])
            bank = primitives(x, t['features'].cuda(), chunk=512)
            path = Path(output) / task / 'candidates' / (row['id'] + '.pt')
            if scale == 'B':
                ix, weights = candidate_bank(bank['correct'][..., 0].cpu().numpy(), row['gt']['truth'], t['spec'], (task, row['id']))
                save(dict(indices=torch.tensor(ix), weights=torch.tensor(weights, dtype=torch.float32)), path)
            record = dict(
                bank={key: value.cpu() for key, value in bank.items()},
                spec=t['spec'],
            )
            path = (
                Path(output)
                / task
                / 'calibration'
                / scale
                / (row['id'] + '.pt')
            )
            save(record, path)
        del backbone
    fitted = fit_dataset(output, task)
    save(fitted['pose'], Path(output) / task / 'pose_threshold.json')
    for scale in SCALES:
        records = copy.deepcopy(fitted['readers'])
        if scale not in ('B', 'L'):
            pieces = []
            for row in rows:
                bank = load(Path(output) / task / 'calibration' / scale / (row['id'] + '.pt'))['bank']['correct']
                indices = load(Path(output) / task / 'candidates' / (row['id'] + '.pt'))['indices']
                valid = torch.tensor(row['gt']['valid'])
                pieces.append(torch.gather(bank, 1, indices[..., None].expand(-1, -1, 18))[valid])
            bank = torch.cat(pieces).cuda()
            quantiles = torch.tensor(
                [index / 18 for index in range(1, 18)],
                device='cuda',
                dtype=torch.float64,
            )
            knots = torch.quantile(
                bank[..., 0].double().flatten(),
                quantiles,
            ).tolist()
            for rep in REPS:
                x = design({'correct': bank}, rep, knots)
                mask = torch.ones(
                    x.shape[:2],
                    device='cuda',
                    dtype=torch.bool,
                )
                records[rep]['scales'][scale] = dict(
                    knots=knots,
                    statistics=statistics(x, mask),
                )
        save(records, Path(output) / task / 'coefficients' / (scale + '.json'))

def run_task(task, output, mode='paper', scales=SCALES):
    from . import spair, davis, wide, reader_fit, dense_fit, dense_evaluation
    if mode == 'refit' and tuple(scales) != SCALES:
        raise ValueError('Full refit uses all four scales; omit --scale')
    if task in ('spair', 'davis'):
        module = spair if task == 'spair' else davis
        if mode == 'paper':
            install_readers(output, task)
            if task == 'davis':
                protocol = csv_rows(ASSETS / 'manifests/fitting/davis_train_protocol.csv')
                validation_sequences = [
                    row['sequence']
                    for row in protocol
                    if row['selection_split'] == 'validation'
                ]
                save(
                    dict(train_validation=validation_sequences),
                    Path(output) / 'davis/train_protocol.json',
                )
            for scale in scales:
                if task == 'spair':
                    paper_spair_features(output, scale)
                else:
                    module.extract(output, scale, paper=True)
                module.evaluate(output, scale)
        else:
            for scale in SCALES:
                module.extract(output, scale)
            reader_fit.fit(output, task)
            validation = {scale: module.evaluate(output, scale, validation=True) for scale in SCALES}
            reader_fit.select(output, task, validation)
            if task == 'davis':
                reader_fit.fit(output, task, all_train=True)
            for scale in SCALES:
                module.evaluate(output, scale)
    elif task in ('megadepth', 'scannet'):
        if mode == 'refit':
            fit_geometry(output, task)
        for scale in scales:
            wide.run(output, task, scale, mode=mode)
    else:
        if mode == 'refit':
            for scale in SCALES:
                dense_fit.axis(output, task, scale)
                dense_fit.calibrate(output, task, scale)
            dense_fit.mixture_fit(output, task)
        for scale in scales:
            dense_evaluation.evaluate(output, task, scale, mode)
