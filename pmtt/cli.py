"""Command-line paths for preparation, paper evaluation and full refitting."""
import argparse
import tomllib
from pathlib import Path
from .settings import REPOSITORY, SCALES, DATASETS, configure

def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--config', type=Path, default=REPOSITORY / 'configs/paper_reproduction.toml')
    result.add_argument('--output', type=Path)
    result.add_argument('--dinov3-source', type=Path)
    result.add_argument('--native-library', type=Path)
    for name in ('s', 'b', 'l', 'hplus'):
        result.add_argument('--weights-' + name, type=Path)
    for name in (*DATASETS, 'imagenet'):
        result.add_argument('--' + name + '-root', type=Path)
    return result

def setup(args):
    config = tomllib.loads(args.config.read_text(encoding='utf8'))
    relative_to = args.config.resolve().parent
    def path(value):
        if not value:
            return None
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (relative_to / p).resolve()
    locations = config.get('paths', {})
    roots = {task: getattr(args, task + '_root', None) or path(locations.get(task)) for task in (*DATASETS, 'imagenet')}
    weights = {
        scale: (
            getattr(args, 'weights_' + name, None)
            or path(locations.get('weights_' + name))
        )
        for scale, name in zip(SCALES, ('s', 'b', 'l', 'hplus'))
    }
    configure(
        data=roots,
        weights=weights,
        dinov3_source=(
            args.dinov3_source or path(locations.get('dinov3_source'))
        ),
        native=args.native_library or path(locations.get('native_library')),
    )
    output = args.output.resolve() if args.output else path(locations.get('output', '../outputs/paper'))
    return output

def prepare(task):
    p = parser('Prepare ' + task + ' from official files without modifying them')
    p.add_argument('--data-root', type=Path, required=True)
    args = p.parse_args()
    output = setup(args)
    configure(data={task: args.data_root})
    from .datasets.preparation import prepare_dataset
    print(prepare_dataset(task, output))

def run(task):
    p = parser('PMTT ' + task + ' reproduction')
    p.add_argument('--mode', choices=('paper', 'refit'), default='paper')
    p.add_argument('--scale', choices=SCALES, action='append')
    args = p.parse_args()
    output = setup(args)
    scales = tuple(args.scale) if args.scale else SCALES
    if task == 'intrinsic':
        from .intrinsic_run import run as run_intrinsic
        run_intrinsic(output, scales, args.mode)
    else:
        from .reproduction import run_task
        from .tables import dataset_report
        run_task(task, output, args.mode, scales)
        dataset_report(output, task, scales)

def all_tasks():
    p = parser('Reproduce both intrinsic cohorts and all six downstream tasks')
    p.add_argument('--mode', choices=('paper', 'refit'), default='paper')
    args = p.parse_args()
    output = setup(args)
    from .datasets.preparation import prepare_dataset
    from .intrinsic_run import run as run_intrinsic
    from .reproduction import run_task
    from .tables import reproduce
    for task in DATASETS:
        prepare_dataset(task, output)
    run_intrinsic(output, SCALES, args.mode)
    for task in DATASETS:
        run_task(task, output, args.mode)
    reproduce(output, output / 'tables')

def tables():
    p = argparse.ArgumentParser(description='Copy reference tables or aggregate newly evaluated outputs')
    p.add_argument('--reference', action='store_true')
    p.add_argument('--input', type=Path, default=REPOSITORY / 'outputs/paper')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from .tables import reproduce
    reproduce(args.input.resolve(), args.output.resolve(), args.reference)
