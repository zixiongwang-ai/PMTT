# Reproduction

Install PMTT as described in [INSTALL.md](INSTALL.md), prepare the datasets in
[DATA.md](DATA.md), and configure paths in
`configs/paper_reproduction.toml`.

The repository supports two modes:

- **paper** uses the supplied exact cohorts, manifests, fitted readers, task
  axes, scalar mixtures, intrinsic caps, calibration statistics, and
  thresholds;
- **refit** repeats the train-side fitting procedures from the fixed fitting
  identities before evaluation.

## Check external checkpoints

```sh
python scripts/check_weights.py \
    --config configs/paper_reproduction.toml
```

Checkpoint identities are defined in `assets/model_registry.toml`.

## Exact paper mode

Paper mode regenerates PMTT representations from official images and the
specified DINOv3 checkpoints. Donor controls regenerate source observations
from the recorded donor identities. No reader or task-axis fitting occurs.

Run individual components:

```sh
python scripts/run_intrinsic.py --mode paper --output outputs/paper
python scripts/run_spair.py --mode paper --output outputs/paper
python scripts/run_davis.py --mode paper --output outputs/paper
python scripts/run_megadepth.py --mode paper --output outputs/paper
python scripts/run_scannet.py --mode paper --output outputs/paper
python scripts/run_nyuv2.py --mode paper --output outputs/paper
python scripts/run_ade20k.py --mode paper --output outputs/paper
```

Run the complete paper protocol:

```sh
python scripts/reproduce_all.py \
    --config configs/paper_reproduction.toml \
    --mode paper \
    --output outputs/paper
```

`--scale B` selects one scale for an individual task command. Repeat
`--scale` to request several scales. All four scales are required to reproduce
the complete 72-cell matrix and adjacent-scale analyses.

CLI options override paths in the configuration:

```text
--dinov3-source
--weights-s
--weights-b
--weights-l
--weights-hplus
--native-library
--imagenet-root
--spair-root
--davis-root
--megadepth-root
--scannet-root
--nyu-root
--ade-root
```

SPair and DAVIS paper evaluation require the training sources used by donor
controls. NYUv2 and ADE20K require the recorded training donor images and
labels. Geometry controls use donor source images from the official pair
distribution.

## Full refit mode

Use a separate, initially empty output directory:

```sh
python scripts/reproduce_all.py \
    --config configs/paper_reproduction.toml \
    --mode refit \
    --output outputs/refit
```

Full refit uses all four backbone scales and the fixed fitting identities.

- SPair selects the reader on its official validation split.
- DAVIS selects on the fixed train-side sequence split and then fits on all
  official training sequences.
- MegaDepth and ScanNet use 96 fitting pairs and 32 validation pairs, then
  refit on all 128 calibration identities. S/H+ standardization is estimated
  on the same identities.
- NYUv2 and ADE20K first fit the Center task axis, then fit scale-shared scalar
  mixtures on the fixed calibration split.

Use a separate output root for each mode, checkpoint set, dataset revision, or
numerical environment. Feature caches and fitting intermediates are local
outputs and are not required as repository assets.

## Outputs

Each task writes to its own output directory. Geometry stores 32-anchor primary
results under `records/` and B/H+ 64-anchor supplementary pose under
`pose64/`. Intrinsic output keeps the Original Cohort and Independent Holdout
Cohort separate.

Generate tables from newly evaluated outputs:

```sh
python scripts/reproduce_tables.py \
    --input outputs/paper \
    --output outputs/paper/tables
```

Copy the aggregate paper reference tables without inference or fitting:

```sh
python scripts/reproduce_tables.py \
    --reference \
    --output outputs/reference_tables
```

Reference aggregates are stored in `assets/reference_results/`.

## Numerical comparison

NYUv2's signed table score is `-AbsRel`; lower raw AbsRel is better. Raw metric
units must not be averaged across tasks. MegaDepth and ScanNet descriptive
scores use all 1,500 pairs, while their paired confirmatory effects use the
1,372 untouched pairs. Bootstrap confidence intervals use 2,000 draws and the
natural evaluation unit for each task.

Small floating-point differences can arise across devices and numerical
libraries. The reference comparison settings are:

- relative tolerance: `1e-7`;
- ordinary absolute tolerance: `1e-8`;
- intrinsic feature tolerance: `1e-6`;
- intrinsic spectrum/error tolerance: `2e-5`.

OpenCV and related numerical-library differences can affect pose quantities.
SVD factor signs are not identifiable; compare factors through spectra or
basis-invariant geometry rather than raw signs.
