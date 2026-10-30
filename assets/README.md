# Scientific assets

The `assets/` directory contains the minimum scientific state required to
recover the exact paper protocol without redistributing datasets or pretrained
backbone weights.

## Cohorts

`cohorts/` stores the ordered Original Cohort and Independent Holdout Cohort.
Each cohort contains 4,096 anchors.

The CSV files provide public image identities, source rows, canonical patch
positions, and observation seeds. The NPZ files provide:

- `crops`: `[row, 33, 4]` original-image `(x, y, width, height)` boxes;
- `positions`: canonical row-major positions in the 14×14 patch grid;
- `visible` and `hidden`: disjoint observation indices;
- `metadata`: relative location, canonical displacement, log area ratio, and
  log aspect ratio;
- `sectors`: direction labels, with `-1` for the Canonical State.

Exception metadata describes geometric feasibility and is independent of
features.

## Manifests

`manifests/` fixes evaluation identities, ordering, splits, and calibration
membership. The `fitting/` directory contains:

- DAVIS train-side selection;
- MegaDepth and ScanNet fitting/validation pair identities;
- NYUv2 and ADE20K axis-fitting and scalar-calibration membership;
- dense donor identities.

NYUv2 IDs use dataset-relative paths under the configured dataset root.
Dataset annotations, pixels, depths, masks, and camera payloads are not
included.

## Readers and task axes

`readers/` contains the fitted SPair, DAVIS, MegaDepth, and ScanNet readers,
including standardization, knots, coefficients, and selected L2 values.
Reader capacities are 24/8/9 for Center/Observed/Manifold.

`task_axes/` contains the per-scale Center task axes:

- NYUv2: `weight[D, 1]`, `bias[1]`;
- ADE20K: `weight[D, 150]`, `bias[150]`, and training class-frequency counts.

Observed and Manifold JSON files contain their selected scalar mixtures,
standardization, intercepts, and L2 values. No optimizer states or feature
banks are included.

## Calibration and deterministic seeds

`calibration/` contains intrinsic caps, pose thresholds, and the deterministic
observation/control seed tables. Seed-table keys are the first 16 bytes of the
SHA-256 digest of compact sorted-key JSON identities. Grouped tables use
`offsets` to locate the consecutive anchor seeds.

The seed tables contain no images, features, predictions, or model outputs.

## Model registry

`model_registry.toml` records the exact external DINOv3 checkpoint filenames,
byte sizes, constructors, and SHA-256 digests used for the paper. The weights
remain external.

## Reference results

`reference_results/` contains compact paper aggregates. The normalized
intrinsic spectrum stores its components in descending order; top-4 energy is
the sum of the first four components.

All public NPZ files open with `allow_pickle=False`.
