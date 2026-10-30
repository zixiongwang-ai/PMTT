# Reference results

`assets/reference_results/` contains the aggregate reference results reported
in the paper.

| File | Contents |
|---|---|
| `intrinsic_summary.csv` | Two cohorts × four scales: coverage, effective rank, energy, spectra, overlap, and controls |
| `primary_72_cells.csv` | Six tasks × four scales × Center/Observed/Manifold |
| `paired_effects.csv` | Paired deltas, 95% intervals, units, and scope |
| `control_summary.csv` | DonorObserved and DestroyedManifold aggregate scores |
| `supplementary_pose.csv` | B/H+ 64-anchor pose-only aggregates |

Manifold coverage exceeds Observed coverage in both intrinsic cohorts at every
scale. The normalized spectrum is ordered from largest to smallest; its first
entry is the top-1 energy fraction, and the sum of its first four entries is
the top-4 energy fraction. Full-32 uses all orbital states and is a diagnostic
ceiling. Pooled coverage is a ratio of error sums and is distinct from the
mean of per-anchor coverage ratios.

The three primary downstream comparisons are:

- **Observed − Center**: Information Acquisition;
- **Manifold − Observed**: Representation Organization;
- **Manifold − Center**: total representation extension.

| Task | Native metric and aggregation |
|---|---|
| SPair-71k | PCK@0.1 of target bounding-box extent; micro over keypoints, paired bootstrap over pairs; category macro reported separately |
| DAVIS-2017 | J&F; frames aggregate within object tracks, and sequence clusters use object-count weighting |
| MegaDepth / ScanNet | Mean pair P@10 in resized target pixels; failed poses retain 180° error |
| NYUv2 | Mean image AbsRel, lower is better; the signed table score is −AbsRel; SqRel, RMSE, log RMSE, and delta1/2/3 are secondary |
| ADE20K | mIoU from the sum of image confusion matrices; image bootstrap preserves the additive-confusion estimand |

Primary geometric correspondence uses 32 deterministic anchors. The 64-anchor
results are supplementary pose analyses and do not enter the 72 primary cells.
Calibration-overlap pairs remain in descriptive all-pair results; the saved
confirmatory intervals exclude them.

See [REPRODUCE.md](REPRODUCE.md) for commands and numerical comparison
conventions.
