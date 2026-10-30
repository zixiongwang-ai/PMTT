# A Patch is More Than a Token (PMTT)

PMTT extends a patch representation from one Canonical State to a
canonical-centered Representation State Family. Dense Orbital Views expose
identity-preserving states of the same image patch. **Observed** retains the
Canonical State and sixteen actual Visible States. **Manifold** organizes the
same Visible information as a bounded, rank-at-most-eight local geometry
within Patch-Level Local Feature Space.

This repository provides:

- the PMTT representation implementation;
- Visible-to-Hidden intrinsic evaluation;
- SPair-71k, DAVIS-2017, MegaDepth-1500, ScanNet-1500, NYUv2, and ADE20K
  evaluation;
- fitting code for the low-capacity readers and dense task axes;
- exact cohort identities, manifests, fitted parameters, calibration
  identities, and aggregate reference results used in the paper.

DINOv3 supplies the frozen S, B, L, and H+ backbones. Datasets and pretrained
weights are external and are not redistributed.

## Getting started

1. Follow [INSTALL.md](INSTALL.md).
2. Obtain and prepare the datasets described in [DATA.md](DATA.md).
3. Configure local paths in
   [`configs/paper_reproduction.toml`](configs/paper_reproduction.toml).
4. Verify the external DINOv3 checkpoints:
   ```sh
   python scripts/check_weights.py \
       --config configs/paper_reproduction.toml
   ```
5. Run the desired evaluation or follow [REPRODUCE.md](REPRODUCE.md).

A complete paper-mode run is launched with:

```sh
python scripts/reproduce_all.py \
    --config configs/paper_reproduction.toml \
    --mode paper \
    --output outputs/paper
```

The repository also supports full refitting of the readers, geometry
calibration, and dense task axes from the fixed fitting identities.

## Scientific comparisons

PMTT separates information acquisition from representation organization:

- **Observed − Center** measures the value of additional same-patch
  observations.
- **Manifold − Observed** measures the value of organizing the same acquired
  information.
- **Manifold − Center** measures the total representation-extension effect.

Full-32 uses all orbital states and serves only as an intrinsic diagnostic
ceiling. It is never used as the inference representation.

## Repository layout

```text
pmtt/        Scientific implementation
scripts/     Data preparation, evaluation, fitting, and table reproduction
configs/     Paper reproduction configuration
assets/      Cohorts, manifests, fitted parameters, and reference results
native/      OpenCV pose bridge
tests/       CPU tests for coordinates, observations, geometry, and metrics
```

The compact reference results are documented in [RESULTS.md](RESULTS.md).
The scientific assets are described in
[`assets/README.md`](assets/README.md).

## Paper

**A Patch is More Than a Token**  
Zixiong Wang and Haiping Ma, 2026.

For inquiries: Zixiong Wang, `zixiongwang.ai@outlook.com`.

## License

PMTT first-party code is released under the [MIT License](LICENSE).
Datasets, DINOv3, and other third-party dependencies remain subject to their
own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
