# DINOv3 integration

PMTT uses the official
[DINOv3 repository](https://github.com/facebookresearch/dinov3) and external
LVD-1689M checkpoints.

Configure the DINOv3 source directory and the S/B/L/H+ checkpoint paths in
`configs/paper_reproduction.toml`. Exact constructors, filenames, byte sizes,
and SHA-256 digests are recorded in
[`assets/model_registry.toml`](../../assets/model_registry.toml).

DINOv3 source and pretrained weights are not vendored or redistributed by
PMTT. They remain subject to their own terms. See
[INSTALL.md](../../INSTALL.md).
