# Third-party dependencies and data

PMTT first-party code is released under the MIT License. Third-party software,
datasets, and pretrained weights retain their own licenses and access terms.

PMTT does not redistribute datasets or pretrained DINOv3 weights.

## Backbone

[DINOv3](https://github.com/facebookresearch/dinov3) supplies the external
backbone implementation and LVD-1689M pretrained checkpoints. Obtain DINOv3
source and weights from its official distribution. Exact checkpoint identities
are listed in `assets/model_registry.toml`.

## Datasets

The repository stores identifiers, ordering, and PMTT-specific parameters, but
not dataset pixels or annotation payloads. Obtain the following datasets from
their official distributions:

- ImageNet ILSVRC2012;
- SPair-71k;
- DAVIS-2017;
- MegaDepth;
- ScanNet;
- NYUv2;
- ADE20K.

See [DATA.md](DATA.md) for the expected local layouts.

## Libraries

PyTorch, torchvision, NumPy, SciPy, Pillow, OmegaConf, torchmetrics,
matplotlib, packaging, pytest, OpenCV, h5py, and HDF5 remain external
dependencies and retain their own terms. OpenCV supplies essential-matrix and
pose estimation through the PMTT C++ bridge. h5py and HDF5 provide portable
reads of MegaDepth depth arrays.
