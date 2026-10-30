# Installation

PMTT requires Python 3.11 or newer and a CUDA-capable PyTorch environment for
full reproduction.

Install the repository in editable mode:

```sh
python -m pip install -e .
```

Keep the repository checkout available because the reproduction scripts read
the adjacent `assets/` directory. The Python wheel contains the `pmtt` package,
while the complete reproduction state remains in the repository tree.

## Recorded numerical environment

The paper used the following Python packages:

| Package | Version |
|---|---:|
| PyTorch | 2.7.1 |
| torchvision | 0.22.1 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Pillow | 12.3.0 |
| OmegaConf | 2.3.0 |
| torchmetrics | 1.7.4 |
| pytest | 8.4.2 |
| matplotlib | 3.10.3 |
| packaging | 26.3 |

`h5py` provides portable HDF5 reads for MegaDepth depth maps. Its version was
not fixed by the recorded environment.

## DINOv3

Obtain the
[official DINOv3 source](https://github.com/facebookresearch/dinov3) and the
official LVD-1689M S/B/L/H+ checkpoints independently. Configure one source
directory and one checkpoint path per scale in
`configs/paper_reproduction.toml`.

The exact checkpoint filenames, byte sizes, and SHA-256 digests used for the
paper are listed in [`assets/model_registry.toml`](assets/model_registry.toml).
Verify local checkpoints without loading them:

```sh
python scripts/check_weights.py \
    --config configs/paper_reproduction.toml
```

The PMTT wrapper constructs:

- `dinov3_vits16`;
- `dinov3_vitb16`;
- `dinov3_vitl16`;
- `dinov3_vith16plus`.

The state dictionary is loaded strictly. PMTT does not perform network model
lookup.

## Native pose bridge

MegaDepth and ScanNet pose evaluation require CMake, a C++ compiler, and
OpenCV development libraries with the `core` and `calib3d` components.
OpenCV 4.5.4 is the recorded pose environment.

Build the bridge with:

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release
```

Set `paths.native_library` in the configuration or pass `--native-library`.
On Windows, multi-configuration builds commonly place `pmtt_pose.dll` under
`native/build/Release`; the corresponding OpenCV DLLs must be discoverable by
the operating system.

## Numerical conventions

Backbone forwards use BF16 autocast. Exported features and geometry use FP32,
with selected Gram reductions and convex optimization in FP64. TF32 is
disabled. Structured extraction uses batches of 17 views. Intrinsic extraction
uses one canonical forward followed by a 32-view orbital batch.

A GPU is required for the reproduction commands. The CPU tests do not load
DINOv3:

```sh
python -m pytest tests
```
