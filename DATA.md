# Data

Obtain every dataset independently under its own terms. PMTT includes identifiers and selection order, not images, masks, depths, keypoint annotations or camera payloads. Preparation reads your files and writes local JSON indexes under the selected output directory. It does not modify or duplicate the raw dataset.

All commands below are run from the repository root. Replace the example relative data directories with your actual paths. Set the same roots in `configs/paper_reproduction.toml` before evaluation; CLI root flags override configuration. Configuration paths resolve relative to the TOML file, while CLI paths resolve relative to the invoking directory. The short config names `nyu` and `ade` refer to NYUv2 and ADE20K.

| Dataset | Root supplied by the user | Required files under that root | Paper evaluation |
|---|---|---|---|
| ImageNet ILSVRC2012 training images | extracted training image tree | `n########/n########_*.JPEG`, matching cohort CSV identifiers | Two separate cohorts of 4,096 anchors |
| SPair-71k, large split | `SPair-71k` | `JPEGImages/<category>/*.jpg`, `PairAnnotation/{trn,val,test}/*.json`, `Layout/large/{trn,val,test}.txt` | 12,234 test pairs |
| DAVIS-2017, 480p | `DAVIS` | `JPEGImages/480p`, `Annotations/480p`, `ImageSets/2017/{train,val}.txt` | 30 sequences, 1,969 target frames, 61 object tracks |
| MegaDepth-1500 calibrated benchmark | `megadepth1500` | `pairs_calibrated.txt`, `views.txt`, `images/<scene>/*.jpg`, `depths/<scene>/*.h5` | All 1,500 pairs |
| ScanNet-1500 calibrated benchmark | `scannet1500` | `pairs_calibrated.txt`, `<scene>/{color,depth,pose,intrinsic}` | All 1,500 pairs |
| NYUv2 aligned RGB/depth frame distribution | directory containing `nyu/` | exact `nyu/<scene>/rgb_*.jpg` and `sync_depth_*.png` identities listed in the manifest; depth PNG values in millimeters | 654 test images |
| ADE20K ADEChallengeData2016 | `ADEChallengeData2016` | `images/{training,validation}` and `annotations/{training,validation}` | 2,000 validation images, 150 classes |

```sh
python scripts/prepare_data/prepare_spair.py --data-root datasets/SPair-71k --output outputs/paper
python scripts/prepare_data/prepare_davis.py --data-root datasets/DAVIS --output outputs/paper
python scripts/prepare_data/prepare_megadepth.py --data-root datasets/megadepth1500 --output outputs/paper
python scripts/prepare_data/prepare_scannet.py --data-root datasets/scannet1500 --output outputs/paper
python scripts/prepare_data/prepare_nyuv2.py --data-root datasets/NYUv2 --output outputs/paper
python scripts/prepare_data/prepare_ade20k.py --data-root datasets/ADEChallengeData2016 --output outputs/paper
```

The prepared indexes are `outputs/paper/manifests/{spair,davis,megadepth,scannet,nyu,ade}.json`. SPair annotations are read from the official files at preparation time. DAVIS source memberships are reconstructed from the official first mask, including area fractions, padding geometry and ignored pixels. Geometry indexes read official calibrated camera rows, check camera conventions, and compute the GT visibility required by the evaluator. These local indexes may contain dataset annotations and are not repository assets.

For MegaDepth, each calibrated pair row contains two image identifiers followed by 30 numbers: two row-major 3×3 intrinsic matrices, a 3×3 relative rotation and a translation. ScanNet rows contain 34 numbers: the two intrinsic matrices and a 4×4 relative transform. Pair order must exactly match the supplied slim manifests. ScanNet requires both depth/color intrinsics and extrinsics; the current protocol requires identity depth/color extrinsics. MegaDepth depth-resolution scaling is handled by the adapter. The calibrated benchmark distributions, rather than arbitrary new pairs from the larger datasets, define the paper's 1,500-pair identity.

NYUv2 needs the aligned frame distribution with the exact identifiers in `assets/manifests/nyuv2.csv`, including 24,231 training frames. The preparer validates that distribution; it does not infer a new train split or convert arbitrary raw video streams into it. Provide aligned millimeter depth PNGs, not differently encoded depth maps. ADE20K expects all 20,210 training identities in the supplied manifest. Fitting lists preserve their exact order and calibration membership.

ImageNet needs no feature or annotation preparation. Set `paths.imagenet` to the extracted training tree. Cohort rows identify each image, canonical patch, crop coordinates, direction labels, Visible/Hidden membership and numeric observation seed. The shared photometric transform is regenerated from that seed before cropping; recropping the untransformed image is not equivalent.

SPair pair identifiers can contain colons inside CSV fields. Output filenames use portable identity keys. If your extracted annotation filenames were normalized on Windows, the preparer resolves a unique filename with the same pair prefix and checks its annotation identity.
