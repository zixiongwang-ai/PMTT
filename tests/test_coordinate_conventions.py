import numpy as np
import torch
from pmtt.sampling import correspondence, sample
from pmtt.evaluation.correspondence import pck
from pmtt.datasets.geometry import resize_spec

def test_canonical_patch_centers_sample_without_half_pixel_shift():
    crop = torch.tensor([[0., 0., 224., 224.]])
    coordinates, valid = correspondence(crop, crop)
    values = torch.arange(196, dtype=torch.float32).reshape(1, 196, 1)
    assert valid.all()
    torch.testing.assert_close(sample(values, coordinates, (14, 14)), values, atol=2e-5, rtol=0)

def test_spair_annotation_offset_and_native_grid_center():
    scores = torch.zeros(1, 4096)
    scores[0, 0] = 1
    annotation = torch.tensor([[0., 0.]])
    result = pck(scores, annotation + .5, (64, 64), [0, 0, 64, 64])
    assert result['ok'].all()
    assert result['rank'].item() == 1

def test_geometry_resize_uses_rounded_pixel_axes():
    spec = resize_spec((1000, 701))
    assert spec['resized'] == [640, 448]
    np.testing.assert_allclose(spec['scale'], [640 / 1000, 448 / 701])
