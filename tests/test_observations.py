import numpy as np
import torch
from PIL import Image
from pmtt.assets import load_cohort_manifest
from pmtt.observations import supported_positions, group

def test_saved_visible_hidden_identity():
    for cohort in ('original', 'independent_holdout'):
        rows, arrays = load_cohort_manifest(cohort)
        assert len(rows) == 4096
        assert arrays['crops'].shape == (4096, 33, 4)
        assert set(arrays['visible']).isdisjoint(arrays['hidden'])
        assert set(arrays['visible']) | set(arrays['hidden']) == set(range(1, 33))
        for selection in (arrays['visible'], arrays['hidden']):
            assert np.array_equal(np.bincount(arrays['sectors'][0, selection], minlength=8), np.full(8, 2))
        for index in (0, 1024, 2048, 4095):
            support = supported_positions(torch.from_numpy(arrays['crops'][index]), arrays['sectors'][index])
            assert support[int(rows[index]['patch_index'])]

def test_constructive_observations_are_deterministic():
    image = Image.new('RGB', (640, 64), (80, 120, 160))
    first = group(image, 'test_image', visible_count=16, hidden_count=16, packed=True, observation_seed=27)
    second = group(image, 'test_image', visible_count=16, hidden_count=16, packed=True, observation_seed=27)
    assert torch.equal(first['crops'], second['crops'])
    assert torch.equal(first['images'], second['images'])
    assert len(torch.unique(first['crops'], dim=0)) == 33
