import numpy as np
from pmtt.datasets.geometry import anchor_grid, resize_spec, grid_xy
from pmtt.evaluation.geometry import assignment, pose_errors

def test_anchor_orientation_and_margin():
    for size, counts in [((800, 600), (8, 4)), ((600, 800), (4, 8))]:
        points = anchor_grid(size)
        assert points.shape == (32, 2)
        assert len(np.unique(points[:, 0])) == counts[0]
        assert len(np.unique(points[:, 1])) == counts[1]
        np.testing.assert_allclose(points.min(0), np.asarray(size) * .08)
        np.testing.assert_allclose(points.max(0), np.asarray(size) * .92)

def test_assignment_is_one_to_one_and_refines_in_resized_pixels():
    spec = dict(grid=[2, 2], scale=[1., 1.])
    scores = np.array([[20., 0., 0., 0.], [19., 18., 0., 0.]])
    match = assignment(scores, spec)
    assert len(set(match['target'])) == 2
    assert np.all(match['refined'] >= 8)
    assert np.all(match['refined'] <= 24)
    assert np.array_equal(grid_xy(spec), [[8, 8], [24, 8], [8, 24], [24, 24]])

def test_pose_failures_are_not_dropped():
    result = pose_errors({'status': 'failure'}, {})
    assert result == dict(rotation_error=180., translation_error=180., pose_error=180.)
