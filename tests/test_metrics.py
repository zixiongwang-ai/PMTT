import numpy as np
import torch
from pmtt.dense import depth_metrics, confusion, seg_metrics, upsample_argmax
from pmtt.evaluation.video import prepare, jf_cached
from pmtt.statistics import summary_value

def test_depth_identity_and_empty_mask():
    truth = np.array([[1., 2.], [4., 8.]])
    result = depth_metrics(truth, truth, np.ones_like(truth, dtype=bool))
    assert result['AbsRel'] == result['SqRel'] == result['RMSE'] == result['log_RMSE'] == 0
    assert result['delta1'] == result['delta2'] == result['delta3'] == 1
    assert depth_metrics(truth, truth, np.zeros_like(truth, dtype=bool)) == dict(count=0, AbsRel=None)

def test_confusion_ignore_and_additive_estimand():
    label = np.array([[0, 1], [-1, 1]])
    pred = np.array([[0, 0], [149, 1]])
    matrix = confusion(pred, label)
    assert matrix.sum() == 3
    assert matrix[1, 0] == matrix[1, 1] == matrix[0, 0] == 1
    assert seg_metrics(matrix)['miou'] == .5
    values = np.r_[matrix.diagonal(), matrix.sum(1), matrix.sum(0)]
    assert summary_value(np.stack([values, values]), 'iou') == .5

def test_upsampling_ties_choose_lowest_class():
    logits = torch.zeros(196, 150)
    logits[:, 7] = logits[:, 8] = 1
    assert torch.all(upsample_argmax(logits, (29, 37)) == 7)

def test_davis_identical_and_empty_objects():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 10:20] = 1
    context = prepare(mask, [1, 2])
    for obj in (1, 2):
        assert jf_cached(mask, obj, context) == dict(J=1., F=1., JF=1.)
