from __future__ import annotations

import numpy as np
import torch
from falcon.data.transforms import (
    aggregate_votes_winsorized,
    box_iou,
    class_letterbox,
    cumulative_error,
    cumulative_score,
    iou,
)


class TestTransforms:
    def test_class_letterbox_same_size(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = class_letterbox(img, new_shape=(224, 224))
        assert result.shape == (224, 224, 3)

    def test_class_letterbox_different_size(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = class_letterbox(img, new_shape=(224, 224))
        assert result.shape[:2] == (224, 224)

    def test_box_iou(self):
        box1 = torch.tensor([[0, 0, 10, 10]])
        box2 = torch.tensor([[5, 5, 15, 15]])
        iou_val = box_iou(box1, box2)
        assert 0 < iou_val.item() < 1

    def test_box_iou_no_overlap(self):
        box1 = torch.tensor([[0, 0, 10, 10]])
        box2 = torch.tensor([[20, 20, 30, 30]])
        iou_val = box_iou(box1, box2)
        assert iou_val.item() == 0

    def test_iou(self):
        val = iou([0, 0, 10, 10], [0, 0, 10, 10])
        assert val == 1.0

    def test_iou_no_overlap(self):
        val = iou([0, 0, 5, 5], [10, 10, 15, 15])
        assert val == 0.0

    def test_aggregate_votes_winsorized(self):
        ages = [20, 21, 22, 23, 100]
        result = aggregate_votes_winsorized(ages)
        assert 20 <= result <= 30

    def test_cumulative_score(self):
        pred = torch.tensor([[25.0], [30.0], [35.0]])
        gt = torch.tensor([[25.0], [31.0], [34.0]])
        score = cumulative_score(pred, gt, L=2)
        assert 0 < score.item() <= 1.0

    def test_cumulative_error(self):
        pred = torch.tensor([[25.0], [50.0], [35.0]])
        gt = torch.tensor([[25.0], [30.0], [34.0]])
        err = cumulative_error(pred, gt, L=10)
        assert 0 <= err.item() <= 1.0
