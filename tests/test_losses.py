from __future__ import annotations

import torch
from falcon.losses import AgeGenderLoss, OrdinalAgeLoss, WeightedMSE


class TestAgeGenderLoss:
    def test_distribution_loss(self):
        loss_fn = AgeGenderLoss(num_age_bins=101, only_age=True)
        preds = torch.randn(4, 101)
        # age in [0, 1] after normalisation fix
        targets = torch.tensor([[0.2], [0.5], [0.0], [0.8]])
        loss = loss_fn(preds, targets)
        assert loss.item() > 0

    def test_distribution_loss_with_gender(self):
        loss_fn = AgeGenderLoss(num_age_bins=101, only_age=False)
        preds = torch.randn(4, 103)
        # age in [0, 1], gender 0/1
        targets = torch.tensor([[0.2, 0.0], [0.5, 1.0], [0.0, 0.0], [0.8, 1.0]])
        loss = loss_fn(preds, targets)
        assert loss.item() > 0


class TestOrdinalAgeLoss:
    def test_ordinal_loss(self):
        loss_fn = OrdinalAgeLoss(num_classes=101)
        logits = torch.randn(2, 101)
        # age in [0, 1] (50/100 = 0.5, 25/100 = 0.25)
        age_target = torch.tensor([[0.5], [0.25]])
        loss = loss_fn(logits, age_target)
        assert loss.item() > 0


class TestWeightedMSE:
    def test_weighted_mse(self):
        loss_fn = WeightedMSE(regression_bound=0.3)
        pred = torch.tensor([[0.5], [0.3], [0.8]])
        target = torch.tensor([[0.5], [0.3], [0.8]])
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0

    def test_weighted_mse_penalizes_outliers(self):
        loss_fn = WeightedMSE(regression_bound=0.3)
        pred = torch.tensor([[0.5], [0.8]])
        target = torch.tensor([[0.5], [0.3]])
        loss = loss_fn(pred, target)
        assert loss.item() > 0
