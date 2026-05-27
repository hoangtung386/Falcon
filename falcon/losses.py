"""Loss functions for age estimation and gender classification.

Provides three loss variants:
- AgeGenderLoss: KL-divergence-based distribution loss with optional gender CE.
- OrdinalAgeLoss: Binary cross-entropy formulation for ordinal age regression.
- WeightedMSE: Mean-squared error with adjusted weight for outliers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AgeGenderLoss", "OrdinalAgeLoss", "WeightedMSE"]


class AgeGenderLoss(nn.Module):
    """KL-divergence loss for age distribution with optional gender cross-entropy.

    Converts the age target into a Gaussian-smoothed label distribution
    (fixed sigma) and minimises KL divergence against predicted logits.
    """

    def __init__(
        self,
        num_age_bins: int = 101,
        age_weight: float = 1.0,
        gender_weight: float = 1.0,
        only_age: bool = False,
        sigma: float = 2.0,
    ):
        super().__init__()
        self.num_age_bins = num_age_bins
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        self.only_age = only_age
        self.sigma = sigma

    def _age_to_distribution(self, age: torch.Tensor) -> torch.Tensor:
        """Build a Gaussian-smoothed label distribution for a given age.

        Args:
            age: Normalised age tensor shaped (B, 1) in [0, 1].

        Returns:
            Soft target distribution shaped (B, num_age_bins).
        """
        bins = torch.arange(self.num_age_bins, device=age.device, dtype=torch.float32)
        age_val = age * self.num_age_bins
        diff = bins - age_val
        target_dist = torch.exp(-(diff**2) / (2 * self.sigma**2))
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)
        return target_dist

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined age + gender loss.

        Args:
            predictions: Raw model output. Shape (B, C) where C = 2 + num_age_bins
                         for age+gender mode, or (B, num_age_bins) for age-only.
            targets: Ground truth tensor with columns [age, gender].

        Returns:
            Scalar loss value.
        """
        age_target = targets[:, 0:1]
        age_pred = predictions[:, 2:] if not self.only_age else predictions

        age_target_dist = self._age_to_distribution(age_target)
        age_loss = F.kl_div(
            age_pred.log_softmax(dim=-1),
            age_target_dist,
            reduction="batchmean",
        )

        if self.only_age:
            return age_loss * self.age_weight

        gender_target = targets[:, 1].long()
        gender_pred = predictions[:, :2]
        gender_loss = F.cross_entropy(gender_pred, gender_target)
        return age_loss * self.age_weight + gender_loss * self.gender_weight


class OrdinalAgeLoss(nn.Module):
    """Ordinal regression loss for age estimation.

    Transforms age regression into a series of binary classification tasks
    and uses binary cross-entropy as the surrogate loss.
    """

    def __init__(self, num_classes: int = 101):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinal loss.

        Args:
            logits: Predicted logits of shape (B, num_classes).
            targets: Ground-truth tensor ``[age_normalized, gender]`` (B, 2)
                     or just age (B, 1) or (B,).

        Returns:
            Scalar ordinal loss.
        """
        age_target = targets[:, 0] if targets.dim() > 1 and targets.size(-1) > 1 else targets
        age_target = age_target.float().squeeze(-1)
        age_target = (age_target * (self.num_classes - 1)).long()
        labels = torch.arange(self.num_classes, device=age_target.device)
        labels = labels.unsqueeze(0).expand(age_target.size(0), -1)
        labels = (labels < age_target.unsqueeze(1)).float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss


class WeightedMSE(nn.Module):
    """Mean-squared error loss with higher weight for large residuals.

    Samples whose absolute error exceeds ``regression_bound`` receive
    a 1.5× multiplier to emphasise hard samples during training.
    """

    def __init__(self, regression_bound: float = 0.3):
        super().__init__()
        self.regression_bound = regression_bound

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE.

        Args:
            pred: Predicted values.
            target: Ground-truth values.

        Returns:
            Scalar weighted MSE.
        """
        diff = torch.abs(pred - target)
        weight = torch.where(diff < self.regression_bound, 1.0, 1.5)
        return (weight * (pred - target) ** 2).mean()
