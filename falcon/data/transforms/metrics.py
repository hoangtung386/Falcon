"""Age-estimation metrics: winsorised voting, cumulative score/error."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

__all__ = ["aggregate_votes_winsorized", "cumulative_score", "cumulative_error"]


def aggregate_votes_winsorized(ages: List[float], max_age_dist: float = 6) -> float:
    """Winsorised mean: clip extreme values before averaging.

    Values outside ``[median - max_age_dist, median + max_age_dist]`` are
    clamped to the interval boundaries.

    Args:
        ages: List of age predictions.
        max_age_dist: Maximum allowed deviation from the median.

    Returns:
        Winsorised mean age.
    """
    median = np.median(ages)
    ages = np.clip(ages, median - max_age_dist, median + max_age_dist)
    return float(np.mean(ages))


def cumulative_score(
    pred_ages: torch.Tensor, gt_ages: torch.Tensor, L: float, tol: float = 1e-6
) -> torch.Tensor:
    """Fraction of predictions within absolute error ≤ *L*.

    Args:
        pred_ages: Predicted ages.
        gt_ages: Ground-truth ages.
        L: Error threshold.
        tol: Numerical tolerance.

    Returns:
        Scalar tensor in [0, 1].
    """
    correct = torch.sum(torch.abs(pred_ages - gt_ages) <= L + tol)
    return correct / pred_ages.shape[0]


def cumulative_error(
    pred_ages: torch.Tensor, gt_ages: torch.Tensor, L: float, tol: float = 1e-6
) -> torch.Tensor:
    """Fraction of predictions with absolute error ≥ *L*.

    Args:
        pred_ages: Predicted ages.
        gt_ages: Ground-truth ages.
        L: Error threshold.
        tol: Numerical tolerance.

    Returns:
        Scalar tensor in [0, 1].
    """
    correct = torch.sum(torch.abs(pred_ages - gt_ages) >= L + tol)
    return correct / pred_ages.shape[0]
