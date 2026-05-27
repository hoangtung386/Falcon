"""Evaluation metrics and result-writing utilities.

Provides:
- ``Metrics``: Tracks age/gender accuracy, cumulative score/error, timing.
- ``time_sync``: CUDA-aware timing utility.
- ``write_results``: Serialises results to CSV or JSON.
"""

from __future__ import annotations

import csv
import json
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

import torch
from falcon.data.transforms import cumulative_error, cumulative_score
from timm.utils import AverageMeter, accuracy

__all__ = ["time_sync", "write_results", "Metrics"]


def time_sync() -> float:
    """Return the current time, synchronising CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def write_results(results_file: str, results, format: str = "csv"):
    """Write evaluation results to a file.

    Args:
        results_file: Output file path.
        results: Single dict or list of dicts.
        format: ``"csv"`` or ``"json"``.
    """
    with open(results_file, mode="w") as f:
        if format == "json":
            json.dump(results, f, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            for r in results:
                w.writerow(r)


class Metrics:
    """Accumulates age and gender evaluation metrics over a validation loop.

    Supports both regression (MAE, CS, CE) and classification (accuracy) modes
    depending on whether *age_classes* are provided.
    """

    def __init__(
        self,
        l_for_cs: int,
        draw_hist: bool,
        age_classes: Optional[List[str]] = None,
    ):
        self.batch_time = AverageMeter()
        self.preproc_batch_time = AverageMeter()
        self.seen = 0
        self.losses = AverageMeter()
        self.top1_m_gender = AverageMeter()
        self.top1_m_age = AverageMeter()

        if age_classes is None:
            self.is_regression = True
            self.av_csl_age = AverageMeter()
            self.max_error = AverageMeter()
            self.per_age_error: Dict[int, List[float]] = defaultdict(list)
            self.l_for_cs = l_for_cs
        else:
            self.is_regression = False

        self.draw_hist = draw_hist

    @staticmethod
    def _resolve_distribution(age_out: torch.Tensor) -> torch.Tensor:
        if age_out.dim() > 1 and age_out.size(1) > 1:
            bins = torch.arange(
                age_out.size(1), device=age_out.device, dtype=torch.float32
            )
            return (age_out.softmax(dim=-1) * bins).sum(dim=-1, keepdim=True)
        return age_out

    def update_regression_age_metrics(
        self, age_out: torch.Tensor, age_target: torch.Tensor
    ):
        """Update regression age metrics (MAE, CS, CE)."""
        age_out = self._resolve_distribution(age_out)
        batch_size = age_out.size(0)
        age_abs_err = torch.abs(age_out - age_target)
        age_acc1 = age_abs_err.sum() / age_out.shape[0]
        age_csl = cumulative_score(age_out, age_target, self.l_for_cs)
        me = cumulative_error(age_out, age_target, 20)

        self.top1_m_age.update(age_acc1.item(), batch_size)
        self.av_csl_age.update(age_csl.item(), batch_size)
        self.max_error.update(me.item(), batch_size)

        if self.draw_hist:
            for i in range(age_out.shape[0]):
                self.per_age_error[int(age_target[i].item())].append(
                    age_abs_err[i].item()
                )

    def update_age_accuracy(self, age_out: torch.Tensor, age_target: torch.Tensor):
        """Update classification age accuracy."""
        batch_size = age_out.size(0)
        if batch_size == 0:
            return
        correct = torch.sum(age_out == age_target)
        age_acc1 = correct * 100.0 / batch_size
        self.top1_m_age.update(age_acc1.item(), batch_size)

    def update_gender_accuracy(self, gender_out, gender_target):
        """Update gender classification accuracy."""
        if gender_out is None or gender_out.size(0) == 0:
            return
        acc = accuracy(gender_out, gender_target, topk=(1,))[0]
        if acc is not None:
            self.top1_m_gender.update(acc.item(), gender_out.size(0))

    def update_loss(self, loss, batch_size: int):
        """Update the running loss meter."""
        self.losses.update(loss.item(), batch_size)

    def update_time(
        self, process_time: float, preprocess_time: float, batch_size: int
    ):
        """Update timing meters."""
        self.seen += batch_size
        self.batch_time.update(process_time)
        self.preproc_batch_time.update(preprocess_time)

    def get_info_str(self, batch_size: int) -> str:
        """Return a formatted string of current metrics for logging."""
        avg_time = (
            self.preproc_batch_time.sum + self.batch_time.sum
        ) / self.batch_time.count
        cur_time = self.batch_time.val + self.preproc_batch_time.val
        middle = (
            f"Time: {cur_time:.3f}s ({avg_time:.3f}s, {batch_size / avg_time:>7.2f}/s)  "
            f"Loss: {self.losses.val:>7.4f} ({self.losses.avg:>6.4f})  "
            f"Gender Acc: {self.top1_m_gender.val:>7.2f} ({self.top1_m_gender.avg:>7.2f}) "
        )
        if self.is_regression:
            age_info = (
                f"Age CS@{self.l_for_cs}: {self.av_csl_age.val:>7.4f} ({self.av_csl_age.avg:>7.4f})  "
                f"Age CE@20: {self.max_error.val:>7.4f} ({self.max_error.avg:>7.4f})  "
                f"Age ME: {self.top1_m_age.val:>7.2f} ({self.top1_m_age.avg:>7.2f})"
            )
        else:
            age_info = (
                f"Age Acc: {self.top1_m_age.val:>7.2f} ({self.top1_m_age.avg:>7.2f})"
            )
        return middle + age_info

    def get_result(self) -> OrderedDict:
        """Return an ordered dictionary of final metric values."""
        results = OrderedDict(
            mean_inference_time=self.batch_time.sum / self.seen * 1e3,
            mean_preprocessing_time=self.preproc_batch_time.sum / self.seen * 1e3,
            agetop1=round(self.top1_m_age.avg, 4),
            agetop1_err=round(100 - self.top1_m_age.avg, 4),
        )
        if self.is_regression:
            results.update(
                dict(
                    max_error=self.max_error.avg,
                    csl=self.av_csl_age.avg,
                    per_age_error=self.per_age_error,
                )
            )
        if self.top1_m_gender.count > 0:
            results.update(
                dict(
                    gendertop1=round(self.top1_m_gender.avg, 4),
                    gendertop1_err=round(100 - self.top1_m_gender.avg, 4),
                )
            )
        return results
