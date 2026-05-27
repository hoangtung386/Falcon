"""YOLO-based face and person detector."""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import PIL
import torch
from falcon.structures import PersonAndFaceResult
from ultralytics import YOLO
from ultralytics.engine.results import Results

__all__ = ["Detector"]


class Detector:
    """Wrapper around a YOLO model for simultaneous face + person detection.

    Args:
        weights: Path to the YOLO weights file.
        device: Torch device string.
        half: Whether to run inference in FP16.
        verbose: Whether to enable YOLO verbose output.
        conf_thresh: Detection confidence threshold.
        iou_thresh: NMS IoU threshold.
    """

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.7,
    ):
        self.yolo = YOLO(weights)
        self.yolo.fuse()
        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        if self.half:
            self.yolo.model = self.yolo.model.half()

        self.detector_names: Dict[int, str] = self.yolo.model.names
        self.detector_kwargs = {
            "conf": conf_thresh,
            "iou": iou_thresh,
            "half": self.half,
            "verbose": verbose,
        }

    def predict(
        self, image: Union[np.ndarray, str, "PIL.Image"]
    ) -> PersonAndFaceResult:
        """Run detection on a single image.

        Args:
            image: BGR array, file path, or PIL Image.

        Returns:
            A ``PersonAndFaceResult`` wrapping the detections.
        """
        results: Results = self.yolo.predict(image, **self.detector_kwargs)[0]
        return PersonAndFaceResult(results)

    def track(
        self, image: Union[np.ndarray, str, "PIL.Image"]
    ) -> PersonAndFaceResult:
        """Run detection with tracking.

        Args:
            image: BGR array, file path, or PIL Image.

        Returns:
            A ``PersonAndFaceResult`` with tracking IDs.
        """
        results: Results = self.yolo.track(
            image, persist=True, **self.detector_kwargs
        )[0]
        return PersonAndFaceResult(results)
