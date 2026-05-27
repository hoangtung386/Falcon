"""High-level inference predictor combining detection and age/gender estimation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from falcon.model.inference import Falcon
from falcon.model.yolo_detector import Detector
from falcon.structures import AGE_GENDER_TYPE, PersonAndFaceResult

__all__ = ["Predictor"]


class Predictor:
    """End-to-end predictor: runs YOLO detection + Falcon age/gender inference.

    Args:
        config: An object with attributes ``detector_weights``, ``device``,
            ``checkpoint``, ``with_persons``, ``disable_faces``, and ``draw``.
        verbose: Whether to enable verbose logging.
    """

    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(
            config.detector_weights, config.device, verbose=verbose
        )
        self.age_gender_model = Falcon(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

    def recognize(
        self, image: np.ndarray
    ) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        """Run full detection + age/gender pipeline on a single image.

        Args:
            image: BGR image array.

        Returns:
            Tuple of ``(detected_objects, annotated_image)``.
        """
        detected_objects = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)
        out_im = detected_objects.plot() if self.draw else None
        return detected_objects, out_im

    def recognize_video(self, source: str) -> Generator:
        """Run detection + age/gender on each frame of a video.

        Yields:
            ``(history, frame)`` where *history* is the tracking-based
            age/gender history and *frame* is the (optionally annotated)
            video frame.
        """
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm.tqdm(range(total_frames)):
            ret, frame = video_capture.read()
            if not ret:
                break

            detected_objects = self.detector.track(frame)
            self.age_gender_model.predict(frame, detected_objects)

            cur_persons, cur_faces = detected_objects.get_results_for_tracking()
            for guid, data in cur_persons.items():
                if None not in data:
                    history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    history[guid].append(data)

            detected_objects.set_tracked_age_gender(history)
            frame = detected_objects.plot() if self.draw else frame
            yield history, frame
