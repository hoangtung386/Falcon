"""Crop container for managing face and person image crops."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np

__all__ = ["PersonAndFaceCrops"]


class PersonAndFaceCrops:
    """Holds collections of cropped face and person images.

    Manages four categories of crops:
    - crops_persons: face–body pairs where both are available.
    - crops_faces: face crops from paired detections.
    - crops_faces_wo_body: face crops with no associated body.
    - crops_persons_wo_face: body crops with no associated face.
    """

    def __init__(self):
        self.crops_persons: Dict[int, np.ndarray] = {}
        self.crops_faces: Dict[int, np.ndarray] = {}
        self.crops_faces_wo_body: Dict[int, np.ndarray] = {}
        self.crops_persons_wo_face: Dict[int, np.ndarray] = {}

    @staticmethod
    def _add_to_output(crops, out_crops, out_crop_inds):
        inds_to_add = list(crops.keys())
        crops_to_add = list(crops.values())
        out_crops.extend(crops_to_add)
        out_crop_inds.extend(inds_to_add)

    def _get_all_faces(self, use_persons: bool, use_faces: bool):
        def add_none(faces_inds, faces_crops, num):
            faces_inds.extend([None] * num)
            faces_crops.extend([None] * num)

        faces_inds: List[Optional[int]] = []
        faces_crops: List[Optional[np.ndarray]] = []

        if not use_faces:
            add_none(
                faces_inds, faces_crops,
                len(self.crops_persons) + len(self.crops_persons_wo_face),
            )
            return faces_inds, faces_crops

        self._add_to_output(self.crops_faces, faces_crops, faces_inds)
        self._add_to_output(self.crops_faces_wo_body, faces_crops, faces_inds)

        if use_persons:
            add_none(faces_inds, faces_crops, len(self.crops_persons_wo_face))

        return faces_inds, faces_crops

    def _get_all_bodies(self, use_persons: bool, use_faces: bool):
        def add_none(bodies_inds, bodies_crops, num):
            bodies_inds.extend([None] * num)
            bodies_crops.extend([None] * num)

        bodies_inds: List[Optional[int]] = []
        bodies_crops: List[Optional[np.ndarray]] = []

        if not use_persons:
            add_none(
                bodies_inds, bodies_crops,
                len(self.crops_faces) + len(self.crops_faces_wo_body),
            )
            return bodies_inds, bodies_crops

        self._add_to_output(self.crops_persons, bodies_crops, bodies_inds)
        if use_faces:
            add_none(bodies_inds, bodies_crops, len(self.crops_faces_wo_body))
        self._add_to_output(self.crops_persons_wo_face, bodies_crops, bodies_inds)

        return bodies_inds, bodies_crops

    def get_faces_with_bodies(self, use_persons: bool, use_faces: bool):
        """Return aligned lists of body and face crops with their indices.

        Args:
            use_persons: Whether to include body crops.
            use_faces: Whether to include face crops.

        Returns:
            Two tuples ``(body_inds, body_crops)`` and ``(face_inds, face_crops)``.
            Missing entries are filled with ``None``.
        """
        bodies_inds, bodies_crops = self._get_all_bodies(use_persons, use_faces)
        faces_inds, faces_crops = self._get_all_faces(use_persons, use_faces)
        return (bodies_inds, bodies_crops), (faces_inds, faces_crops)

    def save(self, out_dir: str = "output"):
        """Write all stored crops to disk as JPEG images.

        Args:
            out_dir: Output directory path.
        """
        os.makedirs(out_dir, exist_ok=True)
        for idx, crops in enumerate([
            self.crops_persons,
            self.crops_faces,
            self.crops_faces_wo_body,
            self.crops_persons_wo_face,
        ]):
            for crop in crops.values():
                if crop is not None:
                    cv2.imwrite(os.path.join(out_dir, f"{idx}_crop.jpg"), crop)
