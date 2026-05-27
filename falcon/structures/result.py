"""Detection result wrapper that manages face–person association, age/gender
assignment, tracking, and crop extraction."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from falcon.data.transforms import aggregate_votes_winsorized, assign_faces, box_iou
from falcon.structures.crops import PersonAndFaceCrops
from falcon.structures.types import AGE_GENDER_TYPE
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator, colors

__all__ = ["PersonAndFaceResult"]

_IOU_THRESH = 0.000001
_MIN_PERSON_SIZE = 50
_CROP_ROUND_RATE = 0.3
_MIN_PERSON_AFTERCUT_RATIO = 0.4


class PersonAndFaceResult:
    """Wraps a YOLO detection result and attaches age/gender predictions.

    Maintains a face-to-person association map so that age/gender predictions
    from a face are also propagated to the corresponding body bounding box.
    Supports tracking-aware temporal smoothing via ``set_tracked_age_gender``.
    """

    def __init__(self, results: Results):
        self.yolo_results = results
        names = set(results.names.values())
        assert "person" in names and "face" in names

        self.face_to_person_map: Dict[int, Optional[int]] = {
            ind: None for ind in self.get_bboxes_inds("face")
        }
        self.unassigned_persons_inds: List[int] = self.get_bboxes_inds("person")
        n = len(self.yolo_results.boxes)
        self.ages: List[Optional[float]] = [None] * n
        self.genders: List[Optional[str]] = [None] * n
        self.gender_scores: List[Optional[float]] = [None] * n

    @property
    def n_objects(self) -> int:
        """Total number of detected objects (faces + persons)."""
        return len(self.yolo_results.boxes)

    @property
    def n_faces(self) -> int:
        """Number of detected face boxes."""
        return len(self.get_bboxes_inds("face"))

    @property
    def n_persons(self) -> int:
        """Number of detected person boxes."""
        return len(self.get_bboxes_inds("person"))

    def get_bboxes_inds(self, category: str) -> List[int]:
        """Return indices of all bounding boxes matching *category*."""
        return [
            ind
            for ind, det in enumerate(self.yolo_results.boxes)
            if self.yolo_results.names[int(det.cls)] == category
        ]

    def get_distance_to_center(self, bbox_ind: int) -> float:
        """Euclidean distance from the bbox center to the image center."""
        im_h, im_w = self.yolo_results[bbox_ind].orig_shape
        x1, y1, x2, y2 = self.get_bbox_by_ind(bbox_ind).cpu().numpy()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return math.dist([cx, cy], [im_w / 2, im_h / 2])

    def plot(
        self,
        conf=False,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        labels=True,
        boxes=True,
        probs=True,
        ages=True,
        genders=True,
        gender_probs=False,
    ):
        """Render detection boxes and age/gender labels on the image.

        Args:
            conf: Whether to show confidence scores.
            line_width: Box outline width.
            font_size: Label font size.
            font: Label font path.
            pil: Use PIL rendering.
            img: Optional image to draw on (uses original if None).
            labels: Whether to show class labels.
            boxes: Whether to draw bounding boxes.
            probs: Whether to show probabilities.
            ages: Whether to show age predictions.
            genders: Whether to show gender predictions.
            gender_probs: Whether to show gender confidence.

        Returns:
            Annotated image as a numpy array.
        """
        colors_by_ind = {}
        for face_ind, person_ind in self.face_to_person_map.items():
            if person_ind is not None:
                colors_by_ind[face_ind] = face_ind + 2
                colors_by_ind[person_ind] = face_ind + 2
            else:
                colors_by_ind[face_ind] = 0
        for person_ind in self.unassigned_persons_inds:
            colors_by_ind[person_ind] = 1

        names = self.yolo_results.names
        annotator = Annotator(
            deepcopy(self.yolo_results.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil,
            example=names,
        )
        pred_boxes = self.yolo_results.boxes

        if pred_boxes and boxes:
            for bb_ind, (d, age, gender, gender_score) in enumerate(
                zip(pred_boxes, self.ages, self.genders, self.gender_scores),
            ):
                c, conf_val, guid = (
                    int(d.cls),
                    float(d.conf) if conf else None,
                    None if d.id is None else int(d.id.item()),
                )
                name = ("" if guid is None else f"id:{guid} ") + names[c]
                label = (f"{name} {conf_val:.2f}" if conf else name) if labels else None

                if ages and age is not None:
                    label += f" {age:.1f}"
                if genders and gender is not None:
                    label += f" {'F' if gender == 'female' else 'M'}"
                if gender_probs and gender_score is not None:
                    label += f" ({gender_score:.1f})"

                annotator.box_label(
                    d.xyxy.squeeze(),
                    label,
                    color=colors(colors_by_ind[bb_ind], True),
                )

        return annotator.result()

    def set_tracked_age_gender(
        self,
        tracked_objects: Dict[int, List[AGE_GENDER_TYPE]],
    ):
        """Apply temporal smoothing to age/gender using tracking history.

        Args:
            tracked_objects: Mapping of track GUID → list of (age, gender) pairs.
        """
        for face_ind, person_ind in self.face_to_person_map.items():
            pguid = self._get_id_by_ind(person_ind)
            fguid = self._get_id_by_ind(face_ind)
            if fguid == -1 and pguid == -1:
                continue
            age, gender = self._gather_tracking_result(tracked_objects, fguid, pguid)
            if age is None or gender is None:
                continue
            self.set_age(face_ind, age)
            self.set_gender(face_ind, gender, 1.0)
            if pguid != -1:
                self.set_gender(person_ind, gender, 1.0)
                self.set_age(person_ind, age)

        for person_ind in self.unassigned_persons_inds:
            pid = self._get_id_by_ind(person_ind)
            if pid == -1:
                continue
            age, gender = self._gather_tracking_result(tracked_objects, -1, pid)
            if age is None or gender is None:
                continue
            self.set_gender(person_ind, gender, 1.0)
            self.set_age(person_ind, age)

    def _get_id_by_ind(self, ind: Optional[int] = None) -> int:
        if ind is None:
            return -1
        obj_id = self.yolo_results.boxes[ind].id
        return -1 if obj_id is None else obj_id.item()

    def get_bbox_by_ind(
        self, ind: int, im_h: int = None, im_w: int = None
    ) -> torch.Tensor:
        """Return the bounding box tensor for detection *ind*.

        Optionally clamp coordinates to image dimensions.
        """
        bb = self.yolo_results.boxes[ind].xyxy.squeeze().type(torch.int32)
        if im_h is not None and im_w is not None:
            bb[0] = torch.clamp(bb[0], min=0, max=im_w - 1)
            bb[1] = torch.clamp(bb[1], min=0, max=im_h - 1)
            bb[2] = torch.clamp(bb[2], min=0, max=im_w - 1)
            bb[3] = torch.clamp(bb[3], min=0, max=im_h - 1)
        return bb

    def set_age(self, ind: Optional[int], age: float):
        """Assign an age prediction to detection *ind*."""
        if ind is not None:
            self.ages[ind] = age

    def set_gender(self, ind: Optional[int], gender: str, gender_score: float):
        """Assign a gender prediction to detection *ind*."""
        if ind is not None:
            self.genders[ind] = gender
            self.gender_scores[ind] = gender_score

    @staticmethod
    def _gather_tracking_result(
        tracked_objects: Dict[int, List[AGE_GENDER_TYPE]],
        fguid: int = -1,
        pguid: int = -1,
        minimum_sample_size: int = 10,
    ) -> AGE_GENDER_TYPE:
        """Aggregate age/gender from tracking history via winsorised voting."""
        assert fguid != -1 or pguid != -1

        face_ages = [r[0] for r in tracked_objects.get(fguid, []) if r[0] is not None]
        face_genders = [
            r[1] for r in tracked_objects.get(fguid, []) if r[1] is not None
        ]
        person_ages = [
            r[0] for r in tracked_objects.get(pguid, []) if r[0] is not None
        ]
        person_genders = [
            r[1] for r in tracked_objects.get(pguid, []) if r[1] is not None
        ]

        if not face_ages and not person_ages:
            return None, None

        all_ages = person_ages + face_ages
        if len(all_ages) >= minimum_sample_size:
            age = aggregate_votes_winsorized(all_ages)
        else:
            face_age = np.mean(face_ages) if face_ages else None
            person_age = np.mean(person_ages) if person_ages else None
            if face_age is None:
                face_age = person_age
            if person_age is None:
                person_age = face_age
            age = (face_age + person_age) / 2.0

        genders = face_genders + person_genders
        gender = max(set(genders), key=genders.count)
        return age, gender

    def get_results_for_tracking(
        self,
    ) -> Tuple[Dict[int, AGE_GENDER_TYPE], Dict[int, AGE_GENDER_TYPE]]:
        """Return per-track age/gender for active face and person tracks."""
        persons: Dict[int, AGE_GENDER_TYPE] = {}
        faces: Dict[int, AGE_GENDER_TYPE] = {}
        names = self.yolo_results.names
        for det, age, gender, _ in zip(
            self.yolo_results.boxes,
            self.ages,
            self.genders,
            self.gender_scores,
        ):
            if det.id is None:
                continue
            cat_id, guid = int(det.cls), int(det.id.item())
            name = names[cat_id]
            if name == "person":
                persons[guid] = (age, gender)
            elif name == "face":
                faces[guid] = (age, gender)
        return persons, faces

    def associate_faces_with_persons(self):
        """Run Hungarian matching to associate face and person detections."""
        face_bboxes_inds = self.get_bboxes_inds("face")
        person_bboxes_inds = self.get_bboxes_inds("person")
        face_bboxes = [self.get_bbox_by_ind(i) for i in face_bboxes_inds]
        person_bboxes = [self.get_bbox_by_ind(i) for i in person_bboxes_inds]

        self.face_to_person_map = {ind: None for ind in face_bboxes_inds}
        assigned, unassigned = assign_faces(person_bboxes, face_bboxes)

        for fi, pi in enumerate(assigned):
            face_ind = face_bboxes_inds[fi]
            person_ind = person_bboxes_inds[pi] if pi is not None else None
            self.face_to_person_map[face_ind] = person_ind

        self.unassigned_persons_inds = [
            person_bboxes_inds[pi] for pi in unassigned
        ]

    def crop_object(
        self,
        full_image: np.ndarray,
        ind: int,
        cut_other_classes: Optional[List[str]] = None,
    ) -> Optional[np.ndarray]:
        """Extract a cropped detection from *full_image*.

        Optionally masks out overlapping objects of specified classes
        (e.g. "face" / "person") so that person crops do not contain
        visible face regions.

        Args:
            full_image: The full input image.
            ind: Detection index to crop.
            cut_other_classes: List of class names to mask out.

        Returns:
            Cropped image or ``None`` if the crop is too small.
        """
        obj_bbox = self.get_bbox_by_ind(ind, *full_image.shape[:2])
        x1, y1, x2, y2 = obj_bbox
        cur_cat = self.yolo_results.names[int(self.yolo_results.boxes[ind].cls)]
        obj_image = full_image[y1:y2, x1:x2].copy()
        crop_h, crop_w = obj_image.shape[:2]

        if cur_cat == "person" and (
            crop_h < _MIN_PERSON_SIZE or crop_w < _MIN_PERSON_SIZE
        ):
            return None

        if not cut_other_classes:
            return obj_image

        other_bboxes = [
            self.get_bbox_by_ind(oi, *full_image.shape[:2])
            for oi in range(len(self.yolo_results.boxes))
        ]
        iou_matrix = (
            box_iou(torch.stack([obj_bbox]), torch.stack(other_bboxes))
            .cpu()
            .numpy()[0]
        )

        for oi, (det, iou_val) in enumerate(
            zip(self.yolo_results.boxes, iou_matrix),
        ):
            other_cat = self.yolo_results.names[int(det.cls)]
            if ind == oi or iou_val < _IOU_THRESH or other_cat not in cut_other_classes:
                continue
            ox1, oy1, ox2, oy2 = det.xyxy.squeeze().type(torch.int32)
            ox1 = max(ox1 - x1, 0)
            oy1 = max(oy1 - y1, 0)
            ox2 = min(ox2 - x1, crop_w)
            oy2 = min(oy2 - y1, crop_h)

            if other_cat != "face":
                if oy1 / crop_h < _CROP_ROUND_RATE:
                    oy1 = 0
                if (crop_h - oy2) / crop_h < _CROP_ROUND_RATE:
                    oy2 = crop_h
                if ox1 / crop_w < _CROP_ROUND_RATE:
                    ox1 = 0
                if (crop_w - ox2) / crop_w < _CROP_ROUND_RATE:
                    ox2 = crop_w

            obj_image[oy1:oy2, ox1:ox2] = 0

        remain_ratio = np.count_nonzero(obj_image) / (
            obj_image.shape[0] * obj_image.shape[1] * obj_image.shape[2]
        )
        return None if remain_ratio < _MIN_PERSON_AFTERCUT_RATIO else obj_image

    def collect_crops(self, image) -> PersonAndFaceCrops:
        """Extract all face and person crops from the image.

        Returns:
            A ``PersonAndFaceCrops`` container.
        """
        crops_data = PersonAndFaceCrops()
        for face_ind, person_ind in self.face_to_person_map.items():
            face_image = self.crop_object(image, face_ind, cut_other_classes=[])
            if person_ind is None:
                crops_data.crops_faces_wo_body[face_ind] = face_image
                continue
            person_image = self.crop_object(
                image,
                person_ind,
                cut_other_classes=["face", "person"],
            )
            crops_data.crops_faces[face_ind] = face_image
            crops_data.crops_persons[person_ind] = person_image

        for person_ind in self.unassigned_persons_inds:
            person_image = self.crop_object(
                image,
                person_ind,
                cut_other_classes=["face", "person"],
            )
            crops_data.crops_persons_wo_face[person_ind] = person_image

        return crops_data
