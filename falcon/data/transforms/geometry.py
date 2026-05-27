"""Bounding-box geometry utilities: IoU computation and Hungarian assignment."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from scipy.optimize import linear_sum_assignment

__all__ = ["box_iou", "iou", "assign_faces"]


def box_iou(
    box1: torch.Tensor, box2: torch.Tensor, over_second: bool = False
) -> torch.Tensor:
    """Compute IoU between two sets of axis-aligned boxes.

    If *over_second* is ``True``, returns the average of standard IoU and
    the ratio ``intersection / area2`` (softened version for matching).

    Args:
        box1: (N, 4) tensor of ``[x1, y1, x2, y2]`` boxes.
        box2: (M, 4) tensor of ``[x1, y1, x2, y2]`` boxes.
        over_second: Blend standard IoU with overlap-over-area2.

    Returns:
        (N, M) tensor of pairwise IoU values.
    """

    def box_area(box: torch.Tensor) -> torch.Tensor:
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (
        torch.min(box1[:, None, 2:], box2[:, 2:])
        - torch.max(box1[:, None, :2], box2[:, :2])
    ).clamp(0).prod(2)

    iou = inter / (area1[:, None] + area2 - inter)
    if over_second:
        return (inter / area2 + iou) / 2
    return iou


def iou(
    bb1: Union[Tuple, List],
    bb2: Union[Tuple, List],
    norm_second_bbox: bool = False,
) -> float:
    """Compute IoU for two single boxes given as (y1, x1, y2, x2).

    Args:
        bb1: First box ``(y1, x1, y2, x2)``.
        bb2: Second box ``(y1, x1, y2, x2)``.
        norm_second_bbox: If ``True``, normalise by the area of *bb2* only.

    Returns:
        IoU value in [0, 1].
    """
    assert bb1[1] < bb1[3] and bb1[0] < bb1[2]
    assert bb2[1] < bb2[3] and bb2[0] < bb2[2]

    x_left = max(bb1[1], bb2[1])
    y_top = max(bb1[0], bb2[0])
    x_right = min(bb1[3], bb2[3])
    y_bottom = min(bb1[2], bb2[2])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[3] - bb1[1]) * (bb1[2] - bb1[0])
    bb2_area = (bb2[3] - bb2[1]) * (bb2[2] - bb2[0])

    if not norm_second_bbox:
        iou_val = intersection_area / float(bb1_area + bb2_area - intersection_area)
    else:
        iou_val = intersection_area / float(bb2_area)

    assert 0.0 <= iou_val <= 1.01
    return iou_val


def assign_faces(
    persons_bboxes: List[torch.Tensor],
    faces_bboxes: List[torch.Tensor],
    iou_thresh: float = 0.0001,
) -> Tuple[List[Optional[int]], List[int]]:
    """Match face boxes to person boxes using Hungarian algorithm.

    Args:
        persons_bboxes: List of person bounding-box tensors.
        faces_bboxes: List of face bounding-box tensors.
        iou_thresh: Minimum IoU to accept a match.

    Returns:
        Tuple ``(assigned_faces, unassigned_persons)`` where
        ``assigned_faces[i]`` is the person index matched to face *i*
        (or ``None``), and ``unassigned_persons`` lists unmatched person
        indices.
    """
    assigned_faces: List[Optional[int]] = [None] * len(faces_bboxes)
    unassigned_persons: List[int] = list(range(len(persons_bboxes)))

    if not persons_bboxes or not faces_bboxes:
        return assigned_faces, unassigned_persons

    cost_matrix = (
        box_iou(
            torch.stack(persons_bboxes), torch.stack(faces_bboxes), over_second=True
        )
        .cpu()
        .numpy()
    )
    matched_persons = set()
    if cost_matrix.size > 0:
        p_indices, f_indices = linear_sum_assignment(cost_matrix, maximize=True)
        for p_idx, f_idx in zip(p_indices, f_indices):
            if cost_matrix[p_idx, f_idx] > iou_thresh and p_idx not in matched_persons:
                assigned_faces[f_idx] = p_idx
                matched_persons.add(p_idx)

    unassigned_persons = [
        p for p in range(len(persons_bboxes)) if p not in matched_persons
    ]
    return assigned_faces, unassigned_persons
