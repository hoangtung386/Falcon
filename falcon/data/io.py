"""I/O utilities: file discovery, input-type detection, CSV annotation parsing."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

__all__ = [
    "PictureInfo",
    "AnnotType",
    "InputType",
    "get_all_files",
    "get_input_type",
    "natural_key",
    "read_csv_annotation_file",
]

IMAGES_EXT: Tuple[str, ...] = (".jpeg", ".jpg", ".png", ".webp", ".bmp", ".gif")
VIDEO_EXT: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class PictureInfo:
    """Metadata for a single image sample.

    Attributes:
        image_path: Absolute path to the image.
        age: Age label (string, ``-1`` for missing).
        gender: Gender label (string, ``-1`` for missing).
        bbox: Face bounding box ``[x0, y0, x1, y1]``.
        person_bbox: Person bounding box ``[x0, y0, x1, y1]``.
    """

    image_path: str
    age: Optional[str]
    gender: Optional[str]
    bbox: List[int] = field(default_factory=lambda: [-1, -1, -1, -1])
    person_bbox: List[int] = field(default_factory=lambda: [-1, -1, -1, -1])

    @property
    def has_person_bbox(self) -> bool:
        """Whether a person bounding box is present."""
        return any(c != -1 for c in self.person_bbox)

    @property
    def has_face_bbox(self) -> bool:
        """Whether a face bounding box is present."""
        return any(c != -1 for c in self.bbox)

    def has_gt(self, only_age: bool = False) -> bool:
        """Check if required ground-truth labels are available.

        Args:
            only_age: If ``True``, only the age label is required.

        Returns:
            ``True`` if the required labels are set.
        """
        if only_age:
            return self.age != "-1"
        return not (self.age == "-1" and self.gender == "-1")

    def clear_person_bbox(self):
        """Reset person bounding box to ``[-1, -1, -1, -1]``."""
        self.person_bbox = [-1, -1, -1, -1]

    def clear_face_bbox(self):
        """Reset face bounding box to ``[-1, -1, -1, -1]``."""
        self.bbox = [-1, -1, -1, -1]


class AnnotType(Enum):
    """Annotation file format type."""

    ORIGINAL = "original"
    PERSONS = "persons"
    NONE = "none"

    @classmethod
    def _missing_(cls, value):
        print(f"WARN: Unknown annotation type {value}.")
        return AnnotType.NONE


class InputType(Enum):
    """Type of input source for inference."""

    IMAGE = 0
    VIDEO = 1
    VIDEO_STREAM = 2


def get_all_files(path: str, extensions: Tuple[str, ...] = IMAGES_EXT) -> List[str]:
    """Recursively collect all files matching *extensions* under *path*.

    Args:
        path: Root directory to search.
        extensions: Tuple of valid extensions (e.g. ``.jpg``).

    Returns:
        Sorted list of matching file paths.
    """
    files = []
    for root, _subdirs, fnames in os.walk(path):
        for name in fnames:
            if "directory" in name:
                continue
            if any(ext.lower() in name.lower() for ext in extensions):
                files.append(os.path.join(root, name))
    return files


def get_input_type(input_path: str) -> InputType:
    """Determine the input type from a path or URL string.

    Args:
        input_path: Local path, directory, or URL.

    Returns:
        The detected ``InputType``.

    Raises:
        ValueError: If the input cannot be classified.
    """
    if os.path.isdir(input_path):
        print("Input is a folder, only images will be processed")
        return InputType.IMAGE
    if os.path.isfile(input_path):
        if input_path.endswith(VIDEO_EXT):
            return InputType.VIDEO
        if input_path.endswith(IMAGES_EXT):
            return InputType.IMAGE
        raise ValueError(
            f"Unsupported file format {input_path}. "
            f"Supported video: {VIDEO_EXT}, images: {IMAGES_EXT}"
        )
    if input_path.startswith("http"):
        if input_path.endswith(IMAGES_EXT):
            return InputType.IMAGE
        return InputType.VIDEO_STREAM
    raise ValueError(f"Unknown input {input_path}")


def natural_key(string_: str) -> List:
    """Split a string into a list suitable for natural (human) sorting.

    Example: ``"img10.jpg"`` → ``["img", 10, ".jpg"]``.
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def read_csv_annotation_file(
    annotation_file: str,
    images_dir: str,
    ignore_without_gt: bool = False,
) -> Tuple[Dict[str, List[PictureInfo]], AnnotType]:
    """Parse a Falcon-format CSV annotation file.

    The CSV is expected to contain at least the columns:
    ``img_name, face_x0, face_y0, face_x1, face_y1, age, gender``.
    If ``person_x0``… columns exist the type is ``PERSONS``.

    Args:
        annotation_file: Path to the CSV file.
        images_dir: Root directory containing the images.
        ignore_without_gt: Skip rows missing ground truth.

    Returns:
        Tuple ``(bboxes_per_image, annot_type)``.
    """
    bboxes_per_image: Dict[str, List[PictureInfo]] = defaultdict(list)
    df = pd.read_csv(annotation_file, sep=",")
    annot_type = AnnotType.PERSONS if "person_x0" in df.columns else AnnotType.ORIGINAL
    print(f"Reading {annotation_file} (type: {annot_type})...")

    missing_images = 0
    for _idx, row in df.iterrows():
        img_path = os.path.join(images_dir, row["img_name"])
        if not os.path.exists(img_path):
            missing_images += 1
            continue

        age, gender = str(row["age"]), str(row["gender"])
        if ignore_without_gt and (age == "-1" or gender == "-1"):
            continue

        face_bbox = list(
            map(int, [row["face_x0"], row["face_y0"], row["face_x1"], row["face_y1"]])
        )
        person_bbox = (
            list(
                map(
                    int,
                    [
                        row["person_x0"],
                        row["person_y0"],
                        row["person_x1"],
                        row["person_y1"],
                    ],
                )
            )
            if annot_type == AnnotType.PERSONS
            else [-1, -1, -1, -1]
        )
        pic_info = PictureInfo(img_path, age, gender, face_bbox, person_bbox)
        bboxes_per_image[img_path].append(pic_info)

    if missing_images:
        print(f"WARNING: Missing images: {missing_images}/{len(df)}")
    return bboxes_per_image, annot_type
