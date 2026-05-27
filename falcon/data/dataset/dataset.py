"""PyTorch Dataset for age + gender regression tasks."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Set

import cv2
import numpy as np
import torch
from falcon.data.dataset.reader import ReaderAgeGender
from PIL import Image
from torchvision import transforms as T

_logger = logging.getLogger("AgeGenderDataset")

__all__ = ["AgeGenderDataset", "convert_to_pil"]


class AgeGenderDataset(torch.utils.data.Dataset):
    """Dataset for age regression (and optional gender classification).

    Expects annotations in Falcon CSV format as parsed by
    :class:`ReaderAgeGender`. Returns normalised age values and gender
    indices.
    """

    def __init__(
        self,
        images_path: str,
        annotations_path: str,
        name: str = None,
        split: str = "train",
        load_bytes: bool = False,
        img_mode: str = "RGB",
        transform=None,
        is_training: bool = False,
        seed: int = 1234,
        target_size: int = 224,
        min_age: float = None,
        max_age: float = None,
        model_with_persons: bool = False,
        use_persons: bool = False,
        disable_faces: bool = False,
        only_age: bool = False,
    ):
        reader = ReaderAgeGender(
            images_path,
            annotations_path,
            split=split,
            seed=seed,
            target_size=target_size,
            with_persons=use_persons,
            disable_faces=disable_faces,
            only_age=only_age,
        )

        self.name = name
        self.model_with_persons = model_with_persons
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self._consecutive_errors = 0
        self.is_training = is_training
        self.random_flip = 0.0

        self.max_age: float = None
        self.min_age: float = None
        self.avg_age: float = None
        self.set_ages_min_max(min_age, max_age)

        self.genders = ["M", "F"]
        self.num_classes_gender = len(self.genders)
        self.age_classes: Optional[List[str]] = self.set_age_classes()
        self.num_classes_age = 1 if self.age_classes is None else len(self.age_classes)
        self.num_classes: int = self.num_classes_age + self.num_classes_gender
        self.target_dtype = torch.float32

    def set_age_classes(self) -> Optional[List[str]]:
        """Override in subclasses to return age category labels.

        Returns:
            ``None`` for regression, a list of age-class names otherwise.
        """
        return None

    def set_ages_min_max(self, min_age: Optional[float], max_age: Optional[float]):
        """Compute age normalisation statistics from annotations.

        When *min_age* and *max_age* are both provided they are used directly,
        otherwise they are inferred from the dataset.
        """
        assert (min_age is None) == (max_age is None), (
            "Both min and max age must be passed or none of them"
        )

        if max_age is not None and min_age is not None:
            _logger.info(
                f"Received predefined min_age {min_age} and max_age {max_age}"
            )
            self.max_age = max_age
            self.min_age = min_age
        else:
            all_ages: Set[int] = set()
            for _img_path, image_samples in self.reader._ann.items():
                for sample in image_samples:
                    if sample.age == "-1":
                        continue
                    all_ages.add(round(float(sample.age)))
            self.max_age = float(max(all_ages))
            self.min_age = float(min(all_ages))

        self.avg_age = (self.max_age + self.min_age) / 2.0

    def _norm_age(self, age: float) -> float:
        """Normalise a raw age into the ``[0, 1]`` range."""
        return (age - self.min_age) / (self.max_age - self.min_age)

    @staticmethod
    def parse_gender(_gender: str) -> float:
        """Parse a gender string into a numeric label.

        Returns:
            ``0.0`` for male, ``1.0`` for female, ``-1.0`` if missing.
        """
        if _gender != "-1":
            return float(0 if _gender in ("M", "0") else 1)
        return -1.0

    def parse_target(self, age: str, gender: str) -> List[Any]:
        """Convert raw annotation strings into numeric targets."""
        if age != "-1":
            age_val = self._norm_age(round(float(age)))
        else:
            age_val = -1.0
        return [age_val, self.parse_gender(gender)]

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        if not transform:
            return
        filtered = []
        for trans in transform.transforms:
            cls_name = type(trans).__name__
            if "Resize" in cls_name or "Crop" in cls_name:
                continue
            filtered.append(trans)
        self._transform = T.Compose(filtered)

    def apply_transforms(self, image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply torchvision transforms to an OpenCV image."""
        if image is None or self.transform is None:
            return image
        pil_image = convert_to_pil(image, self.img_mode)
        for trans in self.transform.transforms:
            pil_image = trans(pil_image)
        return pil_image

    def __getitem__(self, index: int):
        images, target = self.reader[index]
        target = self.parse_target(*target)

        if self.model_with_persons:
            face_image, person_image = images
            person_image = self.apply_transforms(person_image)
        else:
            face_image = images[0]
            person_image = None

        face_image = self.apply_transforms(face_image)

        if person_image is not None:
            img = np.concatenate([face_image, person_image], axis=0)
        else:
            img = face_image

        return img, target

    def __len__(self) -> int:
        return len(self.reader)

    def filename(self, index: int, basename: bool = False, absolute: bool = False) -> str:
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename: bool = False, absolute: bool = False) -> List[str]:
        return self.reader.filenames(basename, absolute)


def convert_to_pil(cv_im: Optional[np.ndarray], img_mode: str = "RGB") -> Image.Image:
    """Convert an OpenCV BGR image to a PIL Image in *img_mode*."""
    if cv_im is None:
        return None
    if img_mode == "RGB":
        cv_im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Incorrect image mode: {img_mode}")
    cv_im = np.ascontiguousarray(cv_im)
    return Image.fromarray(cv_im)
