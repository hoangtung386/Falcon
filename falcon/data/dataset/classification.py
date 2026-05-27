"""Classification-style datasets (FairFace, Adience) that predict age groups."""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from .dataset import AgeGenderDataset

__all__ = ["ClassificationDataset", "FairFaceDataset", "AdienceDataset"]


class ClassificationDataset(AgeGenderDataset):
    """Base class for datasets where age is an ordered category."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_dtype = torch.int32

    def set_age_classes(self) -> Optional[List[str]]:
        raise NotImplementedError

    def parse_target(self, age: str, gender: str) -> List[Any]:
        assert self.age_classes is not None
        if age != "-1":
            assert age in self.age_classes, f"Unknown age category {age}"
            age_ind = self.age_classes.index(age)
        else:
            age_ind = -1
        return [age_ind, int(self.parse_gender(gender))]


class FairFaceDataset(ClassificationDataset):
    """FairFace dataset with 9 age groups."""

    def set_age_classes(self) -> Optional[List[str]]:
        age_classes = [
            "0;2", "3;9", "10;19", "20;29",
            "30;39", "40;49", "50;59", "60;69", "70;120",
        ]
        self._intervals = torch.tensor([0, 3, 10, 20, 30, 40, 50, 60, 70])
        return age_classes


class AdienceDataset(ClassificationDataset):
    """Adience dataset with 8 age groups."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_dtype = torch.int32

    def set_age_classes(self) -> Optional[List[str]]:
        age_classes = [
            "0;2", "4;6", "8;12", "15;20",
            "25;32", "38;43", "48;53", "60;100",
        ]
        self._intervals = torch.tensor([0, 4, 7, 14, 24, 36, 46, 57])
        return age_classes
