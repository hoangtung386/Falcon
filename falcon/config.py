"""Configuration dataclass for model and training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

import torch

__all__ = ["ModelConfig"]


@dataclass
class ModelConfig:
    """Central configuration for Falcon model, data, and training settings.

    Attributes:
        input_size: Input image size (height and width, assumed square).
        max_age: Maximum age value for normalization.
        min_age: Minimum age value for normalization.
        avg_age: Average age value (used for denormalization).
        with_persons_model: Whether the model accepts both face and body crops.
        use_persons: Whether to use person (body) crops at inference.
        disable_faces: Whether to skip face crops entirely.
        only_age: Whether to only predict age (skip gender head).
        num_classes: Total number of output classes.
        num_classes_gender: Number of gender classes (default 2).
        in_chans: Number of input channels (3 for face-only, 6 for face+body).
        mean: Image normalization mean.
        std: Image normalization std.
        crop_pct: Crop percentage for data preprocessing.
        crop_mode: Crop mode (center, etc.).
        device: Torch device for computation.
        age_head_mode: Age prediction head type.
        num_age_bins: Number of bins for distribution-based age head.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        epochs: Number of training epochs.
        warmup_epochs: Number of warmup epochs.
        age_loss_weight: Weight for age loss component.
        gender_loss_weight: Weight for gender loss component.
        random_erasing_prob: Probability of random erasing augmentation.
        mixup_alpha: Alpha parameter for mixup augmentation.
    """

    input_size: int = 224
    max_age: float = 100.0
    min_age: float = 0.0
    avg_age: float = 50.0
    with_persons_model: bool = False
    use_persons: bool = True
    disable_faces: bool = False
    only_age: bool = False
    num_classes: int = 3
    num_classes_gender: int = 2
    in_chans: int = 3
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    crop_pct: float = 1.0
    crop_mode: str = "center"
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    age_head_mode: Literal["regression", "distribution", "ordinal"] = "regression"
    num_age_bins: int = 101

    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 100
    warmup_epochs: int = 5

    age_loss_weight: float = 1.0
    gender_loss_weight: float = 1.0

    random_erasing_prob: float = 0.25
    mixup_alpha: float = 0.2

    @property
    def use_person_crops(self) -> bool:
        """Whether person (body) crops are active."""
        return self.with_persons_model and self.use_persons

    @property
    def use_face_crops(self) -> bool:
        """Whether face crops are active."""
        return not self.disable_faces or not self.with_persons_model
