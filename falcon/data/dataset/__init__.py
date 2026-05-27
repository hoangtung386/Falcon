"""Dataset construction pipeline: builds a Dataset + DataLoader from config."""

from __future__ import annotations

from typing import Tuple

import torch
from falcon.config import ModelConfig
from falcon.data.dataset.classification import AdienceDataset, FairFaceDataset
from falcon.data.dataset.dataset import AgeGenderDataset
from falcon.data.dataset.loader import create_loader

__all__ = ["build", "DATASET_CLASS_MAP"]

DATASET_CLASS_MAP = {
    "utk": AgeGenderDataset,
    "lagenda": AgeGenderDataset,
    "imdb": AgeGenderDataset,
    "agedb": AgeGenderDataset,
    "cacd": AgeGenderDataset,
    "adience": AdienceDataset,
    "fairface": FairFaceDataset,
}


def build(
    name: str,
    images_path: str,
    annotations_path: str,
    split: str,
    model_config: ModelConfig,
    workers: int,
    batch_size: int,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    """Build a dataset and its corresponding data loader.

    Args:
        name: Dataset key (must be in ``DATASET_CLASS_MAP``).
        images_path: Root directory of images.
        annotations_path: Directory containing CSV annotation files.
        split: Comma-separated split names (e.g. ``"train"``, ``"val"``).
        model_config: Global model configuration.
        workers: Number of data-loading workers.
        batch_size: Batch size.

    Returns:
        Tuple ``(dataset, loader)``.
    """
    dataset_class = DATASET_CLASS_MAP[name]
    dataset: torch.utils.data.Dataset = dataset_class(
        images_path=images_path,
        annotations_path=annotations_path,
        name=name,
        split=split,
        target_size=model_config.input_size,
        max_age=model_config.max_age,
        min_age=model_config.min_age,
        model_with_persons=model_config.with_persons_model,
        use_persons=model_config.use_persons,
        disable_faces=model_config.disable_faces,
        only_age=model_config.only_age,
    )

    in_chans = 3 if not model_config.with_persons_model else 6
    input_size = (in_chans, model_config.input_size, model_config.input_size)

    dataset_loader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=batch_size,
        mean=model_config.mean,
        std=model_config.std,
        num_workers=workers,
        crop_pct=model_config.crop_pct,
        crop_mode=model_config.crop_mode,
        pin_memory=False,
        device=model_config.device,
        target_type=dataset.target_dtype,
    )

    return dataset, dataset_loader
