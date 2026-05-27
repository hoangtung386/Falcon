"""Image-level preprocessing: letterbox resizing, classification image preparation."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ["class_letterbox", "prepare_classification_images", "split_batch"]


def class_letterbox(
    im: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = (640, 640),
    color: Tuple[int, int, int] = (0, 0, 0),
    scaleup: bool = True,
) -> np.ndarray:
    """Resize and pad an image to *new_shape* while preserving aspect ratio.

    Args:
        im: Input image (H, W, C).
        new_shape: Target (height, width) or a single int.
        color: Padding color (B, G, R).
        scaleup: Whether to allow upscaling (otherwise only downscale).

    Returns:
            Padded and resized image.
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if shape[0] == new_shape[0] and shape[1] == new_shape[1]:
        return im

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    interpolation = cv2.INTER_AREA if r < 0.5 else cv2.INTER_LINEAR
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=interpolation)

    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im


def prepare_classification_images(
    img_list: List[Optional[np.ndarray]],
    target_size: int = 224,
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Convert a list of image crops into a normalised batched tensor.

    ``None`` entries are replaced with a constant ``-1.0`` tensor so that
    the batch maintains a fixed size.

    Args:
        img_list: List of BGR images (or ``None``).
        target_size: Output spatial size.
        mean: Normalisation mean per channel.
        std: Normalisation std per channel.
        device: Target device for the output tensor.

    Returns:
            Batched float32 tensor of shape (B, 3, target_size, target_size)
            or ``None`` if the input list is empty.
    """
    prepared: List[torch.Tensor] = []
    for img in img_list:
        if img is None:
            placeholder = torch.full(
                (3, target_size, target_size), -1.0, dtype=torch.float32
            )
            prepared.append(placeholder.unsqueeze(0))
            continue

        img = class_letterbox(img, new_shape=(target_size, target_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = (img - mean) / std
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        prepared.append(torch.from_numpy(img).unsqueeze(0))

    if not prepared:
        return None
    result = torch.concat(prepared)
    if device:
        result = result.to(device)
    return result


def split_batch(bs: int, dev: int) -> Tuple[int, int]:
    """Split a batch size into full chunks and remainder.

    Args:
        bs: Desired batch size.
        dev: Divisor (e.g. number of GPUs).

    Returns:
        Tuple ``(full_bs, part_bs)``.
    """
    full_bs = (bs // dev) * dev
    part_bs = bs - full_bs
    return full_bs, part_bs
