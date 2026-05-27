"""Type aliases used across the Falcon codebase."""

from __future__ import annotations

from typing import Tuple

__all__ = ["AGE_GENDER_TYPE"]

AGE_GENDER_TYPE = Tuple[float, str]
"""A predicted age (float) paired with a predicted gender label (str)."""
