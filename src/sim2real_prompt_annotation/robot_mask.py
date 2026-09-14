"""Shared protocol and geometry helpers for robot-mask backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .models import RobotMaskPrediction


class RobotMaskSegmenter(Protocol):
    """Minimal surface implemented by RobotSeg, YOLOE, and CPU-only test fakes."""

    def ensure_ready(self) -> None:
        """Load and validate the configured runtime and local checkpoint."""

    def cache_identity(self) -> dict[str, object]:
        """Return every stable setting that can change predicted masks."""

    def predict(
        self,
        frames: Sequence[np.ndarray],
        queries: Sequence[str] = ("robot",),
    ) -> list[list[RobotMaskPrediction]]:
        """Return zero or more full-resolution masks for each input frame."""


def validate_bgr_frame(value: np.ndarray) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 3
        or value.shape[2] != 3
        or value.size == 0
        or value.dtype != np.uint8
    ):
        raise ValueError("expected a non-empty uint8 HxWx3 BGR frame")
    return value


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the tight exclusive-end bbox of a binary mask."""

    mask = np.asarray(mask) > 0
    if mask.ndim != 2 or not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalize_robot_queries(values: Sequence[object]) -> tuple[str, ...]:
    """Normalize strings or objects/mappings exposing ``query``/``label``."""

    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if isinstance(raw, str):
            value = raw
        elif isinstance(raw, dict):
            value = str(raw.get("query") or raw.get("label") or "")
        else:
            value = str(getattr(raw, "query", getattr(raw, "label", "")))
        value = " ".join(value.split()).strip(" ,.;:")
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    if not result:
        raise ValueError("at least one robot segmentation query is required")
    return tuple(result)
