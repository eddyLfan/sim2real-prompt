"""Build auditable first-frame crops for Phantom-style Multi-Reference training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MediaConfig
from .lerobot import SampleRecord
from .media import (
    _encode_full_resolution,
    _probe_video,
    _read_raw_frames,
    _resize_and_encode,
)
from .models import ReferenceCandidate


@dataclass(frozen=True)
class ReferenceArtifact:
    path: Path
    jpeg: bytes
    row: dict[str, object]


_SCOPE_ORDER = {
    "objects": 0,
    "robot": 1,
    "workspace": 2,
    "environment": 3,
    "background": 4,
}


def _pixel_box(
    candidate: ReferenceCandidate,
    *,
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = candidate.bbox_xyxy
    left = x1 * width / 1000
    top = y1 * height / 1000
    right = x2 * width / 1000
    bottom = y2 * height / 1000
    pad_x = (right - left) * padding
    pad_y = (bottom - top) * padding
    return (
        max(0, int(np.floor(left - pad_x))),
        max(0, int(np.floor(top - pad_y))),
        min(width, int(np.ceil(right + pad_x))),
        min(height, int(np.ceil(bottom + pad_y))),
    )


def build_reference_artifacts(
    record: SampleRecord,
    candidates: list[ReferenceCandidate],
    config: MediaConfig,
    *,
    directory_name: str = "Reference",
    full_resolution: bool = True,
    jpeg_quality: int = 95,
) -> list[ReferenceArtifact]:
    """Crop every valid candidate from Real frame zero; no whole-frame fallback."""

    view = config.reference_view
    if view not in record.real_videos:
        view = record.paired_views[0]
    source = record.real_videos[view]
    _probe_video(source)
    frame = _read_raw_frames(source, [0])[0]
    height, width = frame.shape[:2]
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (_SCOPE_ORDER[item[1].scope], item[0]),
    )
    artifacts: list[ReferenceArtifact] = []
    seen: set[tuple[str, str, tuple[int, int, int, int]]] = set()
    for _, candidate in ranked:
        if candidate.confidence < config.reference_min_confidence:
            continue
        identity = (candidate.scope, candidate.label.lower(), candidate.bbox_xyxy)
        if identity in seen:
            continue
        seen.add(identity)
        x1, y1, x2, y2 = _pixel_box(
            candidate,
            width=width,
            height=height,
            padding=config.reference_crop_padding,
        )
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        crop = frame[y1:y2, x1:x2]
        jpeg = (
            _encode_full_resolution(crop, jpeg_quality)
            if full_resolution
            else _resize_and_encode(
                crop, config.model_copy(update={"jpeg_quality": jpeg_quality})
            )
        )
        digest = hashlib.sha256(jpeg).hexdigest()
        position = len(artifacts)
        relative = (
            Path(directory_name)
            / f"episode_{record.episode_index:06d}"
            / f"reference_{position:02d}.jpg"
        )
        artifacts.append(
            ReferenceArtifact(
                path=record.dataset_root / relative,
                jpeg=jpeg,
                row={
                    "reference_id": f"sha256:{digest}",
                    "reference_path": relative.as_posix(),
                    "source_view": view,
                    "source_frame_index": 0,
                    "scope": candidate.scope,
                    "label": candidate.label,
                    "description": candidate.description,
                    "bbox_xyxy": list(candidate.bbox_xyxy),
                    "crop_xyxy": [x1, y1, x2, y2],
                    "confidence": candidate.confidence,
                    "sha256": digest,
                },
            )
        )
        if len(artifacts) >= config.reference_pool_max_images:
            break
    if not artifacts:
        raise ValueError(f"{record.sample_id}: no valid first-frame Reference crop")
    return artifacts
