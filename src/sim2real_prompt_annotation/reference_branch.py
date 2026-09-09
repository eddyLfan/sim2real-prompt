"""YOLOE-only first-frame Reference extraction branch."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import ReferenceConfig
from .models import (
    Detection,
    ReferenceArtifact,
    ReferenceBranchResult,
    ReferenceQuery,
)
from .yoloe import DetectionRequest, YOLOEDetector, normalize_queries

REFERENCE_DIRECTORY = "Reference"


class NoValidReferenceError(RuntimeError):
    """Raised when YOLOE produces no publishable first-frame crop."""


class ReferenceDetectorProtocol(Protocol):
    """Detector surface used by the branch and its CPU-only test fakes."""

    def predict(
        self,
        frames: Sequence[np.ndarray],
        queries: Sequence[object],
    ) -> list[list[Detection]]:
        """Predict a batch whose frames share one query vocabulary."""


@dataclass(frozen=True, slots=True)
class ReferenceBranchInput:
    """One independently processable first-frame job."""

    sample_id: str
    episode_index: int
    frame0: np.ndarray | bytes
    queries: Sequence[object]
    source_view: str = "camera_head"


@dataclass(frozen=True, slots=True)
class _CandidateCrop:
    detection: Detection
    crop_xyxy: tuple[int, int, int, int]
    jpeg: bytes
    sha256: str
    mask_area_fraction: float


def _decode_frame(frame0: np.ndarray | bytes) -> np.ndarray:
    if isinstance(frame0, bytes):
        encoded = np.frombuffer(frame0, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Real frame zero is not a decodable JPEG image")
        return decoded
    if not isinstance(frame0, np.ndarray):
        raise TypeError("Real frame zero must be a numpy array or encoded JPEG bytes")
    if frame0.ndim != 3 or frame0.shape[2] not in (3, 4):
        raise ValueError("Real frame zero must have shape HxWx3 or HxWx4")
    if frame0.shape[0] == 0 or frame0.shape[1] == 0:
        raise ValueError("Real frame zero must be non-empty")
    return frame0


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _priority(detection: Detection) -> tuple[object, ...]:
    role_order = {
        "primary": 0,
        "destination": 1,
        "secondary": 2,
        "robot": 3,
        "workspace": 4,
        "environment": 5,
        "background": 6,
    }
    return (
        0 if detection.primary else 1,
        0 if detection.required else 1,
        role_order[detection.role],
        -detection.confidence,
        detection.query.casefold(),
        detection.bbox_xyxy,
    )


def _selection_rng(seed: int, sample_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{sample_id}".encode()).digest()
    return random.Random(int.from_bytes(digest, byteorder="big", signed=False))


class ReferenceBranch:
    """Filter YOLOE masks, retain a candidate pool, and select 1--3 crops."""

    def __init__(
        self,
        detector: ReferenceDetectorProtocol,
        config: ReferenceConfig,
        *,
        min_area_fraction: float = 0.0005,
        max_area_fraction: float = 0.85,
        duplicate_iou: float | None = None,
    ) -> None:
        if not 0.0 <= min_area_fraction < max_area_fraction <= 1.0:
            raise ValueError("area fractions must satisfy 0 <= min < max <= 1")
        if duplicate_iou is not None and not 0.0 <= duplicate_iou <= 1.0:
            raise ValueError("duplicate_iou must be in [0, 1]")
        self.detector = detector
        self.config = config
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.duplicate_iou = (
            config.duplicate_iou if duplicate_iou is None else duplicate_iou
        )

    @classmethod
    def from_config(cls, config: ReferenceConfig) -> ReferenceBranch:
        """Build the sole supported detector backend without initializing it."""

        detector = YOLOEDetector(
            config.model_path,
            device=config.device,
            image_size=config.image_size,
            confidence=config.confidence,
            iou_threshold=config.iou_threshold,
            embedding_cache_size=config.embedding_cache_size,
        )
        return cls(detector, config)

    def ensure_ready(self) -> None:
        """Preload the configured backend before the Prompt branch spends API calls."""

        ensure = getattr(self.detector, "ensure_ready", None)
        if ensure is not None:
            ensure()

    def cache_identity(self) -> dict[str, object]:
        """Return detector and post-processing inputs that affect saved crops."""

        detector_identity = getattr(self.detector, "cache_identity", None)
        return {
            "backend": "yoloe",
            "detector": (
                detector_identity()
                if detector_identity is not None
                else type(self.detector).__qualname__
            ),
            "crop_padding": self.config.crop_padding,
            "candidate_pool_size": self.config.candidate_pool_size,
            "min_images": self.config.min_images,
            "max_images": self.config.max_images,
            "selection_seed": self.config.selection_seed,
            "jpeg_quality": self.config.jpeg_quality,
            "min_area_fraction": self.min_area_fraction,
            "max_area_fraction": self.max_area_fraction,
            "duplicate_iou": self.duplicate_iou,
        }

    def _fingerprint(
        self,
        frame: np.ndarray,
        sample_id: str,
        queries: tuple[ReferenceQuery, ...],
    ) -> str:
        settings = {
            "sample_id": sample_id,
            "queries": [query.model_dump(mode="json") for query in queries],
            "model_path": str(self.config.model_path),
            "image_size": self.config.image_size,
            "confidence": self.config.confidence,
            "iou_threshold": self.config.iou_threshold,
            "crop_padding": self.config.crop_padding,
            "candidate_pool_size": self.config.candidate_pool_size,
            "min_images": self.config.min_images,
            "max_images": self.config.max_images,
            "selection_seed": self.config.selection_seed,
            "jpeg_quality": self.config.jpeg_quality,
            "min_area_fraction": self.min_area_fraction,
            "max_area_fraction": self.max_area_fraction,
            "duplicate_iou": self.duplicate_iou,
        }
        digest = hashlib.sha256()
        digest.update(str(frame.shape).encode("ascii"))
        digest.update(str(frame.dtype).encode("ascii"))
        digest.update(frame.tobytes(order="C"))
        digest.update(
            json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        )
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _mask_area_fraction(detection: Detection) -> float:
        if detection.mask_polygon is None:
            return 0.0
        polygon = np.asarray(detection.mask_polygon, dtype=np.float32)
        area = abs(float(cv2.contourArea(polygon)))
        return area / float(detection.image_width * detection.image_height)

    def _filter_detections(
        self,
        detections: Sequence[Detection],
        *,
        width: int,
        height: int,
    ) -> tuple[list[Detection], list[str]]:
        rejected: list[str] = []
        eligible: list[Detection] = []
        for detection in detections:
            identity = f"{detection.query}@{detection.bbox_xyxy}"
            if detection.image_width != width or detection.image_height != height:
                rejected.append(f"{identity}: source dimensions do not match frame0")
                continue
            if detection.confidence < self.config.confidence:
                rejected.append(f"{identity}: confidence below threshold")
                continue
            if detection.mask_polygon is None:
                rejected.append(f"{identity}: YOLOE segmentation mask is missing")
                continue
            area_fraction = self._mask_area_fraction(detection)
            if area_fraction < self.min_area_fraction:
                rejected.append(f"{identity}: segmented area is too small")
                continue
            if area_fraction > self.max_area_fraction:
                rejected.append(f"{identity}: segmented area is too large")
                continue
            eligible.append(detection)

        accepted: list[Detection] = []
        for detection in sorted(eligible, key=_priority):
            if any(
                detection.query.casefold() == kept.query.casefold()
                and _bbox_iou(detection.bbox_xyxy, kept.bbox_xyxy) >= self.duplicate_iou
                for kept in accepted
            ):
                rejected.append(
                    f"{detection.query}@{detection.bbox_xyxy}: duplicate overlap"
                )
                continue
            accepted.append(detection)
        important_by_query: dict[str, Detection] = {}
        for detection in accepted:
            if detection.required or detection.primary:
                important_by_query.setdefault(detection.query.casefold(), detection)
        pool = sorted(important_by_query.values(), key=_priority)
        pool_ids = {id(detection) for detection in pool}
        pool.extend(
            detection for detection in accepted if id(detection) not in pool_ids
        )
        if len(pool) > self.config.candidate_pool_size:
            for detection in pool[self.config.candidate_pool_size :]:
                rejected.append(
                    f"{detection.query}@{detection.bbox_xyxy}: candidate pool limit"
                )
            pool = pool[: self.config.candidate_pool_size]
        return pool, rejected

    def _crop_candidate(
        self, frame: np.ndarray, detection: Detection
    ) -> _CandidateCrop:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox_xyxy
        pad_x = (x2 - x1) * self.config.crop_padding
        pad_y = (y2 - y1) * self.config.crop_padding
        crop_xyxy = (
            max(0, int(np.floor(x1 - pad_x))),
            max(0, int(np.floor(y1 - pad_y))),
            min(width, int(np.ceil(x2 + pad_x))),
            min(height, int(np.ceil(y2 + pad_y))),
        )
        left, top, right, bottom = crop_xyxy
        if right <= left or bottom <= top:
            raise ValueError(f"{detection.query}: padded crop has no pixels")
        crop = frame[top:bottom, left:right]
        success, encoded = cv2.imencode(
            ".jpg",
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if not success:
            raise ValueError(f"{detection.query}: OpenCV failed to encode JPEG crop")
        jpeg = encoded.tobytes()
        digest = hashlib.sha256(jpeg).hexdigest()
        return _CandidateCrop(
            detection=detection,
            crop_xyxy=crop_xyxy,
            jpeg=jpeg,
            sha256=digest,
            mask_area_fraction=self._mask_area_fraction(detection),
        )

    def _prepare_pool(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
    ) -> tuple[list[_CandidateCrop], list[str]]:
        height, width = frame.shape[:2]
        accepted, rejected = self._filter_detections(
            detections, width=width, height=height
        )
        candidates: list[_CandidateCrop] = []
        seen_digests: set[str] = set()
        for detection in accepted:
            candidate = self._crop_candidate(frame, detection)
            if candidate.sha256 in seen_digests:
                rejected.append(
                    f"{detection.query}@{detection.bbox_xyxy}: duplicate crop bytes"
                )
                continue
            seen_digests.add(candidate.sha256)
            candidates.append(candidate)
        return candidates, rejected

    def _select(
        self, candidates: Sequence[_CandidateCrop], sample_id: str
    ) -> list[_CandidateCrop]:
        if len(candidates) < self.config.min_images:
            raise NoValidReferenceError(
                f"{sample_id}: only {len(candidates)} valid YOLOE crops; "
                f"at least {self.config.min_images} required"
            )
        rng = _selection_rng(self.config.selection_seed, sample_id)
        upper = min(self.config.max_images, len(candidates))
        target_count = rng.randint(self.config.min_images, upper)

        priority_by_query: dict[str, _CandidateCrop] = {}
        for candidate in candidates:
            detection = candidate.detection
            if detection.required or detection.primary:
                priority_by_query.setdefault(detection.query.casefold(), candidate)
        priority_candidates = list(priority_by_query.values())
        target_count = max(target_count, min(upper, len(priority_candidates)))

        # Membership is deliberately randomized, including when important queries
        # outnumber the 1--3 output slots.  Primary objects still occupy slots before
        # every other role, and the final stable sort only controls artifact order.
        primary = [
            candidate
            for candidate in priority_candidates
            if candidate.detection.primary
        ]
        if not primary:
            raise NoValidReferenceError(
                f"{sample_id}: no valid primary-object YOLOE crop"
            )
        required_other = [
            candidate
            for candidate in priority_candidates
            if not candidate.detection.primary and candidate.detection.required
        ]
        rng.shuffle(primary)
        rng.shuffle(required_other)
        selected = primary[:target_count]
        selected.extend(required_other[: target_count - len(selected)])
        selected_ids = {id(candidate) for candidate in selected}

        remaining = [
            candidate for candidate in candidates if id(candidate) not in selected_ids
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_count - len(selected)])
        return sorted(selected, key=lambda item: _priority(item.detection))

    def process_detections(
        self,
        *,
        sample_id: str,
        episode_index: int,
        frame0: np.ndarray | bytes,
        queries: Sequence[object],
        detections: Sequence[Detection],
        source_view: str = "camera_head",
    ) -> ReferenceBranchResult:
        """Build an auditable result from already-batched YOLOE detections."""

        if not sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not source_view.strip():
            raise ValueError("source_view must be non-empty")
        frame = _decode_frame(frame0)
        normalized_queries = normalize_queries(queries)
        candidates, rejected = self._prepare_pool(frame, detections)
        if not candidates:
            detail = "; ".join(rejected) if rejected else "YOLOE returned no detections"
            raise NoValidReferenceError(
                f"{sample_id}: no valid YOLOE first-frame Reference crop ({detail})"
            )
        selected = self._select(candidates, sample_id)

        artifacts: list[ReferenceArtifact] = []
        for ordinal, candidate in enumerate(selected):
            detection = candidate.detection
            relative_path = (
                Path(REFERENCE_DIRECTORY)
                / f"episode_{episode_index:06d}"
                / f"reference_{ordinal:02d}.jpg"
            )
            artifacts.append(
                ReferenceArtifact(
                    sample_id=sample_id,
                    reference_id=f"sha256:{candidate.sha256}",
                    relative_path=relative_path,
                    jpeg=candidate.jpeg,
                    source_view=source_view,
                    source_frame_index=0,
                    query=detection.query,
                    role=detection.role,
                    confidence=detection.confidence,
                    bbox_xyxy=detection.bbox_xyxy,
                    crop_xyxy=candidate.crop_xyxy,
                    sha256=candidate.sha256,
                    description=f"YOLOE-S Seg crop of {detection.query}",
                    provenance={
                        "backend": "yoloe",
                        "model": str(self.config.model_path),
                        "required": detection.required,
                        "primary": detection.primary,
                        "mask_area_fraction": candidate.mask_area_fraction,
                        "selection_seed": self.config.selection_seed,
                    },
                )
            )

        return ReferenceBranchResult(
            sample_id=sample_id,
            candidate_pool=tuple(candidate.detection for candidate in candidates),
            selected_artifacts=tuple(artifacts),
            rejected_reasons=tuple(rejected),
            input_fingerprint=self._fingerprint(frame, sample_id, normalized_queries),
        )

    def process(
        self,
        *,
        sample_id: str,
        episode_index: int,
        frame0: np.ndarray | bytes,
        queries: Sequence[object],
        source_view: str = "camera_head",
    ) -> ReferenceBranchResult:
        """Run YOLOE and post-process one Real first frame."""

        frame = _decode_frame(frame0)
        normalized_queries = normalize_queries(queries)
        detections = self.detector.predict([frame], normalized_queries)
        if len(detections) != 1:
            raise RuntimeError("Reference detector returned an invalid batch length")
        return self.process_detections(
            sample_id=sample_id,
            episode_index=episode_index,
            frame0=frame,
            queries=normalized_queries,
            detections=detections[0],
            source_view=source_view,
        )

    def process_batch(
        self, requests: Sequence[ReferenceBranchInput]
    ) -> list[ReferenceBranchResult]:
        """Batch YOLOE by query signature while preserving request order."""

        requests = list(requests)
        if not requests:
            return []
        frames = [_decode_frame(request.frame0) for request in requests]
        detection_requests = [
            DetectionRequest(frame, request.queries, request.sample_id)
            for frame, request in zip(frames, requests, strict=True)
        ]
        predict_requests = getattr(self.detector, "predict_requests", None)
        if predict_requests is not None:
            detections = predict_requests(detection_requests)
        else:
            detections = [
                self.detector.predict([frame], request.queries)[0]
                for frame, request in zip(frames, requests, strict=True)
            ]
        if len(detections) != len(requests):
            raise RuntimeError("Reference detector returned an invalid batch length")
        return [
            self.process_detections(
                sample_id=request.sample_id,
                episode_index=request.episode_index,
                frame0=frame,
                queries=request.queries,
                detections=batch,
                source_view=request.source_view,
            )
            for request, frame, batch in zip(requests, frames, detections, strict=True)
        ]
