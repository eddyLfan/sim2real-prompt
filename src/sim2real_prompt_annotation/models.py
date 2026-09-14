"""Serializable contracts shared by the prompt and scene-Reference branches."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import PROMPT_FRAME_COUNT, REFERENCE_IMAGE_COUNT

Split = Literal["train", "validation"]


def clean_text(value: str) -> str:
    """Collapse transport noise and whitespace without rewriting semantics."""

    value = value.replace("\x00", " ").strip()
    return re.sub(r"\s+", " ", value)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PromptPayload(StrictModel):
    """The complete useful content of the single VLM response."""

    prompt: str = Field(min_length=1, max_length=800)

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("prompt must be non-empty")
        return value


class RealFrame(StrictModel):
    """One resized JPEG used in the ordered VLM video payload."""

    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    jpeg: bytes = Field(min_length=1)


def _valid_bgr(value: Any, *, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 3
        or value.shape[2] != 3
        or value.size == 0
        or value.dtype != np.uint8
    ):
        raise ValueError(f"{name} must be a non-empty uint8 HxWx3 array")
    return value


class RealFrameBundle(StrictModel):
    """One-open decode output: eight Prompt JPEGs and the original Real frame zero."""

    sample_id: str = Field(min_length=1)
    view: str = Field(min_length=1)
    video_path: Path
    frame_count: int = Field(ge=1)
    fps: float = Field(gt=0.0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    first_frame_bgr: Any
    prompt_frames: tuple[RealFrame, ...]

    @field_validator("sample_id", "view")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("identity must be non-empty")
        return value

    @field_validator("first_frame_bgr")
    @classmethod
    def valid_first_frame(cls, value: Any) -> np.ndarray:
        return _valid_bgr(value, name="first_frame_bgr")

    @model_validator(mode="after")
    def valid_prompt_sequence(self) -> RealFrameBundle:
        if len(self.prompt_frames) != PROMPT_FRAME_COUNT:
            raise ValueError(
                f"prompt_frames must contain exactly {PROMPT_FRAME_COUNT} frames"
            )
        indices = [frame.frame_index for frame in self.prompt_frames]
        if indices != sorted(set(indices)):
            raise ValueError("prompt frame indices must be unique and increasing")
        if indices[0] != 0:
            raise ValueError("prompt frame sequence must include Real frame zero")
        if indices[-1] >= self.frame_count:
            raise ValueError("prompt frame index exceeds decoded video length")
        if self.first_frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError("first_frame_bgr dimensions do not match bundle metadata")
        return self

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(frame.frame_index for frame in self.prompt_frames)


class EpisodeRecord(StrictModel):
    """Validated source identity and the sole Real video consumed downstream."""

    sample_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    dataset_root: Path
    episode_index: int = Field(ge=0)
    episode_length: int = Field(ge=1)
    split: Split
    fps: float = Field(gt=0.0)
    robot_type: str | None = None
    task: str = Field(min_length=1)
    subtasks: tuple[dict[str, Any], ...] = ()
    real_view: str = Field(min_length=1)
    real_frame_height: int = Field(ge=1)
    real_frame_width: int = Field(ge=1)
    real_video: Path
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "sample_id", "source_id", "dataset_name", "domain", "task", "real_view"
    )
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("required identity/task text must be non-empty")
        return value

    @field_validator("robot_type")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return clean_text(value) or None

    def annotation_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "dataset": self.dataset_name,
            "domain": self.domain,
            "episode_index": self.episode_index,
            "episode_length_frames": self.episode_length,
            "split": self.split,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "task": self.task,
            "real_view": self.real_view,
            "real_frame_height": self.real_frame_height,
            "real_frame_width": self.real_frame_width,
            "subtasks": list(self.subtasks),
        }


class PromptResult(StrictModel):
    sample_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=800)
    model: str = Field(min_length=1)
    request_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    frame_indices: tuple[int, ...]
    input_fingerprint: str = Field(min_length=1)

    @field_validator("sample_id", "prompt", "model", "input_fingerprint")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("result text must be non-empty")
        return value

    @field_validator("frame_indices")
    @classmethod
    def exact_ordered_frames(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) != PROMPT_FRAME_COUNT:
            raise ValueError(
                f"frame_indices must contain exactly {PROMPT_FRAME_COUNT} values"
            )
        if list(values) != sorted(set(values)) or values[0] != 0:
            raise ValueError("frame_indices must be unique, increasing, and start at 0")
        return values


class RobotMaskPrediction(StrictModel):
    """One full-resolution binary robot mask produced by a segmenter."""

    backend: str = Field(min_length=1)
    query: str = Field(default="robot", min_length=1, max_length=120)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_xyxy: tuple[int, int, int, int]
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    mask: Any

    @field_validator("backend", "query")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = clean_text(value).strip(" ,.;:")
        if not value:
            raise ValueError("robot mask text must be non-empty")
        return value

    @field_validator("mask")
    @classmethod
    def valid_binary_mask(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray) or value.ndim != 2 or value.size == 0:
            raise ValueError("robot mask must be a non-empty HxW numpy array")
        if value.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
            raise ValueError("robot mask must have bool or uint8 dtype")
        values = np.unique(value)
        if not set(int(item) for item in values).issubset({0, 1, 255}):
            raise ValueError("robot mask must be binary")
        return np.ascontiguousarray(value > 0, dtype=np.uint8)

    @model_validator(mode="after")
    def valid_geometry(self) -> RobotMaskPrediction:
        if self.mask.shape != (self.image_height, self.image_width):
            raise ValueError("robot mask dimensions do not match the source image")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 <= self.image_width and 0 <= y1 < y2 <= self.image_height):
            raise ValueError("robot mask bbox must lie inside the source image")
        if not np.any(self.mask):
            raise ValueError("robot mask must contain foreground pixels")
        ys, xs = np.nonzero(self.mask)
        actual = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if actual != self.bbox_xyxy:
            raise ValueError("robot mask bbox must be the tight mask boundary")
        return self

    @property
    def mask_sha256(self) -> str:
        return hashlib.sha256(self.mask.tobytes(order="C")).hexdigest()

    @property
    def mask_area_fraction(self) -> float:
        return float(np.count_nonzero(self.mask)) / float(self.mask.size)


class RobotMaskDiagnostic(StrictModel):
    """Compact, JSON-safe mask evidence retained in the branch checkpoint."""

    backend: str = Field(min_length=1)
    query: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_xyxy: tuple[int, int, int, int]
    mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mask_area_fraction: float = Field(gt=0.0, le=1.0)

    @classmethod
    def from_prediction(cls, value: RobotMaskPrediction) -> RobotMaskDiagnostic:
        return cls(
            backend=value.backend,
            query=value.query,
            confidence=value.confidence,
            bbox_xyxy=value.bbox_xyxy,
            mask_sha256=value.mask_sha256,
            mask_area_fraction=value.mask_area_fraction,
        )


class SceneReferenceArtifact(StrictModel):
    """The sole full-scene JPEG created by robot removal and inpainting."""

    sample_id: str = Field(min_length=1)
    reference_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    relative_path: Path
    jpeg: bytes = Field(min_length=1)
    source_view: str = Field(min_length=1)
    source_frame_index: Literal[0] = 0
    scope: Literal["environment"] = "environment"
    reference_kind: Literal["robot_removed_scene"] = "robot_removed_scene"
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    source_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mask_area_fraction: float = Field(gt=0.0, lt=1.0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id", "source_view")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("artifact text must be non-empty")
        return value

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("relative_path must stay inside the dataset root")
        return value

    @model_validator(mode="after")
    def valid_artifact(self) -> SceneReferenceArtifact:
        digest = hashlib.sha256(self.jpeg).hexdigest()
        if self.sha256 != digest or self.reference_id != f"sha256:{digest}":
            raise ValueError("reference_id and sha256 must identify the JPEG bytes")
        return self

    @property
    def row(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "reference_path": self.relative_path.as_posix(),
            "source_view": self.source_view,
            "source_frame_index": self.source_frame_index,
            "scope": self.scope,
            "reference_kind": self.reference_kind,
            "width": self.width,
            "height": self.height,
            "source_frame_sha256": self.source_frame_sha256,
            "mask_sha256": self.mask_sha256,
            "mask_area_fraction": self.mask_area_fraction,
            "sha256": self.sha256,
            "provenance": self.provenance,
        }


class SceneReferenceResult(StrictModel):
    sample_id: str = Field(min_length=1)
    artifact: SceneReferenceArtifact
    robot_masks: tuple[RobotMaskDiagnostic, ...] = Field(min_length=1)
    removal_mask_png: bytes = Field(min_length=1)
    rejected_reasons: tuple[str, ...] = ()
    input_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_sample(self) -> SceneReferenceResult:
        if self.artifact.sample_id != self.sample_id:
            raise ValueError("scene Reference artifact must belong to sample_id")
        decoded = cv2.imdecode(
            np.frombuffer(self.removal_mask_png, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if decoded is None or decoded.shape != (
            self.artifact.height,
            self.artifact.width,
        ):
            raise ValueError(
                "removal_mask_png must decode at the scene Reference dimensions"
            )
        values = set(int(item) for item in np.unique(decoded))
        if not values.issubset({0, 1, 255}) or not np.any(decoded):
            raise ValueError("removal_mask_png must contain a non-empty binary mask")
        binary = np.ascontiguousarray(decoded > 0, dtype=np.uint8)
        mask_digest = hashlib.sha256(binary.tobytes(order="C")).hexdigest()
        if mask_digest != self.artifact.mask_sha256:
            raise ValueError(
                "removal_mask_png pixels do not match artifact.mask_sha256"
            )
        area_fraction = float(np.count_nonzero(binary)) / float(binary.size)
        if not np.isclose(area_fraction, self.artifact.mask_area_fraction):
            raise ValueError(
                "removal_mask_png area does not match artifact.mask_area_fraction"
            )
        return self

    @property
    def selected_artifacts(self) -> tuple[SceneReferenceArtifact]:
        """Compatibility view for publication code during the schema-v3 migration."""

        return (self.artifact,)


# Compatibility aliases for one migration cycle. New code should use the scene names.
ReferenceArtifact = SceneReferenceArtifact
ReferenceBranchResult = SceneReferenceResult


class EpisodePromptRow(StrictModel):
    """Stable Transfer contract for ``episodes_prompt.jsonl``."""

    episode_index: int = Field(ge=0)
    prompt: str = Field(min_length=1)
    reference_ids: list[str] = Field(
        min_length=REFERENCE_IMAGE_COUNT, max_length=REFERENCE_IMAGE_COUNT
    )

    @field_validator("reference_ids")
    @classmethod
    def unique_reference_ids(cls, values: list[str]) -> list[str]:
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("reference_ids must be non-empty and unique")
        return values


class EpisodeReferenceRow(StrictModel):
    """Schema-v3 exact-one scene contract for ``reference_images.jsonl``."""

    schema_version: Literal[3] = 3
    episode_index: int = Field(ge=0)
    references: list[dict[str, Any]] = Field(
        min_length=REFERENCE_IMAGE_COUNT, max_length=REFERENCE_IMAGE_COUNT
    )
