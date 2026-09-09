"""Small, serializable contracts shared by the two preprocessing branches."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import MAX_REFERENCE_IMAGES, PROMPT_FRAME_COUNT

Split = Literal["train", "validation"]
ReferenceRole = Literal[
    "primary",
    "destination",
    "secondary",
    "robot",
    "workspace",
    "environment",
    "background",
]


def clean_text(value: str) -> str:
    """Collapse transport noise and whitespace without rewriting semantics."""

    value = value.replace("\x00", " ").strip()
    return re.sub(r"\s+", " ", value)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ReferenceQuery(StrictModel):
    """One short English detector prompt returned by the prompt VLM call."""

    query: str = Field(min_length=1, max_length=120)
    role: ReferenceRole
    required: bool = False

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = clean_text(value).strip(" .;:,!")
        if not value:
            raise ValueError("reference query must be non-empty")
        return value

    @property
    def primary(self) -> bool:
        return self.role == "primary"


# A migration-friendly synonym used in design documents and detector code.
ObjectQuery = ReferenceQuery


class PromptPayload(StrictModel):
    """The complete useful content of one VLM response."""

    prompt: str = Field(min_length=1, max_length=800)
    reference_queries: list[ReferenceQuery] = Field(min_length=1, max_length=8)

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("prompt must be non-empty")
        return value

    @field_validator("reference_queries")
    @classmethod
    def unique_queries(cls, values: list[ReferenceQuery]) -> list[ReferenceQuery]:
        seen: set[tuple[str, ReferenceRole]] = set()
        result: list[ReferenceQuery] = []
        for value in values:
            key = (value.query.casefold(), value.role)
            if key not in seen:
                seen.add(key)
                result.append(value)
        if not result:
            raise ValueError("at least one unique reference query is required")
        return result


class RealFrame(StrictModel):
    """One resized JPEG used in the ordered VLM video payload."""

    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    jpeg: bytes = Field(min_length=1)


class RealFrameBundle(StrictModel):
    """One-pass decode output shared by Prompt and Reference workers."""

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
        if (
            not isinstance(value, np.ndarray)
            or value.ndim != 3
            or value.shape[2] != 3
            or value.size == 0
            or value.dtype != np.uint8
        ):
            raise ValueError("first_frame_bgr must be a non-empty uint8 HxWx3 array")
        return value

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
            "subtasks": list(self.subtasks),
        }


class PromptResult(StrictModel):
    sample_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=800)
    reference_queries: tuple[ReferenceQuery, ...] = Field(min_length=1, max_length=8)
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


class Detection(StrictModel):
    """Serializable YOLOE result in full-resolution first-frame coordinates."""

    query: str = Field(min_length=1, max_length=120)
    role: ReferenceRole
    required: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    mask_polygon: tuple[tuple[float, float], ...] | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = clean_text(value).strip(" .;:,!")
        if not value:
            raise ValueError("detection query must be non-empty")
        return value

    @model_validator(mode="after")
    def valid_geometry(self) -> Detection:
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive width and height")
        if x1 < 0 or y1 < 0 or x2 > self.image_width or y2 > self.image_height:
            raise ValueError("bbox_xyxy must lie inside the source image")
        if self.mask_polygon is not None:
            if len(self.mask_polygon) < 3:
                raise ValueError("mask_polygon must contain at least three points")
            if any(
                x < 0 or y < 0 or x > self.image_width or y > self.image_height
                for x, y in self.mask_polygon
            ):
                raise ValueError("mask_polygon must lie inside the source image")
        return self

    @property
    def primary(self) -> bool:
        return self.role == "primary"


class ReferenceArtifact(StrictModel):
    """One selected JPEG plus all fields needed for schema-v2 publication."""

    sample_id: str = Field(min_length=1)
    reference_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    relative_path: Path
    jpeg: bytes = Field(min_length=1)
    source_view: str = Field(min_length=1)
    source_frame_index: Literal[0] = 0
    query: str = Field(min_length=1, max_length=120)
    role: ReferenceRole
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]
    crop_xyxy: tuple[int, int, int, int]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    description: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id", "source_view", "query")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        value = clean_text(value).strip(" .;:,!")
        if not value:
            raise ValueError("artifact text must be non-empty")
        return value

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return clean_text(value) or None

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("relative_path must stay inside the dataset root")
        return value

    @model_validator(mode="after")
    def valid_artifact(self) -> ReferenceArtifact:
        digest = hashlib.sha256(self.jpeg).hexdigest()
        if self.sha256 != digest or self.reference_id != f"sha256:{digest}":
            raise ValueError("reference_id and sha256 must identify the JPEG bytes")
        x1, y1, x2, y2 = self.crop_xyxy
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("crop_xyxy must have valid non-negative pixel bounds")
        return self

    @property
    def primary(self) -> bool:
        return self.role == "primary"

    @property
    def row(self) -> dict[str, Any]:
        scope = (
            self.role
            if self.role in {"robot", "workspace", "environment", "background"}
            else "objects"
        )
        row: dict[str, Any] = {
            "reference_id": self.reference_id,
            "reference_path": self.relative_path.as_posix(),
            "source_view": self.source_view,
            "source_frame_index": self.source_frame_index,
            "query": self.query,
            "label": self.query,
            "role": self.role,
            "scope": scope,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "crop_xyxy": list(self.crop_xyxy),
            "sha256": self.sha256,
        }
        if self.description is not None:
            row["description"] = self.description
        if self.provenance:
            row["provenance"] = self.provenance
        return row


class ReferenceBranchResult(StrictModel):
    sample_id: str = Field(min_length=1)
    candidate_pool: tuple[Detection, ...]
    selected_artifacts: tuple[ReferenceArtifact, ...] = Field(
        min_length=1, max_length=MAX_REFERENCE_IMAGES
    )
    rejected_reasons: tuple[str, ...] = ()
    input_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_sample(self) -> ReferenceBranchResult:
        if any(
            artifact.sample_id != self.sample_id for artifact in self.selected_artifacts
        ):
            raise ValueError("every selected artifact must belong to sample_id")
        return self


class EpisodePromptRow(StrictModel):
    """Stable Transfer training contract for ``episodes_prompt.jsonl``."""

    episode_index: int = Field(ge=0)
    prompt: str = Field(min_length=1)
    reference_ids: list[str] = Field(min_length=1, max_length=MAX_REFERENCE_IMAGES)

    @field_validator("reference_ids")
    @classmethod
    def unique_reference_ids(cls, values: list[str]) -> list[str]:
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("reference_ids must be non-empty and unique")
        return values


class EpisodeReferenceRow(StrictModel):
    """Stable schema-v2 contract for ``reference_images.jsonl``."""

    schema_version: Literal[2] = 2
    episode_index: int = Field(ge=0)
    references: list[dict[str, Any]] = Field(
        min_length=1, max_length=MAX_REFERENCE_IMAGES
    )
