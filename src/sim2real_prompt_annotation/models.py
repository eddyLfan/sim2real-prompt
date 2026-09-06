"""Project-specific schemas for Sim-to-Real prompt annotation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Source = Literal["metadata", "sim", "real", "reference", "pair", "inference"]
Severity = Literal["warning", "error"]
ActiveArm = Literal["left", "right", "both", "unspecified"]
ReferenceScope = Literal["robot", "objects", "workspace", "background"]
REFERENCE_SCOPE_ORDER: tuple[ReferenceScope, ...] = (
    "robot",
    "objects",
    "workspace",
    "background",
)
IssueCategory = Literal[
    "unsupported_claim",
    "metadata_conflict",
    "sim_invariant_error",
    "target_visual_error",
    "reference_scope_error",
    "prompt_content_error",
    "redundancy",
    "overdescription",
]


def clean_text(value: str) -> str:
    """Normalize whitespace while preserving semantic punctuation."""

    value = value.replace("\x00", " ").strip()
    value = re.sub(r"[ \t]+", " ", value)
    return re.sub(r"\s*\n\s*", " ", value)


def _clean_unique(values: list[str]) -> list[str]:
    return list(
        dict.fromkeys(clean_text(value) for value in values if clean_text(value))
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceText(StrictModel):
    text: str = Field(min_length=1)
    source: Source
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("text must contain non-whitespace content")
        return value

    @field_validator("evidence")
    @classmethod
    def normalize_evidence(cls, values: list[str]) -> list[str]:
        return _clean_unique(values)


class TaskObject(StrictModel):
    """A task-relevant entity aligned by semantic role across Sim and Real."""

    role: str = Field(min_length=1, max_length=80)
    identity: str = Field(min_length=1, max_length=120)
    state: str | None = Field(default=None, max_length=160)
    appearance: str | None = Field(default=None, max_length=160)
    geometry_affordance: str | None = Field(default=None, max_length=160)
    source: Source
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("role", "identity", "state", "appearance", "geometry_affordance")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = clean_text(value)
        return value or None

    @field_validator("evidence")
    @classmethod
    def normalize_evidence(cls, values: list[str]) -> list[str]:
        return _clean_unique(values)


class MetadataGroundedText(StrictModel):
    """English slot text grounded by one verbatim task-metadata span."""

    text: str = Field(min_length=1, max_length=120)
    metadata_span: str = Field(min_length=1, max_length=240)

    @field_validator("text", "metadata_span")
    @classmethod
    def normalize_grounded_text(cls, value: str) -> str:
        value = clean_text(value).strip(" .;:,!")
        if not value:
            raise ValueError("grounded text must be non-empty")
        return value


class TaskSemantics(StrictModel):
    """Prompt-critical slots derived only from authoritative task metadata."""

    metadata_task: str = Field(min_length=1)
    robot: str = Field(min_length=1, max_length=100)
    active_arm: ActiveArm
    action: MetadataGroundedText
    primary_objects: list[MetadataGroundedText] = Field(min_length=1, max_length=4)
    goal: MetadataGroundedText | None = None
    constraints: list[MetadataGroundedText] = Field(default_factory=list, max_length=3)

    @field_validator("metadata_task", "robot")
    @classmethod
    def normalize_slot_text(cls, value: str) -> str:
        value = clean_text(value).strip(" .;:,!")
        if not value:
            raise ValueError("slot text must be non-empty")
        return value

    @field_validator("action")
    @classmethod
    def action_is_one_base_phrase(
        cls, value: MetadataGroundedText
    ) -> MetadataGroundedText:
        text = value.text.lower().removeprefix("to ").strip(" .;:,!")
        if re.search(r"\b(?:and|then|followed by)\b", text):
            raise ValueError("action must contain one metadata-derived action")
        return value.model_copy(update={"text": text})

    @field_validator("primary_objects")
    @classmethod
    def objects_are_atomic(
        cls, values: list[MetadataGroundedText]
    ) -> list[MetadataGroundedText]:
        for value in values:
            if re.search(r"\b(?:with|containing|holding)\b", value.text.lower()):
                raise ValueError(
                    "primary objects must be atomic metadata-derived noun phrases"
                )
        return values

    @field_validator("goal")
    @classmethod
    def goal_is_prepositional(
        cls, value: MetadataGroundedText | None
    ) -> MetadataGroundedText | None:
        if value is None:
            return None
        if not re.match(
            r"^(?:into|onto|on|in|to|beside|under|over|between|inside|outside|"
            r"at|from|toward|through|against)\b",
            value.text.lower(),
        ):
            raise ValueError("goal must be one metadata-derived prepositional phrase")
        if re.search(r"\b(?:and|then|while|followed by)\b", value.text.lower()):
            raise ValueError("goal cannot contain another action or phase")
        return value


class TaskDescription(StrictModel):
    summary: EvidenceText
    robot: EvidenceText
    semantics: TaskSemantics
    objects: list[TaskObject] = Field(min_length=1)


class SimInvariants(StrictModel):
    """Dynamic facts controlled by Sim/RobotState and excluded from the prompt."""

    robot_motion: EvidenceText
    object_motion: EvidenceText | None = None
    contacts_and_state_changes: EvidenceText | None = None
    spatial_relations: EvidenceText | None = None
    camera_and_timing: EvidenceText | None = None


class TargetVisuals(StrictModel):
    """Coarse target-domain appearance observed in the paired Real video."""

    robot_appearance: EvidenceText | None = None
    workspace: EvidenceText | None = None
    background: EvidenceText | None = None
    lighting: EvidenceText

    @model_validator(mode="after")
    def require_scene_context(self) -> TargetVisuals:
        if self.workspace is None and self.background is None:
            raise ValueError("target visuals require workspace or background context")
        return self


class ReferenceDescription(StrictModel):
    """Visibility and intended use of the exact sampled Reference frame."""

    view: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    visible_content: list[str]
    use_for: list[ReferenceScope]
    unclear_or_occluded: list[ReferenceScope]

    @field_validator("view")
    @classmethod
    def normalize_view(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("view must be non-empty")
        return value

    @field_validator("visible_content")
    @classmethod
    def normalize_visible_content(cls, values: list[str]) -> list[str]:
        return _clean_unique(values)

    @field_validator("use_for", "unclear_or_occluded")
    @classmethod
    def unique_scopes(cls, values: list[ReferenceScope]) -> list[ReferenceScope]:
        selected = set(values)
        return [scope for scope in REFERENCE_SCOPE_ORDER if scope in selected]

    @model_validator(mode="after")
    def scopes_do_not_overlap(self) -> ReferenceDescription:
        overlap = set(self.use_for) & set(self.unclear_or_occluded)
        if overlap:
            raise ValueError(
                f"reference scopes cannot be both usable and unclear: {sorted(overlap)}"
            )
        return self


class StructuredAnnotation(StrictModel):
    sample_id: str = Field(min_length=1)
    task: TaskDescription
    sim_invariants: SimInvariants
    target_visuals: TargetVisuals
    reference: ReferenceDescription


class ValidationIssue(StrictModel):
    category: IssueCategory
    field: str
    claim: str
    reason: str
    severity: Severity = "warning"


class ValidationResult(StrictModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    def has_severe_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)
