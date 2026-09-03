"""Strict canonical schemas shared by annotation, validation, and rendering."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Source = Literal["metadata", "sim", "real", "reference", "pair", "inference"]
Severity = Literal["warning", "error"]
TEXT_FIELDS = (
    "robot",
    "camera",
    "task",
    "actions",
    "scene",
    "objects",
    "environment",
    "lighting",
    "imaging",
    "preserve",
)


def clean_text(value: str) -> str:
    """Normalize whitespace without changing semantic punctuation."""

    value = value.replace("\x00", " ").strip()
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", " ", value)
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AnnotationField(StrictModel):
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
        return list(
            dict.fromkeys(clean_text(value) for value in values if clean_text(value))
        )


class AppearanceEntry(StrictModel):
    entity: str = Field(min_length=1)
    attributes: list[str] = Field(min_length=1)
    source: Source
    confidence: float = Field(ge=0.0, le=1.0)
    visible_in_reference: bool
    evidence: list[str] = Field(default_factory=list)

    @field_validator("entity")
    @classmethod
    def normalize_entity(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("entity must contain non-whitespace content")
        return value

    @field_validator("attributes")
    @classmethod
    def normalize_attributes(cls, values: list[str]) -> list[str]:
        result = list(
            dict.fromkeys(clean_text(value) for value in values if clean_text(value))
        )
        if not result:
            raise ValueError("list must contain at least one non-empty value")
        return result

    @field_validator("evidence")
    @classmethod
    def normalize_evidence(cls, values: list[str]) -> list[str]:
        return list(
            dict.fromkeys(clean_text(value) for value in values if clean_text(value))
        )


class StructuredAnnotation(StrictModel):
    sample_id: str = Field(min_length=1)
    robot: AnnotationField | None = None
    camera: AnnotationField | None = None
    task: AnnotationField | None = None
    actions: AnnotationField | None = None
    scene: AnnotationField | None = None
    objects: AnnotationField | None = None
    environment: AnnotationField | None = None
    appearance: list[AppearanceEntry] = Field(default_factory=list)
    lighting: AnnotationField | None = None
    imaging: AnnotationField | None = None
    preserve: AnnotationField | None = None


class ClaimIssue(StrictModel):
    field: str
    claim: str
    reason: str
    severity: Severity = "warning"


class RedundancyIssue(StrictModel):
    fields: list[str] = Field(min_length=2)
    repeated_information: str
    reason: str
    severity: Severity = "warning"


class FieldBoundaryIssue(StrictModel):
    field: str
    belongs_in: str
    text: str
    reason: str
    severity: Severity = "warning"


class OmissionIssue(StrictModel):
    field: str
    missing_information: str
    evidence: list[str] = Field(default_factory=list)
    severity: Severity = "warning"


class AttributionIssue(StrictModel):
    field: str
    declared_source: Source
    expected_source: Source
    reason: str
    severity: Severity = "warning"


class DescriptionIssue(StrictModel):
    field: str
    text: str
    reason: str
    severity: Severity = "warning"


class RecommendedEdit(StrictModel):
    operation: Literal[
        "remove_field",
        "replace_text",
        "move_text",
        "remove_appearance",
        "set_visibility",
    ]
    field: str
    target_field: str | None = None
    entity: str | None = None
    original_text: str | None = None
    replacement: str | None = None
    visible_in_reference: bool | None = None
    reason: str
    automatic_safe: bool = False


class ValidationResult(StrictModel):
    unsupported_claims: list[ClaimIssue] = Field(default_factory=list)
    redundant_information: list[RedundancyIssue] = Field(default_factory=list)
    field_boundary_violations: list[FieldBoundaryIssue] = Field(default_factory=list)
    important_omissions: list[OmissionIssue] = Field(default_factory=list)
    low_confidence_fields: list[str] = Field(default_factory=list)
    attribution_errors: list[AttributionIssue] = Field(default_factory=list)
    overdescription: list[DescriptionIssue] = Field(default_factory=list)
    recommended_edits: list[RecommendedEdit] = Field(default_factory=list)
    retry_recommended: bool = False

    def has_severe_errors(self) -> bool:
        issue_groups = (
            self.unsupported_claims,
            self.field_boundary_violations,
            self.important_omissions,
            self.attribution_errors,
        )
        return self.retry_recommended or any(
            issue.severity == "error" for group in issue_groups for issue in group
        )
