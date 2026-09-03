"""Local checks and deterministic cleanup around the VLM critic pass."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .config import AnnotationConfig
from .models import (
    TEXT_FIELDS,
    AnnotationField,
    AppearanceEntry,
    AttributionIssue,
    DescriptionIssue,
    FieldBoundaryIssue,
    RecommendedEdit,
    RedundancyIssue,
    StructuredAnnotation,
    ValidationResult,
    clean_text,
)

APPEARANCE_TERMS = {
    "black",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "grey",
    "matte",
    "glossy",
    "shiny",
    "transparent",
    "translucent",
    "plastic",
    "metal",
    "metallic",
    "ceramic",
    "wood",
    "wooden",
    "fabric",
    "woven",
    "scratched",
    "worn",
}
IMAGING_PHRASES = {
    "motion blur",
    "sensor noise",
    "white balance",
    "exposure",
    "lens distortion",
    "depth of field",
    "compression artifacts",
}
ENVIRONMENT_PHRASES = {
    "robotics laboratory",
    "robotics lab",
    "industrial workcell",
    "warehouse",
    "home kitchen",
    "office-like workspace",
    "background environment",
    "laboratory background",
}
EMPTY_ADJECTIVES = {
    "beautiful",
    "highly realistic",
    "high-quality",
    "high quality",
    "detailed",
    "cinematic",
    "professional",
    "stunning",
}


def _sentences(text: str) -> list[str]:
    return [
        clean_text(item)
        for item in re.split(r"(?<=[.!?;])\s+", text)
        if clean_text(item)
    ]


def _semantic_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_local_issues(
    annotation: StructuredAnnotation,
    metadata: dict[str, Any],
    *,
    confidence_threshold: float = 0.55,
) -> ValidationResult:
    result = ValidationResult()
    objects = annotation.objects.text.lower() if annotation.objects else ""
    object_terms = sorted(
        term
        for term in APPEARANCE_TERMS
        if re.search(rf"\b{re.escape(term)}\b", objects)
    )
    if object_terms:
        result.field_boundary_violations.append(
            FieldBoundaryIssue(
                field="objects",
                belongs_in="appearance",
                text=", ".join(object_terms),
                reason=(
                    "Object identity/state is mixed with color, material, or "
                    "surface attributes."
                ),
                severity="warning",
            )
        )

    camera = annotation.camera.text.lower() if annotation.camera else ""
    camera_terms = sorted(term for term in IMAGING_PHRASES if term in camera)
    if camera_terms:
        result.field_boundary_violations.append(
            FieldBoundaryIssue(
                field="camera",
                belongs_in="imaging",
                text=", ".join(camera_terms),
                reason=(
                    "Image-formation properties do not belong in camera/view "
                    "configuration."
                ),
                severity="warning",
            )
        )

    scene = annotation.scene.text.lower() if annotation.scene else ""
    scene_terms = sorted(term for term in ENVIRONMENT_PHRASES if term in scene)
    if scene_terms:
        result.field_boundary_violations.append(
            FieldBoundaryIssue(
                field="scene",
                belongs_in="environment",
                text=", ".join(scene_terms),
                reason=(
                    "Detailed target-domain background/environment is mixed "
                    "into Scene."
                ),
                severity="warning",
            )
        )

    seen: dict[str, tuple[str, str]] = {}
    for field_name in TEXT_FIELDS:
        field = getattr(annotation, field_name)
        if field is None:
            continue
        if field.confidence < confidence_threshold:
            result.low_confidence_fields.append(field_name)
        for sentence in _sentences(field.text):
            key = _semantic_key(sentence)
            if key in seen and seen[key][0] != field_name:
                previous_field, previous_text = seen[key]
                result.redundant_information.append(
                    RedundancyIssue(
                        fields=[previous_field, field_name],
                        repeated_information=previous_text,
                        reason="The same sentence appears in multiple fields.",
                    )
                )
            else:
                seen[key] = (field_name, sentence)
        lowered = field.text.lower()
        empty_terms = sorted(term for term in EMPTY_ADJECTIVES if term in lowered)
        if empty_terms:
            result.overdescription.append(
                DescriptionIssue(
                    field=field_name,
                    text=", ".join(empty_terms),
                    reason="Decorative adjectives add no controllable information.",
                )
            )
        if len(field.text.split()) > 70:
            result.overdescription.append(
                DescriptionIssue(
                    field=field_name,
                    text=field.text,
                    reason=(
                        "Field is excessively long for a low-redundancy training "
                        "condition."
                    ),
                )
            )

    for index, appearance in enumerate(annotation.appearance):
        label = f"appearance[{index}]"
        if appearance.confidence < confidence_threshold:
            result.low_confidence_fields.append(label)
        if appearance.visible_in_reference and not any(
            evidence.startswith("reference:") or evidence == "reference_image"
            for evidence in appearance.evidence
        ):
            result.field_boundary_violations.append(
                FieldBoundaryIssue(
                    field=label,
                    belongs_in="appearance",
                    text=appearance.entity,
                    reason=(
                        "visible_in_reference=true lacks reference-image evidence "
                        "and must be visually checked."
                    ),
                    severity="error",
                )
            )

    authoritative = {
        "robot": metadata.get("robot_type"),
        "camera": metadata.get("camera_views"),
        "task": metadata.get("task"),
    }
    canonical_fields = metadata.get("canonical_fields") or {}
    for field_name, value in authoritative.items():
        field = getattr(annotation, field_name)
        if (
            value
            and field is not None
            and field.source != "metadata"
            and field_name not in canonical_fields
        ):
            result.attribution_errors.append(
                AttributionIssue(
                    field=field_name,
                    declared_source=field.source,
                    expected_source="metadata",
                    reason=(
                        f"Reliable {field_name} metadata is available and has "
                        "priority."
                    ),
                    severity="error",
                )
            )
    result.low_confidence_fields = list(dict.fromkeys(result.low_confidence_fields))
    return result


def merge_validation(
    primary: ValidationResult, local: ValidationResult
) -> ValidationResult:
    merged = primary.model_copy(deep=True)
    list_fields = (
        "unsupported_claims",
        "redundant_information",
        "field_boundary_violations",
        "important_omissions",
        "attribution_errors",
        "overdescription",
        "recommended_edits",
    )
    for field_name in list_fields:
        values = [*getattr(merged, field_name), *getattr(local, field_name)]
        unique: dict[str, Any] = {}
        for value in values:
            unique[value.model_dump_json()] = value
        setattr(merged, field_name, list(unique.values()))
    merged.low_confidence_fields = list(
        dict.fromkeys([*merged.low_confidence_fields, *local.low_confidence_fields])
    )
    merged.retry_recommended = merged.retry_recommended or local.retry_recommended
    return merged


def normalize_annotation_evidence(
    annotation: StructuredAnnotation,
) -> StructuredAnnotation:
    """Fill the canonical generic marker implied by reference-sourced appearance.

    ``source=reference`` plus ``visible_in_reference=true`` already declares the
    observation source. Some providers omit only the redundant evidence string;
    normalize that bookkeeping omission before validation instead of spending a
    full annotation retry on it.
    """

    normalized = annotation.model_copy(deep=True)
    for item in normalized.appearance:
        has_reference_evidence = any(
            evidence.startswith("reference:") or evidence == "reference_image"
            for evidence in item.evidence
        )
        if (
            item.source == "reference"
            and item.visible_in_reference
            and not has_reference_evidence
        ):
            item.evidence = [*item.evidence, "reference_image"]
    return StructuredAnnotation.model_validate(normalized.model_dump())


def _append_field_text(
    field: AnnotationField | None, text: str, source: str
) -> AnnotationField:
    if field is None:
        return AnnotationField(text=text, source=source, confidence=0.8, evidence=[])
    return field.model_copy(update={"text": clean_text(f"{field.text} {text}")})


def _apply_edit(annotation: StructuredAnnotation, edit: RecommendedEdit) -> None:
    if not edit.automatic_safe:
        return
    if edit.operation == "remove_field" and edit.field in TEXT_FIELDS:
        setattr(annotation, edit.field, None)
    elif (
        edit.operation == "replace_text"
        and edit.field in TEXT_FIELDS
        and edit.replacement
    ):
        field = getattr(annotation, edit.field)
        if field is not None and (
            edit.original_text is None or field.text == edit.original_text
        ):
            field.text = clean_text(edit.replacement)
    elif (
        edit.operation == "replace_text"
        and edit.field == "appearance"
        and edit.entity
        and edit.original_text
        and edit.replacement
    ):
        for item in annotation.appearance:
            if item.entity.lower() != edit.entity.lower():
                continue
            item.attributes = [
                clean_text(attribute.replace(edit.original_text, edit.replacement))
                if edit.original_text in attribute
                else attribute
                for attribute in item.attributes
            ]
    elif (
        edit.operation == "move_text"
        and edit.field in TEXT_FIELDS
        and edit.target_field in TEXT_FIELDS
    ):
        source_field = getattr(annotation, edit.field)
        if (
            source_field is None
            or not edit.original_text
            or edit.original_text not in source_field.text
        ):
            return
        target = getattr(annotation, edit.target_field)
        setattr(
            annotation,
            edit.target_field,
            _append_field_text(target, edit.original_text, source_field.source),
        )
        remaining = clean_text(source_field.text.replace(edit.original_text, " "))
        setattr(
            annotation,
            edit.field,
            source_field.model_copy(update={"text": remaining}) if remaining else None,
        )
    elif edit.operation == "remove_appearance" and edit.entity:
        annotation.appearance = [
            item
            for item in annotation.appearance
            if item.entity.lower() != edit.entity.lower()
        ]
    elif (
        edit.operation == "set_visibility"
        and edit.entity
        and edit.visible_in_reference is not None
    ):
        for item in annotation.appearance:
            if item.entity.lower() == edit.entity.lower():
                item.visible_in_reference = edit.visible_in_reference


def apply_automatic_safe_edits(
    annotation: StructuredAnnotation,
    validation: ValidationResult,
) -> StructuredAnnotation:
    clean = annotation.model_copy(deep=True)
    for edit in validation.recommended_edits:
        _apply_edit(clean, edit)
    return StructuredAnnotation.model_validate(clean.model_dump())


def _deduplicate_field_text(text: str) -> str:
    output: list[str] = []
    keys: set[str] = set()
    for sentence in _sentences(clean_text(text)):
        key = _semantic_key(sentence)
        if key and key not in keys:
            output.append(sentence)
            keys.add(key)
    return " ".join(output)


def canonicalize_annotation(
    annotation: StructuredAnnotation,
    validation: ValidationResult,
    config: AnnotationConfig,
    metadata: dict[str, Any],
    *,
    apply_safe_edits: bool,
) -> StructuredAnnotation:
    clean = annotation.model_copy(deep=True)
    if apply_safe_edits:
        clean = apply_automatic_safe_edits(clean, validation)

    canonical_fields = metadata.get("canonical_fields") or {}
    if isinstance(canonical_fields, dict):
        for field_name, value in canonical_fields.items():
            if field_name not in TEXT_FIELDS or value is None:
                continue
            if isinstance(value, str):
                text = value
                evidence = [f"metadata:canonical_fields.{field_name}"]
            elif isinstance(value, dict) and value.get("text"):
                text = str(value["text"])
                evidence = list(
                    value.get("evidence") or [f"metadata:canonical_fields.{field_name}"]
                )
            else:
                continue
            setattr(
                clean,
                field_name,
                AnnotationField(
                    text=text, source="metadata", confidence=1.0, evidence=evidence
                ),
            )

    for field_name in TEXT_FIELDS:
        field = getattr(clean, field_name)
        if field is None:
            continue
        if (
            config.drop_low_confidence_fields
            and field.source != "metadata"
            and field.confidence < config.confidence_threshold
        ):
            setattr(clean, field_name, None)
            continue
        normalized = _deduplicate_field_text(field.text)
        setattr(clean, field_name, field.model_copy(update={"text": normalized}))

    appearances: list[AppearanceEntry] = []
    seen_attributes: defaultdict[str, set[str]] = defaultdict(set)
    for item in clean.appearance:
        if (
            config.drop_low_confidence_fields
            and item.source != "metadata"
            and item.confidence < config.confidence_threshold
        ):
            continue
        entity_key = _semantic_key(item.entity)
        attributes = [
            attribute
            for attribute in item.attributes
            if _semantic_key(attribute) not in seen_attributes[entity_key]
        ]
        if not attributes:
            continue
        seen_attributes[entity_key].update(_semantic_key(value) for value in attributes)
        appearances.append(item.model_copy(update={"attributes": attributes}))
    clean.appearance = appearances
    return StructuredAnnotation.model_validate(clean.model_dump())
