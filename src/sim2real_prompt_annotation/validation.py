"""Local validation for the project-specific conditioning contract."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .models import (
    EvidenceText,
    StructuredAnnotation,
    ValidationIssue,
    ValidationResult,
    clean_text,
)
from .renderer import PromptLengthError, PromptRenderer

EMPTY_PHRASES = {
    "beautiful",
    "cinematic",
    "professional",
    "high-quality",
    "high quality",
    "highly detailed",
    "stunning",
    "masterpiece",
}
TRAJECTORY_PHRASES = {
    "frame-by-frame",
    "frame by frame",
    "precise trajectory",
    "exact trajectory",
    "action phase",
    "action phases",
    "followed by",
}
SOURCE_PREFIXES = {
    "metadata": ("metadata:", "meta/"),
    "sim": ("sim:",),
    "real": ("real:",),
    "reference": ("reference:",),
    "pair": ("sim:", "real:", "reference:", "metadata:", "meta/"),
    "inference": (),
}


def _issue(
    category: str,
    field: str,
    claim: str,
    reason: str,
    *,
    severity: str = "warning",
) -> ValidationIssue:
    return ValidationIssue(
        category=category,  # type: ignore[arg-type]
        field=field,
        claim=claim,
        reason=reason,
        severity=severity,  # type: ignore[arg-type]
    )


def _evidence_fields(
    annotation: StructuredAnnotation,
) -> Iterable[tuple[str, EvidenceText]]:
    yield "task.summary", annotation.task.summary
    yield "task.robot", annotation.task.robot
    for name in (
        "robot_motion",
        "object_motion",
        "contacts_and_state_changes",
        "spatial_relations",
        "camera_and_timing",
    ):
        value = getattr(annotation.sim_invariants, name)
        if value is not None:
            yield f"sim_invariants.{name}", value
    for name in ("robot_appearance", "workspace", "background", "lighting"):
        value = getattr(annotation.target_visuals, name)
        if value is not None:
            yield f"target_visuals.{name}", value


def _semantic_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_local_issues(
    annotation: StructuredAnnotation,
    metadata: dict[str, Any],
    *,
    confidence_threshold: float,
    renderer: PromptRenderer,
) -> ValidationResult:
    issues: list[ValidationIssue] = []

    if metadata.get("task") and annotation.task.summary.source != "metadata":
        issues.append(
            _issue(
                "metadata_conflict",
                "task.summary",
                annotation.task.summary.text,
                "Task metadata is available and must be the authoritative source.",
                severity="error",
            )
        )
    if metadata.get("robot_type") and annotation.task.robot.source != "metadata":
        issues.append(
            _issue(
                "metadata_conflict",
                "task.robot",
                annotation.task.robot.text,
                "Robot metadata is available and must be the authoritative source.",
                severity="error",
            )
        )

    for path, value in _evidence_fields(annotation):
        if value.confidence < confidence_threshold:
            issues.append(
                _issue(
                    "unsupported_claim",
                    path,
                    value.text,
                    f"Confidence {value.confidence:.2f} is below "
                    f"{confidence_threshold:.2f}.",
                )
            )
        prefixes = SOURCE_PREFIXES[value.source]
        if value.source != "inference" and not value.evidence:
            issues.append(
                _issue(
                    "unsupported_claim",
                    path,
                    value.text,
                    "A directly sourced field must include evidence IDs.",
                    severity="error",
                )
            )
        elif (
            prefixes
            and value.evidence
            and not any(evidence.startswith(prefixes) for evidence in value.evidence)
        ):
            issues.append(
                _issue(
                    "unsupported_claim",
                    path,
                    value.text,
                    f"Evidence IDs do not support declared source={value.source}.",
                    severity="error",
                )
            )

    for index, obj in enumerate(annotation.task.objects):
        path = f"task.objects[{index}]"
        if obj.confidence < confidence_threshold:
            issues.append(
                _issue(
                    "unsupported_claim",
                    path,
                    obj.identity,
                    f"Confidence {obj.confidence:.2f} is below "
                    f"{confidence_threshold:.2f}.",
                )
            )
        if obj.source != "inference" and not obj.evidence:
            issues.append(
                _issue(
                    "unsupported_claim",
                    path,
                    obj.identity,
                    "A directly sourced task object must include evidence IDs.",
                    severity="error",
                )
            )

    allowed_sim_sources = {"sim", "pair", "metadata"}
    for name in (
        "robot_motion",
        "object_motion",
        "contacts_and_state_changes",
        "spatial_relations",
        "camera_and_timing",
    ):
        value = getattr(annotation.sim_invariants, name)
        if value is not None and value.source not in allowed_sim_sources:
            issues.append(
                _issue(
                    "sim_invariant_error",
                    f"sim_invariants.{name}",
                    value.text,
                    "Dynamic invariants must be derived from Sim, paired evidence, "
                    "or authoritative metadata.",
                    severity="error",
                )
            )

    for name in ("robot_appearance", "workspace", "background", "lighting"):
        value = getattr(annotation.target_visuals, name)
        if value is not None and value.source not in {"real", "pair"}:
            issues.append(
                _issue(
                    "target_visual_error",
                    f"target_visuals.{name}",
                    value.text,
                    "Target appearance must come from the Real video or paired "
                    "evidence.",
                    severity="error",
                )
            )

    reference_scopes = annotation.reference.use_for
    plan_scopes = annotation.prompt_plan.reference_scopes
    if reference_scopes != plan_scopes:
        issues.append(
            _issue(
                "reference_scope_error",
                "prompt_plan.reference_scopes",
                str(plan_scopes),
                "Prompt reference scopes must exactly match reference.use_for, "
                "including order.",
                severity="error",
            )
        )

    plan_texts = [
        annotation.prompt_plan.task_clause,
        *annotation.prompt_plan.setting_clauses,
    ]
    lowered_plan = " ".join(plan_texts).lower()
    for phrase in sorted(EMPTY_PHRASES):
        if phrase in lowered_plan:
            issues.append(
                _issue(
                    "overdescription",
                    "prompt_plan",
                    phrase,
                    "Decorative quality language adds no controllable information.",
                )
            )
    for phrase in sorted(TRAJECTORY_PHRASES):
        if phrase in lowered_plan:
            issues.append(
                _issue(
                    "prompt_content_error",
                    "prompt_plan",
                    phrase,
                    "Trajectory and action-phase detail belongs to Sim, not Prompt.",
                    severity="error",
                )
            )

    if len(annotation.prompt_plan.task_clause.split()) > 30:
        issues.append(
            _issue(
                "overdescription",
                "prompt_plan.task_clause",
                annotation.prompt_plan.task_clause,
                "The task clause must stay below 30 whitespace-delimited words.",
                severity="error",
            )
        )
    for index, clause in enumerate(annotation.prompt_plan.setting_clauses):
        if len(clause.split()) > 14:
            issues.append(
                _issue(
                    "overdescription",
                    f"prompt_plan.setting_clauses[{index}]",
                    clause,
                    "A setting clause must stay below 14 words.",
                    severity="error",
                )
            )

    seen: dict[str, str] = {}
    for path, text in [
        ("prompt_plan.task_clause", annotation.prompt_plan.task_clause),
        *[
            (f"prompt_plan.setting_clauses[{index}]", clause)
            for index, clause in enumerate(annotation.prompt_plan.setting_clauses)
        ],
    ]:
        key = _semantic_key(text)
        if key in seen:
            issues.append(
                _issue(
                    "redundancy",
                    path,
                    text,
                    f"Duplicates {seen[key]}.",
                )
            )
        else:
            seen[key] = path

    try:
        renderer.render(annotation)
    except PromptLengthError as error:
        issues.append(
            _issue(
                "overdescription",
                "prompt_plan",
                "rendered prompt",
                str(error),
                severity="error",
            )
        )

    return ValidationResult(issues=issues)


def merge_validation(
    primary: ValidationResult, local: ValidationResult
) -> ValidationResult:
    unique: dict[str, ValidationIssue] = {}
    for issue in [*primary.issues, *local.issues]:
        unique[issue.model_dump_json()] = issue
    return ValidationResult(
        issues=list(unique.values()),
        retry_recommended=(primary.retry_recommended or local.retry_recommended),
    )


def canonicalize_annotation(
    annotation: StructuredAnnotation,
) -> StructuredAnnotation:
    """Return a normalized deep copy after all semantic checks have passed."""

    payload = annotation.model_dump()
    payload["sample_id"] = clean_text(payload["sample_id"])
    return StructuredAnnotation.model_validate(payload)
