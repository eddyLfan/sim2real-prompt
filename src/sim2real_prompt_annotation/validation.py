"""Minimal correctness checks for the final training Prompt."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .models import (
    StructuredAnnotation,
    ValidationIssue,
    ValidationResult,
    clean_text,
)
from .renderer import PromptLengthError, PromptRenderer
from .task_metadata import canonical_robot_description, task_contract, task_payload

TRAJECTORY_PHRASES = {
    "frame-by-frame",
    "frame by frame",
    "precise trajectory",
    "exact trajectory",
    "action phase",
    "action phases",
    "followed by",
}


def _error(category: str, field: str, claim: str, reason: str) -> ValidationIssue:
    return ValidationIssue(
        category=category,  # type: ignore[arg-type]
        field=field,
        claim=claim,
        reason=reason,
        severity="error",
    )


def _metadata_span_supported(task: str, span: str) -> bool:
    """Accept exact spans and minor inflection differences, reject foreign concepts."""

    payload = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", task_payload(task).lower())
    needle = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", span.lower())
    if not needle:
        return False
    if needle in payload:
        return True
    match = SequenceMatcher(None, needle, payload).find_longest_match()
    return match.size / len(needle) >= 0.8


def find_local_issues(
    annotation: StructuredAnnotation,
    metadata: dict[str, Any],
    rendered_prompt: str,
    *,
    renderer: PromptRenderer,
) -> ValidationResult:
    """Reject only issues that make the final conditioning Prompt incorrect."""

    issues: list[ValidationIssue] = []
    semantics = annotation.task.semantics
    metadata_task = metadata.get("task")
    metadata_robot = metadata.get("robot_type")

    if metadata_task and semantics.metadata_task != metadata_task:
        issues.append(
            _error(
                "metadata_conflict",
                "task.semantics.metadata_task",
                semantics.metadata_task,
                "Prompt task does not match authoritative task metadata.",
            )
        )

    expected_robot = canonical_robot_description(metadata_robot, semantics.robot)
    if metadata_robot and semantics.robot != expected_robot:
        issues.append(
            _error(
                "metadata_conflict",
                "task.semantics.robot",
                semantics.robot,
                f"Expected canonical robot description: {expected_robot}.",
            )
        )

    contract = task_contract(metadata_task)
    if semantics.active_arm != contract.active_arm:
        issues.append(
            _error(
                "metadata_conflict",
                "task.semantics.active_arm",
                semantics.active_arm,
                f"Task metadata requires active_arm={contract.active_arm}.",
            )
        )

    if metadata_task:
        grounded = [
            semantics.action,
            *semantics.primary_objects,
            *semantics.constraints,
        ]
        if semantics.goal is not None:
            grounded.append(semantics.goal)
        unsupported = sorted(
            {
                value.metadata_span
                for value in grounded
                if not _metadata_span_supported(metadata_task, value.metadata_span)
            }
        )
        if unsupported:
            issues.append(
                _error(
                    "metadata_conflict",
                    "task.semantics",
                    renderer.task_text(annotation),
                    "Prompt contains concepts unsupported by task metadata: "
                    + ", ".join(unsupported),
                )
            )

    if not annotation.reference.use_for:
        issues.append(
            _error(
                "reference_scope_error",
                "reference.use_for",
                "no usable appearance scope",
                "Reference image does not provide any reliable appearance scope.",
            )
        )

    lowered_prompt = rendered_prompt.lower()
    for phrase in sorted(TRAJECTORY_PHRASES):
        if phrase in lowered_prompt:
            issues.append(
                _error(
                    "prompt_content_error",
                    "rendered_prompt",
                    phrase,
                    "Trajectory and action-phase detail must not enter Prompt.",
                )
            )

    try:
        renderer.validate_length(rendered_prompt)
    except PromptLengthError as error:
        issues.append(
            _error(
                "overdescription",
                "rendered_prompt",
                "rendered prompt",
                str(error),
            )
        )

    return ValidationResult(issues=issues)


def canonicalize_annotation(
    annotation: StructuredAnnotation,
) -> StructuredAnnotation:
    """Return a normalized deep copy after schema validation."""

    payload = annotation.model_dump()
    payload["sample_id"] = clean_text(payload["sample_id"])
    return StructuredAnnotation.model_validate(payload)
