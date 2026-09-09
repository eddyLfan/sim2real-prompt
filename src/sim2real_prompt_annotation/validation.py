"""Pure validation for prompt and Reference branch products."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import PromptResult, ReferenceBranchResult, ReferenceQuery, clean_text

_PROMPT_FORBIDDEN_PHRASES = (
    "reference image",
    "reference crop",
    "bounding box",
    "segmentation mask",
    "sim-to-real",
    "simulation video",
    "real video",
    "input frame",
)


class ProductValidationError(ValueError):
    """A branch product is unsafe to publish to the training dataset."""


def prompt_word_count(prompt: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", prompt))


def normalize_prompt(prompt: str) -> str:
    value = clean_text(prompt).strip()
    if value and value[-1] not in ".!?":
        value += "."
    return value


def validate_prompt_result(
    result: PromptResult,
    *,
    max_words: int = 56,
    max_characters: int = 560,
) -> PromptResult:
    """Validate and normalize the one-sentence VLM product."""

    prompt = validate_prompt_text(
        result.prompt,
        max_words=max_words,
        max_characters=max_characters,
    )
    if not any(query.role == "primary" for query in result.reference_queries):
        raise ProductValidationError("VLM returned no primary task-object query")
    if any(
        query.role == "primary" and not query.required
        for query in result.reference_queries
    ):
        raise ProductValidationError("every primary task-object query must be required")
    return result.model_copy(update={"prompt": prompt})


def validate_prompt_text(
    value: str,
    *,
    max_words: int = 56,
    max_characters: int = 560,
) -> str:
    """Validate and normalize a published prompt without branch metadata."""

    prompt = normalize_prompt(value)
    if not prompt:
        raise ProductValidationError("prompt is empty")
    if len(prompt) > max_characters:
        raise ProductValidationError(
            f"prompt has {len(prompt)} characters; maximum is {max_characters}"
        )
    words = prompt_word_count(prompt)
    if words > max_words:
        raise ProductValidationError(
            f"prompt has {words} words; maximum is {max_words}"
        )
    terminals = re.findall(r"[.!?]+(?=\s|$)", prompt)
    if len(terminals) != 1:
        raise ProductValidationError("prompt must contain exactly one sentence")
    lowered = prompt.casefold()
    forbidden = next(
        (phrase for phrase in _PROMPT_FORBIDDEN_PHRASES if phrase in lowered), None
    )
    if forbidden is not None:
        raise ProductValidationError(
            f"prompt exposes preprocessing language: {forbidden!r}"
        )
    return prompt


def validate_reference_result(
    result: ReferenceBranchResult,
    queries: Sequence[ReferenceQuery],
    *,
    min_images: int = 1,
    max_images: int = 3,
) -> ReferenceBranchResult:
    """Verify detection coverage and the selected 1--3 first-frame crops."""

    selected = result.selected_artifacts
    if not min_images <= len(selected) <= max_images:
        raise ProductValidationError(
            f"Reference count {len(selected)} is outside {min_images}--{max_images}"
        )
    ids = [artifact.reference_id for artifact in selected]
    if len(ids) != len(set(ids)):
        raise ProductValidationError("selected Reference identities are not unique")
    if not any(artifact.role == "primary" for artifact in selected):
        raise ProductValidationError(
            "selected References contain no primary task object"
        )
    if any(artifact.source_frame_index != 0 for artifact in selected):
        raise ProductValidationError("every Reference must originate from Real frame 0")

    detected_queries = {item.query.casefold() for item in result.candidate_pool}
    missing = sorted(
        query.query
        for query in queries
        if query.required and query.query.casefold() not in detected_queries
    )
    if missing:
        raise ProductValidationError(
            "YOLOE did not detect required task entities: " + ", ".join(missing)
        )
    return result
