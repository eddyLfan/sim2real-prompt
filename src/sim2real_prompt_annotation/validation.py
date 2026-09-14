"""Pure validation for Prompt and robot-removed scene Reference products."""

from __future__ import annotations

import hashlib
import math
import re

import cv2
import numpy as np

from .models import PromptResult, SceneReferenceResult, clean_text

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


def validate_reference_result(result: SceneReferenceResult) -> SceneReferenceResult:
    """Fail closed unless the branch produced one full-size frame-zero scene."""

    artifact = result.artifact
    if artifact.source_frame_index != 0:
        raise ProductValidationError("scene Reference must come from Real frame zero")
    if artifact.scope != "environment":
        raise ProductValidationError("scene Reference scope must be 'environment'")
    if artifact.reference_kind != "robot_removed_scene":
        raise ProductValidationError("Reference kind must be 'robot_removed_scene'")
    if not result.robot_masks:
        raise ProductValidationError("scene Reference has no robot-mask evidence")
    if not artifact.provenance:
        raise ProductValidationError("scene Reference provenance must not be empty")
    if artifact.provenance.get("operation") != "robot_removal_inpainting":
        raise ProductValidationError("scene Reference provenance has wrong operation")
    segmenter = artifact.provenance.get("segmenter")
    if not isinstance(segmenter, dict) or not segmenter:
        raise ProductValidationError(
            "scene Reference provenance lacks segmenter identity"
        )
    inpainter = artifact.provenance.get("inpainter")
    if not isinstance(inpainter, dict) or not inpainter:
        raise ProductValidationError(
            "scene Reference provenance lacks inpainter identity"
        )
    final_mask = artifact.provenance.get("final_mask")
    if not isinstance(final_mask, dict):
        raise ProductValidationError("scene Reference provenance lacks final mask")
    final_area = final_mask.get("area_fraction")
    if (
        final_mask.get("sha256") != artifact.mask_sha256
        or isinstance(final_area, bool)
        or not isinstance(final_area, (int, float))
        or not math.isclose(final_area, artifact.mask_area_fraction)
    ):
        raise ProductValidationError(
            "scene Reference final-mask provenance differs from artifact"
        )
    quality_control = artifact.provenance.get("quality_control")
    if (
        not isinstance(quality_control, dict)
        or quality_control.get("outside_mask_unchanged") is not True
    ):
        raise ProductValidationError(
            "scene Reference does not prove pixels outside the mask were preserved"
        )
    residual_qa = artifact.provenance.get("residual_qa")
    if not isinstance(residual_qa, dict):
        raise ProductValidationError("scene Reference provenance lacks residual QA")
    expected_residual_fields = {
        "enabled",
        "detector",
        "queries",
        "area_fraction",
        "threshold",
        "pass",
    }
    if set(residual_qa) != expected_residual_fields:
        raise ProductValidationError(
            "scene Reference residual QA fields are incomplete or unexpected"
        )
    enabled = residual_qa.get("enabled")
    detector = residual_qa.get("detector")
    queries = residual_qa.get("queries")
    residual_area = residual_qa.get("area_fraction")
    residual_threshold = residual_qa.get("threshold")
    residual_pass = residual_qa.get("pass")
    if not isinstance(enabled, bool):
        raise ProductValidationError("scene Reference residual QA enabled is invalid")
    if (enabled and (not isinstance(detector, dict) or not detector)) or (
        not enabled and detector is not None
    ):
        raise ProductValidationError(
            "scene Reference residual QA detector identity is inconsistent"
        )
    if (
        not isinstance(queries, list)
        or not queries
        or any(not isinstance(query, str) or not query.strip() for query in queries)
    ):
        raise ProductValidationError("scene Reference residual QA queries are invalid")
    if (
        isinstance(residual_area, bool)
        or not isinstance(residual_area, (int, float))
        or not math.isfinite(residual_area)
        or not 0.0 <= residual_area <= 1.0
        or isinstance(residual_threshold, bool)
        or not isinstance(residual_threshold, (int, float))
        or not math.isfinite(residual_threshold)
        or not 0.0 <= residual_threshold < 1.0
        or not isinstance(residual_pass, bool)
    ):
        raise ProductValidationError("scene Reference residual QA metrics are invalid")
    expected_pass = residual_area <= residual_threshold
    if residual_pass is not expected_pass or residual_pass is not True:
        raise ProductValidationError("scene Reference failed residual robot QA")
    quality_enabled = quality_control.get("residual_check_enabled")
    quality_area = quality_control.get("residual_mask_area_fraction")
    quality_threshold = quality_control.get("max_residual_area_fraction")
    if (
        quality_enabled is not enabled
        or isinstance(quality_area, bool)
        or not isinstance(quality_area, (int, float))
        or not math.isfinite(quality_area)
        or not 0.0 <= quality_area <= 1.0
        or not math.isclose(quality_area, residual_area)
        or isinstance(quality_threshold, bool)
        or not isinstance(quality_threshold, (int, float))
        or not math.isfinite(quality_threshold)
        or not 0.0 <= quality_threshold < 1.0
        or not math.isclose(quality_threshold, residual_threshold)
    ):
        raise ProductValidationError(
            "scene Reference residual QA differs from quality control"
        )

    digest = hashlib.sha256(artifact.jpeg).hexdigest()
    if digest != artifact.sha256 or artifact.reference_id != f"sha256:{digest}":
        raise ProductValidationError("scene Reference JPEG identity is inconsistent")
    decoded = cv2.imdecode(
        np.frombuffer(artifact.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if decoded is None or decoded.size == 0:
        raise ProductValidationError("scene Reference is not a decodable JPEG")
    if decoded.shape[:2] != (artifact.height, artifact.width):
        raise ProductValidationError(
            "scene Reference JPEG dimensions differ from artifact metadata"
        )
    return result
