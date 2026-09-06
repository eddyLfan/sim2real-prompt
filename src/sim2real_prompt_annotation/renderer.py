"""Deterministically render one compact training prompt per episode."""

from __future__ import annotations

import re

from .config import RendererConfig
from .models import ReferenceScope, StructuredAnnotation, clean_text

SCOPE_TEXT: dict[ReferenceScope, str] = {
    "robot": "robot appearance",
    "objects": "task-object appearance",
    "workspace": "workspace appearance",
    "background": "background appearance",
}

APPEARANCE_WORD_LIMITS = {"workspace": 6, "background": 8, "lighting": 6}


class PromptLengthError(ValueError):
    """Raised instead of silently truncating important conditioning content."""


def _join_natural(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def prompt_word_count(prompt: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", prompt))


class PromptRenderer:
    def __init__(self, config: RendererConfig):
        self.config = config

    @staticmethod
    def _reference_instruction(scopes: list[ReferenceScope]) -> str:
        if not scopes:
            return "Reference image has no usable appearance scope"
        labels = _join_natural([SCOPE_TEXT[scope] for scope in scopes])
        return f"Match {labels} to the reference image"

    @staticmethod
    def _object_is_goal_target(object_text: str, goal_text: str) -> bool:
        """Avoid rendering a destination as both an object and the goal."""

        def normalize(value: str) -> str:
            words = re.findall(r"[a-z0-9]+", value.lower())
            while words and words[0] in {"a", "an", "the"}:
                words.pop(0)
            return " ".join(words)

        object_key = normalize(object_text)
        goal_key = normalize(goal_text)
        return bool(object_key and re.search(rf"\b{re.escape(object_key)}\b", goal_key))

    @staticmethod
    def task_text(annotation: StructuredAnnotation) -> str:
        """Render metadata-only task slots without consulting visual descriptions."""

        slots = annotation.task.semantics
        arm = {
            "left": "its left arm",
            "right": "its right arm",
            "both": "both arms",
            "unspecified": "its manipulators",
        }[slots.active_arm]
        object_values = [value.text for value in slots.primary_objects]
        if slots.goal is not None and len(object_values) > 1:
            retained = [
                value
                for value in object_values
                if not PromptRenderer._object_is_goal_target(value, slots.goal.text)
            ]
            if retained:
                object_values = retained
        objects = _join_natural(object_values)
        values = [
            f"the {slots.robot} using {arm} to {slots.action.text} {objects}",
        ]
        if slots.goal:
            values.append(slots.goal.text)
        if slots.constraints:
            values.append(_join_natural([value.text for value in slots.constraints]))
        return clean_text(" ".join(values)).rstrip(".!?;:, ")

    @staticmethod
    def appearance_text(annotation: StructuredAnnotation) -> str:
        """Render bounded coarse setting fields while retaining full annotation."""

        visuals = annotation.target_visuals
        values: list[str] = []
        for name in ("workspace", "background", "lighting"):
            value = getattr(visuals, name)
            if value is None:
                continue
            words = value.text.split()[: APPEARANCE_WORD_LIMITS[name]]
            while words and words[-1].lower().strip(".,;:") in {
                "and",
                "with",
                "of",
                "on",
                "in",
                "at",
                "from",
            }:
                words.pop()
            if words:
                values.append(" ".join(words).rstrip(".,;:"))
        return _join_natural(values)

    def render(self, annotation: StructuredAnnotation) -> str:
        sentences = [f"Real-world video of {self.task_text(annotation)}."]
        sentences.append(
            f"{self._reference_instruction(annotation.reference.use_for)}."
        )
        appearance = self.appearance_text(annotation)
        if appearance:
            sentences.append(f"Render the scene with {appearance}.")
        return clean_text(" ".join(sentences))

    def validate_length(self, prompt: str) -> None:
        """Validate an already-rendered prompt without coupling this to rendering."""

        words = prompt_word_count(prompt)
        if words > self.config.max_prompt_words:
            raise PromptLengthError(
                f"Prompt has {words} words; maximum is "
                f"{self.config.max_prompt_words}: {prompt}"
            )
        if len(prompt) > self.config.max_prompt_characters:
            raise PromptLengthError(
                f"Prompt has {len(prompt)} characters; maximum is "
                f"{self.config.max_prompt_characters}: {prompt}"
            )
