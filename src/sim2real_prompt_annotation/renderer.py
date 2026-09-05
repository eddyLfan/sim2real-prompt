"""Deterministically render one compact training prompt per episode."""

from __future__ import annotations

import re

from .config import RendererConfig
from .models import ReferenceScope, StructuredAnnotation, clean_text

SCOPE_TEXT: dict[ReferenceScope, str] = {
    "robot": "robot appearance",
    "objects": "task objects",
    "workspace": "workspace",
    "background": "background environment",
}


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
            return (
                "Do not copy visual details from the reference image; "
                "follow the explicit text attributes"
            )
        labels = _join_natural([SCOPE_TEXT[scope] for scope in scopes])
        return (
            f"Use only the {labels} from the reference image; "
            "explicit text attributes take priority"
        )

    def render(self, annotation: StructuredAnnotation) -> str:
        plan = annotation.prompt_plan
        sentences = [f"Real-world video of {plan.task_clause}."]
        sentences.append(f"{self._reference_instruction(plan.reference_scopes)}.")
        if plan.setting_clauses:
            sentences.append(
                f"Render the scene with {_join_natural(plan.setting_clauses)}."
            )
        prompt = clean_text(" ".join(sentences))
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
        return prompt
