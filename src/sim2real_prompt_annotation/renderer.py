"""Deterministically render one compact training prompt per episode."""

from __future__ import annotations

import re

from .config import RendererConfig
from .models import ReferenceScope, StructuredAnnotation, clean_text

SCOPE_TEXT: dict[ReferenceScope, str] = {
    "robot": "robot",
    "objects": "object",
    "workspace": "workspace",
    "background": "background",
    "lighting": "lighting",
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
        labels = _join_natural([SCOPE_TEXT[scope] for scope in scopes])
        target = f"fine {labels} details" if labels else "fine visual details"
        return (
            f"Use the reference for {target}; explicit text attributes take priority."
        )

    def render(self, annotation: StructuredAnnotation) -> str:
        plan = annotation.prompt_plan
        sentences = [f"Real-world video of {plan.task_clause}."]
        if plan.setting_clauses:
            prefix = "Setting: " if self.config.include_setting_label else ""
            sentences.append(f"{prefix}{'; '.join(plan.setting_clauses)}.")
        sentences.append(self._reference_instruction(plan.reference_scopes))
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
