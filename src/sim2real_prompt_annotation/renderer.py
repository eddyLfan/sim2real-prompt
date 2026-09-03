"""Deterministic canonical annotation to training-prompt renderer."""

from __future__ import annotations

import re

from .config import RendererConfig, VariantConfig
from .models import TEXT_FIELDS, StructuredAnnotation, clean_text

INSTRUCTION = (
    "Render the simulated video as a realistic real-world video.\n"
    "Preserve the geometry, motion, states, interactions, and temporal structure "
    "from the simulation.\n"
    "Use the reference image for the visual appearance of the robot, objects, and "
    "background environment, and use the text for semantic specification and "
    "explicit appearance controls."
)

FIELD_ORDER = (
    "robot",
    "camera",
    "task",
    "actions",
    "scene",
    "objects",
    "environment",
    "appearance",
    "lighting",
    "imaging",
    "preserve",
)


def _semantic_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _is_duplicate(text: str, accepted: list[str]) -> bool:
    tokens = _semantic_tokens(text)
    if not tokens:
        return True
    for previous in accepted:
        previous_tokens = _semantic_tokens(previous)
        if tokens == previous_tokens:
            return True
        union = tokens | previous_tokens
        if union and len(tokens & previous_tokens) / len(union) >= 0.94:
            return True
    return False


def _clean_sentences(text: str, accepted: list[str]) -> str:
    pieces = re.split(r"(?<=[.!?;])\s+", clean_text(text))
    output: list[str] = []
    local: list[str] = []
    for piece in pieces:
        piece = clean_text(piece)
        if not piece or _is_duplicate(piece, [*accepted, *local]):
            continue
        output.append(piece)
        local.append(piece)
    accepted.extend(local)
    return " ".join(output)


class PromptRenderer:
    def __init__(self, config: RendererConfig):
        self.config = config

    def _appearance(
        self, annotation: StructuredAnnotation, variant: VariantConfig
    ) -> str:
        if variant.appearance_mode == "none":
            return ""
        if variant.appearance_mode == "reference":
            return self.config.reference_appearance_text
        values: list[str] = []
        for item in annotation.appearance:
            if variant.omit_visible_in_reference and item.visible_in_reference:
                continue
            values.append(f"{item.entity}: {' '.join(item.attributes)}")
        return "; ".join(values)

    def _condition_lines(
        self, annotation: StructuredAnnotation, variant: VariantConfig
    ) -> list[str]:
        allowed = set(variant.fields)
        accepted: list[str] = []
        result: list[str] = []
        for field_name in FIELD_ORDER:
            if field_name not in allowed:
                continue
            if field_name == "appearance":
                value = self._appearance(annotation, variant)
            elif field_name == "environment" and variant.environment_reference_override:
                value = self.config.minimal_environment_text
            elif field_name in TEXT_FIELDS:
                field = getattr(annotation, field_name)
                value = field.text if field is not None else ""
            else:
                value = ""
            value = (
                _clean_sentences(value, accepted)
                if self.config.deduplicate_across_fields
                else clean_text(value)
            )
            if value:
                result.append(f"{field_name.title()}: {value}")
        return result

    def render(self, annotation: StructuredAnnotation, variant_name: str) -> str:
        if variant_name not in self.config.variants:
            raise KeyError(f"Unknown prompt variant: {variant_name}")
        header = f"[Instruction]\n{INSTRUCTION}\n\n[Condition]"
        output = header
        for line in self._condition_lines(
            annotation, self.config.variants[variant_name]
        ):
            candidate = f"{output}\n{line}"
            if len(candidate) <= self.config.max_prompt_length:
                output = candidate
                continue
            available = self.config.max_prompt_length - len(output) - 1
            if available > len(line.split(":", 1)[0]) + 4:
                truncated = line[:available].rsplit(" ", 1)[0].rstrip(" ,;:")
                output = f"{output}\n{truncated}"
            break
        return output.strip()

    def render_all(self, annotation: StructuredAnnotation) -> dict[str, str]:
        return {name: self.render(annotation, name) for name in self.config.variants}
