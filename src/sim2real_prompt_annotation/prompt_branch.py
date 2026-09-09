"""Real-only VLM branch for prompt text and YOLOE reference queries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import PROMPT_FRAME_COUNT
from .models import PromptPayload, PromptResult, RealFrame
from .qwen import VLMClient

DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts/prompt_system.txt"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _input_fingerprint(
    *,
    task_description: str,
    robot_metadata: Mapping[str, Any],
    images: Sequence[RealFrame],
    system_prompt: str,
    provider_identity: str,
    temperature: float,
    max_tokens: int,
) -> str:
    digest = hashlib.sha256()
    request = {
        "schema": "prompt-branch-v1",
        "task_description": task_description,
        "robot_metadata": robot_metadata,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "provider": provider_identity,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "frames": [
            {
                "frame_index": image.frame_index,
                "timestamp_seconds": image.timestamp_seconds,
                "jpeg_sha256": hashlib.sha256(image.jpeg).hexdigest(),
            }
            for image in images
        ],
    }
    digest.update(_canonical_json(request).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


class PromptBranch:
    """Make exactly one VLM call for one episode's prompt-side products."""

    def __init__(
        self,
        client: VLMClient,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
        provider_identity: str | None = None,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.client = client
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        ).strip()
        if not self.system_prompt:
            raise ValueError("system_prompt must be non-empty")
        self.temperature = temperature
        self.max_tokens = max_tokens
        client_config = getattr(client, "config", None)
        self.provider_identity = provider_identity or str(
            getattr(client_config, "model", type(client).__qualname__)
        )

    @staticmethod
    def _validate_images(images: Sequence[RealFrame]) -> tuple[RealFrame, ...]:
        ordered = tuple(images)
        if len(ordered) != PROMPT_FRAME_COUNT:
            raise ValueError(
                f"Prompt branch requires exactly {PROMPT_FRAME_COUNT} Real frames; "
                f"received {len(ordered)}"
            )
        indices = [image.frame_index for image in ordered]
        if indices != sorted(set(indices)):
            raise ValueError("Real prompt frames must have unique ascending indices")
        if indices[0] != 0:
            raise ValueError(
                "Real prompt frames must include frame 0 as the first frame"
            )
        return ordered

    @staticmethod
    def _user_text(task_description: str, robot_metadata: Mapping[str, Any]) -> str:
        authoritative_input = {
            "task_description": task_description,
            "robot_metadata": dict(robot_metadata),
        }
        return (
            "AUTHORITATIVE TASK AND ROBOT METADATA:\n"
            f"{_canonical_json(authoritative_input)}\n\n"
            "Use the eight ordered Real frames supplied after this text only for "
            "visible execution and scene context. Return the prompt and YOLOE "
            "reference queries in the requested schema."
        )

    def run(
        self,
        *,
        sample_id: str,
        task_description: str,
        robot_metadata: Mapping[str, Any],
        images: Sequence[RealFrame],
    ) -> PromptResult:
        sample_id = sample_id.strip()
        task_description = task_description.strip()
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        if not task_description:
            raise ValueError("task_description must be non-empty")
        ordered = self._validate_images(images)
        fingerprint = _input_fingerprint(
            task_description=task_description,
            robot_metadata=robot_metadata,
            images=ordered,
            system_prompt=self.system_prompt,
            provider_identity=self.provider_identity,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Deliberately one call: caching, rate limiting, and retries belong to an
        # injected VLMClient service and therefore cover both output fields together.
        response = self.client.generate(
            sample_id=sample_id,
            stage="prompt",
            system_prompt=self.system_prompt,
            user_text=self._user_text(task_description, robot_metadata),
            images=ordered,
            response_model=PromptPayload,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        payload = PromptPayload.model_validate(response.payload.model_dump())
        return PromptResult(
            sample_id=sample_id,
            prompt=payload.prompt,
            reference_queries=payload.reference_queries,
            model=response.model,
            request_id=response.request_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            frame_indices=[image.frame_index for image in ordered],
            input_fingerprint=fingerprint,
        )

    def cache_identity(self) -> dict[str, object]:
        """Return only inputs that can change this branch's semantic product."""

        return {
            "system_prompt_sha256": hashlib.sha256(
                self.system_prompt.encode("utf-8")
            ).hexdigest(),
            "provider_identity": self.provider_identity,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def generate(
        self,
        *,
        sample_id: str,
        task_description: str,
        robot_metadata: Mapping[str, Any],
        images: Sequence[RealFrame],
    ) -> PromptResult:
        """Alias for callers that model the branch as a provider service."""

        return self.run(
            sample_id=sample_id,
            task_description=task_description,
            robot_metadata=robot_metadata,
            images=images,
        )
