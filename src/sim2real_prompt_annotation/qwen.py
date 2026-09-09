"""Structured VLM provider used by the Real-only prompt branch."""

from __future__ import annotations

import base64
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import ProviderConfig
from .models import RealFrame


class ResponseParseError(ValueError):
    """The provider returned content that does not satisfy the requested schema."""

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


@dataclass(frozen=True)
class VLMResponse:
    payload: BaseModel
    raw_text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None


class VLMClient(ABC):
    """A cache/retry wrapper may implement this same one-request interface."""

    @abstractmethod
    def generate(
        self,
        *,
        sample_id: str,
        stage: str,
        system_prompt: str,
        user_text: str,
        images: Sequence[RealFrame],
        response_model: type[BaseModel],
        temperature: float,
        max_tokens: int,
    ) -> VLMResponse:
        """Make one provider request and validate one structured response."""


def response_schema_instruction(response_model: type[BaseModel]) -> str:
    schema: dict[str, Any] = response_model.model_json_schema()
    return (
        "Return only one JSON object matching this JSON Schema. Do not wrap it in "
        "Markdown fences:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )


def _data_url(payload: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(payload).decode('ascii')}"


def _json_text(raw: str) -> str:
    value = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.I)
    return fenced.group(1).strip() if fenced else value


class QwenOpenAIClient(VLMClient):
    """OpenAI-compatible Qwen client with no hidden retry or media discovery."""

    def __init__(self, config: ProviderConfig):
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key. Export {config.api_key_env}; "
                "never put it in YAML or code."
            )
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.resolved_base_url(),
            timeout=config.timeout_seconds,
            # Retries are deliberately owned by the pipeline/service layer so a
            # prompt result is cached and retried as one logical operation.
            max_retries=0,
        )

    @staticmethod
    def _content(user_text: str, images: Sequence[RealFrame]) -> list[dict[str, Any]]:
        """Build an ordered Real-frame payload without Sim or native-video inputs."""

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for position, image in enumerate(images):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"REAL FRAME {position + 1}/{len(images)}: "
                        f"source_index={image.frame_index}, "
                        f"time={image.timestamp_seconds:.3f}s"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(image.jpeg)},
                }
            )
        return content

    def _response_format(
        self, stage: str, response_model: type[BaseModel]
    ) -> dict[str, Any]:
        if self.config.response_format == "json_schema":
            name = re.sub(r"[^A-Za-z0-9_-]", "_", stage)[:64] or "response"
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            }
        return {"type": "json_object"}

    def generate(
        self,
        *,
        sample_id: str,
        stage: str,
        system_prompt: str,
        user_text: str,
        images: Sequence[RealFrame],
        response_model: type[BaseModel],
        temperature: float,
        max_tokens: int,
    ) -> VLMResponse:
        del sample_id  # Correlation is kept by the caller, not sent to the model.
        if not images:
            raise ValueError("At least one Real frame is required")

        schema_text = response_schema_instruction(response_model)
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": self._content(f"{user_text}\n\n{schema_text}", images),
                },
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format=self._response_format(stage, response_model),
            extra_body={"enable_thinking": self.config.enable_thinking},
        )
        raw = completion.choices[0].message.content or ""
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False)
        try:
            payload = response_model.model_validate_json(_json_text(raw))
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            raise ResponseParseError(
                f"{stage} returned invalid structured JSON: {error}", raw
            ) from error

        usage = completion.usage
        return VLMResponse(
            payload=payload,
            raw_text=raw,
            model=completion.model or self.config.model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            request_id=completion.id,
        )
