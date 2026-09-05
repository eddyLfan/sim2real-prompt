"""Centralized configuration for the project-specific annotation pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("SIM2REAL_PROMPT_DATASET_ROOT", str(Path.cwd() / "data"))
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("SIM2REAL_PROMPT_OUTPUT_ROOT", str(Path.cwd() / "outputs"))
)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProviderConfig(ConfigModel):
    name: Literal["qwen_openai"] = "qwen_openai"
    model: str = "qwen3.7-plus"
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: str | None = None
    base_url_env: str = "DASHSCOPE_BASE_URL"
    response_format: Literal["json_object", "json_schema"] = "json_object"
    enable_thinking: bool = False
    timeout_seconds: float = Field(default=180.0, gt=0)

    def resolved_base_url(self) -> str:
        value = self.base_url or os.getenv(self.base_url_env)
        if not value:
            raise ValueError(
                f"Set provider.base_url or environment variable {self.base_url_env}. "
                "Alibaba Model Studio endpoints are region/workspace specific."
            )
        return value.rstrip("/")


class MediaConfig(ConfigModel):
    mode: Literal["selected_frames", "native_video"] = "selected_frames"
    strategy: Literal["uniform", "keyframe"] = "uniform"
    max_frames: int = Field(default=6, ge=2, le=64)
    resize_long_edge: int = Field(default=512, ge=64, le=4096)
    jpeg_quality: int = Field(default=85, ge=30, le=100)
    views: list[str] = Field(default_factory=list)
    reference_view: str = "camera_head"
    reference_seed: int = 42
    native_video_fps: float = Field(default=2.0, ge=0.1, le=10.0)
    max_native_video_mb: int = Field(default=100, ge=1)


class DiscoveryConfig(ConfigModel):
    min_episode_frames: int = Field(default=1, ge=1)
    required_views: list[str] = Field(default_factory=list)

    @field_validator("required_views")
    @classmethod
    def normalize_required_views(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class AnnotationConfig(ConfigModel):
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=3000, ge=256)
    retry_count: int = Field(default=2, ge=0, le=10)
    severe_retry_count: int = Field(default=1, ge=0, le=5)
    confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    system_prompt: Path = PACKAGE_ROOT / "prompts/annotation_system.txt"


class CriticConfig(ConfigModel):
    enabled: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1800, ge=256)
    retry_count: int = Field(default=2, ge=0, le=10)
    fail_on_severe_after_retries: bool = True
    system_prompt: Path = PACKAGE_ROOT / "prompts/critic_system.txt"


class RendererConfig(ConfigModel):
    max_prompt_words: int = Field(default=55, ge=15, le=160)
    max_prompt_characters: int = Field(default=480, ge=120, le=2000)


class BatchConfig(ConfigModel):
    concurrency: int = Field(default=4, ge=1, le=64)
    api_retry_count: int = Field(default=4, ge=0, le=20)
    backoff_initial_seconds: float = Field(default=1.0, ge=0.0)
    backoff_max_seconds: float = Field(default=30.0, ge=0.0)
    failed_retry_rounds: int = Field(default=1, ge=0, le=10)
    failed_retry_backoff_seconds: float = Field(default=2.0, ge=0.0)
    skip_completed: bool = True
    require_complete: bool = True
    log_costs: bool = True
    input_cost_per_million: float | None = Field(default=None, ge=0.0)
    output_cost_per_million: float | None = Field(default=None, ge=0.0)


class DatasetPromptExportConfig(ConfigModel):
    enabled: bool = True
    filename: str = "episodes_prompt.jsonl"
    merge_existing: bool = True

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        path = Path(value)
        if path.name != value or path.suffix != ".jsonl":
            raise ValueError("dataset prompt filename must be a plain .jsonl filename")
        return value


class PipelineConfig(ConfigModel):
    dataset_root: Path = DEFAULT_DATASET_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    metadata_manifest: Path | None = None
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    renderer: RendererConfig = Field(default_factory=RendererConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    dataset_prompt_export: DatasetPromptExportConfig = Field(
        default_factory=DatasetPromptExportConfig
    )


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    def resolve_relative(value: object) -> object:
        if value is None:
            return None
        candidate = Path(value).expanduser()  # type: ignore[arg-type]
        return candidate if candidate.is_absolute() else path.parent / candidate

    for field_name in ("dataset_root", "output_root", "metadata_manifest"):
        if field_name in payload:
            payload[field_name] = resolve_relative(payload[field_name])
    for section in ("annotation", "critic"):
        section_payload = payload.get(section)
        if isinstance(section_payload, dict) and "system_prompt" in section_payload:
            section_payload["system_prompt"] = resolve_relative(
                section_payload["system_prompt"]
            )

    dataset_root = os.environ.get("SIM2REAL_PROMPT_DATASET_ROOT")
    output_root = os.environ.get("SIM2REAL_PROMPT_OUTPUT_ROOT")
    if dataset_root:
        payload["dataset_root"] = Path(dataset_root).expanduser().resolve()
    if output_root:
        payload["output_root"] = Path(output_root).expanduser().resolve()
    return PipelineConfig.model_validate(payload)
