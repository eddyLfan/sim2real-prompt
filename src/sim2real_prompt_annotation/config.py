"""Configuration for the Real-only prompt and Reference preprocessing pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("SIM2REAL_PROMPT_DATASET_ROOT", str(Path.cwd() / "data"))
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("SIM2REAL_PROMPT_OUTPUT_ROOT", str(Path.cwd() / "outputs"))
)
MIN_EPISODE_FRAMES = 81
PROMPT_FRAME_COUNT = 8
MAX_REFERENCE_IMAGES = 3


class ConfigModel(BaseModel):
    """Strict base class shared by every YAML configuration section."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DatasetConfig(ConfigModel):
    """Strict LeRobot discovery and the single Real camera used downstream."""

    root: Path = DEFAULT_DATASET_ROOT
    dataset_glob: str = "*"
    real_view: str = "camera_head"
    min_episode_frames: int = Field(default=MIN_EPISODE_FRAMES, ge=1)
    metadata_manifest: Path | None = None
    split_manifest: Path | None = None

    @field_validator("dataset_glob", "real_view")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be a non-empty string")
        return value


class ProviderConfig(ConfigModel):
    """OpenAI-compatible VLM endpoint configuration."""

    name: Literal["qwen_openai"] = "qwen_openai"
    model: str = "qwen3.7-plus"
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: str | None = None
    base_url_env: str = "DASHSCOPE_BASE_URL"
    response_format: Literal["json_object", "json_schema"] = "json_object"
    enable_thinking: bool = False
    timeout_seconds: float = Field(default=180.0, gt=0)

    @field_validator("model", "api_key_env", "base_url_env")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be a non-empty string")
        return value

    def resolved_base_url(self) -> str:
        value = self.base_url or os.getenv(self.base_url_env)
        if not value:
            raise ValueError(
                f"Set prompt.provider.base_url or environment variable "
                f"{self.base_url_env}."
            )
        return value.rstrip("/")


class PromptConfig(ConfigModel):
    """Eight-frame Real-video VLM request and transport policy."""

    frame_count: Literal[8] = PROMPT_FRAME_COUNT
    sampling: Literal["uniform"] = "uniform"
    resize_long_edge: int = Field(default=512, ge=64, le=4096)
    jpeg_quality: int = Field(default=85, ge=30, le=100)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    system_prompt: Path = PACKAGE_ROOT / "prompts/prompt_system.txt"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=64, le=4096)


class ReferenceConfig(ConfigModel):
    """YOLOE-S Seg inference and deterministic Multi-Reference selection."""

    backend: Literal["yoloe"] = "yoloe"
    model_path: Path = Path("yoloe-11s-seg.pt")
    device: str = "cuda:0"
    image_size: int = Field(default=640, ge=128, le=4096)
    batch_size: int = Field(default=32, ge=1, le=512)
    embedding_cache_size: int = Field(default=64, ge=1, le=4096)
    # Open-vocabulary task nouns score lower than closed-set COCO labels. 0.15
    # retains accurate masks on the production smoke corpus while the branch's
    # mask/area/primary checks still reject unusable crops.
    confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    duplicate_iou: float = Field(default=0.85, ge=0.0, le=1.0)
    crop_padding: float = Field(default=0.12, ge=0.0, le=0.5)
    candidate_pool_size: int = Field(default=8, ge=1, le=64)
    min_images: int = Field(default=1, ge=1, le=MAX_REFERENCE_IMAGES)
    max_images: int = Field(default=MAX_REFERENCE_IMAGES, ge=1, le=MAX_REFERENCE_IMAGES)
    selection_seed: int = 42
    jpeg_quality: int = Field(default=95, ge=30, le=100)

    @field_validator("device")
    @classmethod
    def nonempty_device(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reference.device must be non-empty")
        return value

    @model_validator(mode="after")
    def valid_reference_counts(self) -> ReferenceConfig:
        if self.min_images > self.max_images:
            raise ValueError("reference.min_images cannot exceed max_images")
        if self.candidate_pool_size < self.max_images:
            raise ValueError(
                "reference.candidate_pool_size cannot be smaller than max_images"
            )
        return self


class RuntimeConfig(ConfigModel):
    """Concurrency, retries, resumability, and failure policy."""

    decode_workers: int = Field(default=4, ge=1, le=64)
    api_concurrency: int = Field(default=4, ge=1, le=64)
    api_retry_count: int = Field(default=4, ge=0, le=20)
    backoff_initial_seconds: float = Field(default=1.0, ge=0.0)
    backoff_max_seconds: float = Field(default=30.0, ge=0.0)
    resume: bool = True
    fail_fast: bool = False
    log_costs: bool = True
    input_cost_per_million: float | None = Field(default=None, ge=0.0)
    output_cost_per_million: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def valid_backoff(self) -> RuntimeConfig:
        if self.backoff_initial_seconds > self.backoff_max_seconds:
            raise ValueError(
                "runtime.backoff_initial_seconds cannot exceed backoff_max_seconds"
            )
        return self


class OutputConfig(ConfigModel):
    """Intermediate checkpoints and the two published dataset manifests."""

    root: Path = DEFAULT_OUTPUT_ROOT
    prompt_filename: Literal["episodes_prompt.jsonl"] = "episodes_prompt.jsonl"
    reference_filename: Literal["reference_images.jsonl"] = "reference_images.jsonl"
    reference_directory: Literal["Reference"] = "Reference"


class PipelineConfig(ConfigModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def _resolve_relative_paths(payload: dict[str, Any], config_path: Path) -> None:
    """Resolve YAML-owned paths without changing paths supplied by environment."""

    path_fields = (
        ("dataset", "root"),
        ("dataset", "metadata_manifest"),
        ("dataset", "split_manifest"),
        ("prompt", "system_prompt"),
        ("reference", "model_path"),
        ("output", "root"),
    )
    for section_name, field_name in path_fields:
        section = payload.get(section_name)
        if not isinstance(section, dict) or field_name not in section:
            continue
        value = section[field_name]
        if value is None:
            continue
        candidate = Path(value).expanduser()
        section[field_name] = (
            candidate if candidate.is_absolute() else config_path.parent / candidate
        )


def load_config(path: str | Path) -> PipelineConfig:
    """Load strict YAML and resolve relative filesystem paths beside that YAML."""

    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    _resolve_relative_paths(payload, config_path)

    dataset_root = os.environ.get("SIM2REAL_PROMPT_DATASET_ROOT")
    output_root = os.environ.get("SIM2REAL_PROMPT_OUTPUT_ROOT")
    if dataset_root:
        payload.setdefault("dataset", {})["root"] = Path(dataset_root).expanduser()
    if output_root:
        payload.setdefault("output", {})["root"] = Path(output_root).expanduser()
    return PipelineConfig.model_validate(payload)
