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
REFERENCE_IMAGE_COUNT = 1


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
    model: str = "qwen3.5-plus"
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
    """Frame-zero robot removal and full-scene Reference construction."""

    # RobotSeg is the authoritative robot-specific segmenter. Its official runtime
    # is loaded lazily, so metadata-only commands do not need the optional package.
    backend: Literal["robotseg"] = "robotseg"
    model_path: Path = Path("robotseg.pt")
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    robotseg_config: str = "configs/robotseg-infer"
    robot_category: Literal["robot", "arm", "gripper"] = "robot"
    device: str = "cuda:0"
    batch_size: int = Field(default=16, ge=1, le=512)

    # A small fixed vocabulary keeps YOLOE text embeddings and detector batches
    # reusable. YOLOE is never a mask fallback; it is used only for residual robot
    # quality control after inpainting.
    robot_queries: tuple[str, ...] = Field(
        default=("robot", "robot arm", "robot gripper"),
        min_length=1,
        max_length=8,
    )
    residual_check: bool = True
    yoloe_model_path: Path = Path("yoloe-11s-seg.pt")
    yoloe_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    yoloe_text_model_path: Path = Path("weights/mobileclip_blt.ts")
    yoloe_text_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    yoloe_image_size: int = Field(default=640, ge=128, le=4096)
    yoloe_confidence: float = Field(default=0.05, ge=0.0, le=1.0)
    yoloe_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding_cache_size: int = Field(default=16, ge=1, le=4096)

    # Morphology intentionally expands the removal region beyond the predicted
    # silhouette so robot-colored edge pixels do not survive the composite.
    mask_close_kernel: int = Field(default=9, ge=1, le=255)
    mask_dilation_pixels: int = Field(default=12, ge=0, le=512)
    min_mask_area_fraction: float = Field(default=0.002, gt=0.0, lt=1.0)
    max_mask_area_fraction: float = Field(default=0.75, gt=0.0, lt=1.0)
    max_residual_area_fraction: float = Field(default=0.002, ge=0.0, lt=1.0)
    min_inpaint_change_fraction: float = Field(default=0.01, ge=0.0, le=1.0)

    # The official Big-LaMa layout is a local directory containing config.yaml and
    # models/best.ckpt. A TorchScript .pt file is also accepted for deployments that
    # have exported one. Runtime code never downloads models implicitly.
    inpainting_backend: Literal["big_lama"] = "big_lama"
    inpainting_model_path: Path = Path("weights/big-lama")
    inpainting_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    inpainting_device: str | None = None
    inpainting_modulo: int = Field(default=8, ge=1, le=128)
    jpeg_quality: int = Field(default=95, ge=30, le=100)

    @field_validator("device", "inpainting_device")
    @classmethod
    def nonempty_device(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("reference.device must be non-empty")
        return value

    @field_validator("robotseg_config")
    @classmethod
    def nonempty_robotseg_config(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reference.robotseg_config must be non-empty")
        return value

    @field_validator("robot_queries")
    @classmethod
    def normalized_robot_queries(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for raw in values:
            value = " ".join(raw.split()).strip(" ,.;:")
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                result.append(value)
        if not result:
            raise ValueError("reference.robot_queries must not be empty")
        return tuple(result)

    @model_validator(mode="after")
    def valid_quality_bounds(self) -> ReferenceConfig:
        if self.mask_close_kernel % 2 == 0:
            raise ValueError("reference.mask_close_kernel must be odd")
        if self.min_mask_area_fraction >= self.max_mask_area_fraction:
            raise ValueError("reference mask area fractions must satisfy min < max")
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
        ("reference", "yoloe_model_path"),
        ("reference", "yoloe_text_model_path"),
        ("reference", "inpainting_model_path"),
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
