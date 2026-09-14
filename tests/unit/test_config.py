from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sim2real_prompt_annotation.config import (
    REFERENCE_IMAGE_COUNT,
    PipelineConfig,
    load_config,
)


def test_defaults_freeze_real_only_scene_reference_contract() -> None:
    config = PipelineConfig()

    assert config.dataset.real_view == "camera_head"
    assert config.prompt.frame_count == 8
    assert config.prompt.sampling == "uniform"
    assert config.prompt.max_tokens == 256
    assert REFERENCE_IMAGE_COUNT == 1
    assert config.reference.backend == "robotseg"
    assert config.reference.model_path.name == "robotseg.pt"
    assert config.reference.inpainting_model_path.as_posix() == "weights/big-lama"
    assert config.reference.yoloe_text_model_path.as_posix() == (
        "weights/mobileclip_blt.ts"
    )
    assert config.reference.residual_check is True
    assert config.output.prompt_filename == "episodes_prompt.jsonl"
    assert config.output.reference_filename == "reference_images.jsonl"


def test_prompt_frame_count_cannot_drift_from_eight() -> None:
    with pytest.raises(ValidationError, match="Input should be 8"):
        PipelineConfig.model_validate({"prompt": {"frame_count": 6}})


def test_relative_paths_are_resolved_beside_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "run.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
dataset:
  root: ../dataset
  metadata_manifest: metadata.jsonl
output:
  root: ../output
reference:
  model_path: weights/robotseg.pt
  yoloe_model_path: weights/yoloe.pt
  yoloe_text_model_path: weights/mobileclip.ts
  inpainting_model_path: weights/big-lama
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dataset.root == tmp_path / "configs" / "../dataset"
    assert config.dataset.metadata_manifest == config_path.parent / "metadata.jsonl"
    assert config.output.root == tmp_path / "configs" / "../output"
    assert config.reference.model_path == config_path.parent / "weights/robotseg.pt"
    assert config.reference.yoloe_model_path == config_path.parent / "weights/yoloe.pt"
    assert config.reference.yoloe_text_model_path == (
        config_path.parent / "weights/mobileclip.ts"
    )
    assert (
        config.reference.inpainting_model_path
        == config_path.parent / "weights/big-lama"
    )


def test_reference_backend_is_explicit_and_has_no_automatic_yoloe_fallback() -> None:
    with pytest.raises(ValidationError, match="robotseg"):
        PipelineConfig.model_validate({"reference": {"backend": "yoloe"}})
    with pytest.raises(ValidationError, match="use_yoloe_fallback"):
        PipelineConfig.model_validate({"reference": {"use_yoloe_fallback": True}})


def test_reference_mask_bounds_and_kernel_are_strict() -> None:
    with pytest.raises(ValidationError, match="must satisfy min < max"):
        PipelineConfig.model_validate(
            {
                "reference": {
                    "min_mask_area_fraction": 0.8,
                    "max_mask_area_fraction": 0.5,
                }
            }
        )
    with pytest.raises(ValidationError, match="must be odd"):
        PipelineConfig.model_validate({"reference": {"mask_close_kernel": 4}})


def test_reference_model_sha256_must_be_lowercase_hex() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        PipelineConfig.model_validate({"reference": {"model_sha256": "A" * 64}})


def test_formal_config_pins_parent_mobileclip_asset() -> None:
    repository = Path(__file__).resolve().parents[2]

    config = load_config(repository / "config.test.yaml")

    assert (
        config.reference.yoloe_text_model_path.resolve()
        == (repository.parent / "weights/mobileclip_blt.ts").resolve()
    )
    assert config.reference.yoloe_text_model_sha256 == (
        "a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54"
    )
