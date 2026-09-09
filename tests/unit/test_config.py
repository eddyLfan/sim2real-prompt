from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sim2real_prompt_annotation.config import PipelineConfig, load_config


def test_defaults_freeze_real_only_eight_frame_contract() -> None:
    config = PipelineConfig()

    assert config.dataset.real_view == "camera_head"
    assert config.prompt.frame_count == 8
    assert config.prompt.sampling == "uniform"
    assert config.prompt.max_tokens == 256
    assert config.reference.backend == "yoloe"
    assert config.reference.model_path.name == "yoloe-11s-seg.pt"
    assert (config.reference.min_images, config.reference.max_images) == (1, 3)
    assert config.reference.embedding_cache_size == 64
    assert config.reference.duplicate_iou == 0.85
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
  model_path: weights/yoloe.pt
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dataset.root == tmp_path / "configs" / "../dataset"
    assert config.dataset.metadata_manifest == config_path.parent / "metadata.jsonl"
    assert config.output.root == tmp_path / "configs" / "../output"
    assert config.reference.model_path == config_path.parent / "weights/yoloe.pt"


def test_reference_selection_bounds_are_consistent() -> None:
    with pytest.raises(ValidationError, match="cannot exceed max_images"):
        PipelineConfig.model_validate({"reference": {"min_images": 3, "max_images": 2}})
    with pytest.raises(ValidationError, match="cannot be smaller"):
        PipelineConfig.model_validate(
            {"reference": {"candidate_pool_size": 2, "max_images": 3}}
        )
