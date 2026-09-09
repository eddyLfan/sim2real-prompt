from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim2real_prompt_annotation.config import DatasetConfig
from sim2real_prompt_annotation.dataset import (
    canonical_episode_id,
    discover_episodes,
    sample_artifact_stem,
)


def _write_dataset(root: Path, *, view: str = "camera_head") -> Path:
    dataset = root / "paired_demo"
    (dataset / "meta").mkdir(parents=True)
    info = {
        "source_id": "source:demo/one",
        "domain": "lab-a",
        "robot_type": "dual_arm",
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": "0:2"},
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            f"camera_observations.color_images.{view}": {"dtype": "video"},
            f"camera_observations.color_images.{view}_sim": {"dtype": "video"},
        },
    }
    (dataset / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    rows = [
        {"episode_index": index, "length": 81, "tasks": ["place the red cup"]}
        for index in range(2)
    ]
    (dataset / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    for index in range(2):
        path = (
            dataset
            / "videos/chunk-000"
            / f"camera_observations.color_images.{view}"
            / f"episode_{index:06d}.mp4"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not decoded during discovery")
    return dataset


def test_discovery_uses_only_configured_real_view(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)

    records = discover_episodes(DatasetConfig(root=dataset))

    assert len(records) == 2
    assert records[0].sample_id == "15:source:demo/one:0"
    assert records[0].source_id == "source:demo/one"
    assert records[0].domain == "lab-a"
    assert records[0].split == "train"
    assert records[0].real_view == "camera_head"
    assert records[0].real_video.name == "episode_000000.mp4"
    assert "_sim" not in str(records[0].real_video)


def test_configured_real_view_never_falls_back(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, view="camera_front")

    with pytest.raises(ValueError, match="camera_head.*unavailable"):
        discover_episodes(DatasetConfig(root=dataset, real_view="camera_head"))


def test_discovery_requires_paired_sim_metadata_without_reading_sim(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path)
    path = dataset / "meta/info.json"
    info = json.loads(path.read_text())
    del info["features"]["camera_observations.color_images.camera_head_sim"]
    path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="paired Sim view.*unavailable"):
        discover_episodes(DatasetConfig(root=dataset))


def test_domain_cannot_mix_splits(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info.pop("splits")
    info_path.write_text(json.dumps(info), encoding="utf-8")
    path = dataset / "meta/episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["split"] = "train"
    rows[1]["split"] = "validation"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="mixes.*splits"):
        discover_episodes(DatasetConfig(root=dataset))


def test_split_sources_must_agree(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    path = dataset / "meta/episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["split"] = "validation"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="row/info/external splits conflict"):
        discover_episodes(DatasetConfig(root=dataset))


@pytest.mark.parametrize("selector", ["0-2", ["0"], "2:0", True, "0:2,"])
def test_invalid_info_split_selector_fails_closed(
    tmp_path: Path,
    selector: object,
) -> None:
    dataset = _write_dataset(tmp_path)
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["splits"] = {"train": selector}
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="info.splits"):
        discover_episodes(DatasetConfig(root=dataset))


def test_canonical_identity_is_injective_and_artifact_safe() -> None:
    first = canonical_episode_id("a:b", 12)
    second = canonical_episode_id("a", 12)

    assert first != second
    assert len(sample_artifact_stem(first)) == 64
    assert "/" not in sample_artifact_stem(first)
