from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sim2real_prompt_annotation import DatasetProcessingPipeline
from sim2real_prompt_annotation import processing as processing_module
from sim2real_prompt_annotation.dataset_validation import inspect_dataset


def _write_video(path: Path, frames: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    writer.release()


def _dataset(root: Path) -> Path:
    dataset = root / "paired_demo"
    (dataset / "meta").mkdir(parents=True)
    features: dict[str, dict[str, object]] = {
        "camera_head": {"dtype": "video", "shape": [48, 64, 3]},
        "camera_head_sim": {"dtype": "video", "shape": [48, 64, 3]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "dual_arm",
        "total_episodes": 2,
        "total_frames": 8,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }
    (dataset / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (dataset / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "place object"}) + "\n", encoding="utf-8"
    )
    episode_rows = []
    episode_stats_rows = []
    for episode_index in range(2):
        episode_rows.append(
            json.dumps(
                {"episode_index": episode_index, "length": 4, "tasks": ["place object"]}
            )
        )
        episode_stats_rows.append(
            json.dumps({"episode_index": episode_index, "stats": {}})
        )
        values: dict[str, pa.Array] = {
            "episode_index": pa.array([episode_index] * 4, type=pa.int64()),
            "frame_index": pa.array(range(4), type=pa.int64()),
            "index": pa.array(
                range(episode_index * 4, episode_index * 4 + 4), type=pa.int64()
            ),
            "task_index": pa.array([0] * 4, type=pa.int64()),
        }
        parquet_path = dataset / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(values), parquet_path)
        for key in ("camera_head", "camera_head_sim"):
            _write_video(
                dataset / f"videos/chunk-000/{key}/episode_{episode_index:06d}.mp4"
            )
    (dataset / "meta/episodes.jsonl").write_text(
        "\n".join(episode_rows) + "\n", encoding="utf-8"
    )
    (dataset / "meta/episodes_stats.jsonl").write_text(
        "\n".join(episode_stats_rows) + "\n", encoding="utf-8"
    )
    (dataset / "meta/stats.json").write_text(
        json.dumps({"index": {"count": [8]}}), encoding="utf-8"
    )
    return dataset


def test_inspection_reports_repairable_outputs(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    report = inspect_dataset(dataset)
    assert report["core_valid"] is True
    assert report["prompt"]["status"] == "missing"
    assert report["reference"]["status"] == "missing"
    assert report["counts"]["decoded_videos"] == 4


def test_missing_pose_is_ignored(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    report = DatasetProcessingPipeline(dataset).run(check_only=True)
    assert report["status"] == "needs_processing"
    assert "pose" not in report["before"]


def test_default_intermediate_output_is_outside_dataset(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    pipeline = DatasetProcessingPipeline(dataset)

    assert pipeline.output_root == processing_module.DEFAULT_OUTPUT_ROOT / dataset.name
    assert dataset not in pipeline.output_root.parents


def test_prompt_requires_multi_reference_ids(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    rows = [
        {
            "episode_index": episode_index,
            "prompt": "A robot manipulates an object in a bright laboratory.",
            "reference_ids": [f"ref-{episode_index}"],
        }
        for episode_index in range(2)
    ]
    (dataset / "meta/episodes_prompt.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = inspect_dataset(dataset, probe_videos=False)

    assert report["prompt"]["status"] == "complete"
