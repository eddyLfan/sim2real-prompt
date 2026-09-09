from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sim2real_prompt_annotation.config import PromptConfig
from sim2real_prompt_annotation.models import EpisodeRecord
from sim2real_prompt_annotation.video import (
    decode_jpeg,
    decode_real_video,
    uniform_frame_indices,
)


def _write_video(path: Path, *, frame_count: int = 81) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((96, 128, 3), index * 3 % 256, dtype=np.uint8))
    writer.release()


def _record(path: Path) -> EpisodeRecord:
    return EpisodeRecord(
        sample_id="4:demo:0",
        source_id="demo",
        dataset_name="dataset",
        domain="lab",
        dataset_root=path.parent,
        episode_index=0,
        episode_length=81,
        split="train",
        fps=10,
        robot_type="dual_arm",
        task="move the cup",
        real_view="camera_head",
        real_video=path,
    )


def test_uniform_indices_include_both_endpoints() -> None:
    assert uniform_frame_indices(81) == (0, 11, 23, 34, 46, 57, 69, 80)


def test_real_video_is_opened_once_and_yields_shared_frame_zero(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "real.mp4"
    _write_video(path)
    native_capture = cv2.VideoCapture
    calls: list[str] = []

    def counting_capture(video_path: str):
        calls.append(video_path)
        return native_capture(video_path)

    monkeypatch.setattr(cv2, "VideoCapture", counting_capture)

    bundle = decode_real_video(
        _record(path),
        PromptConfig(resize_long_edge=64),
    )

    assert calls == [str(path)]
    assert bundle.frame_indices == uniform_frame_indices(81)
    assert len(bundle.prompt_frames) == 8
    assert (bundle.width, bundle.height) == (128, 96)
    assert bundle.first_frame_bgr.shape == (96, 128, 3)
    assert bundle.first_frame_bgr.dtype == np.uint8
    assert decode_jpeg(bundle.prompt_frames[0].jpeg).shape[:2] == (48, 64)
    assert bundle.prompt_frames[0].frame_index == 0


def test_video_frame_count_must_match_episode_metadata(tmp_path: Path) -> None:
    path = tmp_path / "real.mp4"
    _write_video(path)
    record = _record(path).model_copy(update={"episode_length": 82})

    with pytest.raises(ValueError, match="81 frames but episode metadata declares 82"):
        decode_real_video(record, PromptConfig())
