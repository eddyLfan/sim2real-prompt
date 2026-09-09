from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest
from openai import APITimeoutError

from sim2real_prompt_annotation import Sim2RealPreprocessingPipeline
from sim2real_prompt_annotation.config import PipelineConfig
from sim2real_prompt_annotation.dataset import discover_episodes
from sim2real_prompt_annotation.models import Detection, PromptPayload
from sim2real_prompt_annotation.pipeline import _prompt_metadata, prompt_cache_key
from sim2real_prompt_annotation.qwen import VLMClient, VLMResponse
from sim2real_prompt_annotation.reference_branch import ReferenceBranch


def _write_video(path: Path, *, episode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96)
    )
    assert writer.isOpened()
    yy, xx = np.indices((96, 128))
    for frame_index in range(81):
        frame = np.stack(
            (
                (xx + frame_index) % 256,
                (yy * 2 + episode * 31) % 256,
                (xx + yy + episode * 17) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        writer.write(frame)
    writer.release()


def _write_dataset(root: Path, *, episodes: int = 2) -> Path:
    dataset = root / "paired_demo"
    (dataset / "meta").mkdir(parents=True)
    real_key = "camera_observations.color_images.camera_head"
    info = {
        "source_id": "integration-source",
        "domain": "integration-lab",
        "robot_type": "dual_arm_robot",
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": f"0:{episodes}"},
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            real_key: {"dtype": "video"},
            f"{real_key}_sim": {"dtype": "video"},
        },
    }
    (dataset / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    rows = [
        {
            "episode_index": index,
            "length": 81,
            "tasks": ["place the red cup into the storage box"],
        }
        for index in range(episodes)
    ]
    (dataset / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    for index in range(episodes):
        _write_video(
            dataset / "videos/chunk-000" / real_key / f"episode_{index:06d}.mp4",
            episode=index,
        )
    return dataset


class _VLM(VLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> VLMResponse:
        self.calls += 1
        self.requests.append(kwargs)
        assert len(kwargs["images"]) == 8
        assert kwargs["images"][0].frame_index == 0
        payload = PromptPayload(
            prompt=(
                "A dual-arm robot places a red cup into a storage box on a "
                "workbench under diffuse lighting."
            ),
            reference_queries=[
                {"query": "red cup", "role": "primary", "required": True},
                {
                    "query": "storage box",
                    "role": "destination",
                    "required": True,
                },
            ],
        )
        return VLMResponse(
            payload=payload,
            raw_text=payload.model_dump_json(),
            model="fake-vlm",
            input_tokens=100,
            output_tokens=20,
        )


class _FlakyVLM(_VLM):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def generate(self, **kwargs: Any) -> VLMResponse:
        self.attempts += 1
        if self.attempts == 1:
            raise APITimeoutError(httpx.Request("POST", "https://vlm.invalid"))
        return super().generate(**kwargs)


class _Detector:
    def __init__(self, *, empty: bool = False) -> None:
        self.calls = 0
        self.empty = empty

    def predict_requests(self, requests: list[Any]) -> list[list[Detection]]:
        self.calls += 1
        output: list[list[Detection]] = []
        for request in requests:
            if self.empty:
                output.append([])
                continue
            height, width = request.frame.shape[:2]
            values: list[Detection] = []
            for position, query in enumerate(request.queries):
                x1, y1 = 8 + position * 40, 12 + position * 18
                x2, y2 = x1 + 24, y1 + 24
                values.append(
                    Detection(
                        query=query.query,
                        role=query.role,
                        required=query.required,
                        confidence=0.92 - position * 0.02,
                        bbox_xyxy=(x1, y1, x2, y2),
                        image_width=width,
                        image_height=height,
                        mask_polygon=(
                            (x1, y1),
                            (x2, y1),
                            (x2, y2),
                            (x1, y2),
                        ),
                    )
                )
            output.append(values)
        return output


class _UnavailableDetector(_Detector):
    def ensure_ready(self) -> None:
        raise RuntimeError("missing YOLOE weights")


class _DeterministicFailureDetector(_Detector):
    def predict(self, frames: list[np.ndarray], queries: list[Any]) -> list[Any]:
        del frames, queries
        self.calls += 1
        raise ValueError("invalid detector configuration")


def _pipeline(
    dataset: Path,
    output: Path,
    vlm: VLMClient,
    detector: _Detector,
    *,
    retries: int = 0,
) -> Sim2RealPreprocessingPipeline:
    config = PipelineConfig.model_validate(
        {
            "dataset": {"root": dataset},
            "output": {"root": output},
            "runtime": {
                "decode_workers": 2,
                "api_concurrency": 2,
                "api_retry_count": retries,
                "backoff_initial_seconds": 0,
            },
            "reference": {"batch_size": 2, "device": "cpu"},
        }
    )
    branch = ReferenceBranch(detector, config.reference)
    return Sim2RealPreprocessingPipeline(
        config, vlm_client=vlm, reference_branch=branch
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_end_to_end_publishes_and_resumes_both_branches(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    output = tmp_path / "checkpoints"
    vlm = _VLM()
    detector = _Detector()

    first = _pipeline(dataset, output, vlm, detector).run()

    assert first["status"] == "complete"
    assert first["selected_episodes"] == 2
    assert first["api_requests"] == 2
    assert first["detector_episodes"] == 2
    assert first["audit"]["status"] == "complete"
    assert vlm.calls == 2
    assert detector.calls == 1

    prompt_rows = _jsonl(dataset / "meta/episodes_prompt.jsonl")
    reference_rows = _jsonl(dataset / "meta/reference_images.jsonl")
    assert len(prompt_rows) == len(reference_rows) == 2
    assert set(prompt_rows[0]) == {"episode_index", "prompt", "reference_ids"}
    assert reference_rows[0]["schema_version"] == 2
    assert 1 <= len(reference_rows[0]["references"]) <= 3
    assert prompt_rows[0]["reference_ids"] == [
        item["reference_id"] for item in reference_rows[0]["references"]
    ]
    for position, item in enumerate(reference_rows[0]["references"]):
        assert item["source_frame_index"] == 0
        assert item["source_view"] == "camera_head"
        assert item["reference_path"] == (
            f"Reference/episode_000000/reference_{position:02d}.jpg"
        )
        assert (dataset / item["reference_path"]).is_file()

    no_call_vlm = _VLM()
    no_call_detector = _Detector()
    second = _pipeline(dataset, output, no_call_vlm, no_call_detector).run()

    assert second["status"] == "complete"
    assert second["prompt_cache_hits"] == 2
    assert second["reference_cache_hits"] == 2
    assert second["decoded_episodes"] == 0
    assert second["api_requests"] == 0
    assert second["detector_episodes"] == 0
    assert no_call_vlm.calls == 0
    assert no_call_detector.calls == 0


def test_no_yoloe_detection_fails_without_full_frame_fallback(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    report = _pipeline(
        dataset,
        tmp_path / "checkpoints",
        _VLM(),
        _Detector(empty=True),
    ).run()

    assert report["status"] == "partial"
    assert report["published_episodes"] == 0
    assert report["failures"][0]["stage"] == "reference"
    assert not (dataset / "Reference/episode_000000/reference_00.jpg").exists()


def test_inspect_and_audit_need_no_provider_or_detector(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    pipeline = Sim2RealPreprocessingPipeline(
        {"dataset": {"root": dataset}, "output": {"root": tmp_path / "out"}}
    )

    inspection = pipeline.inspect()
    audit = pipeline.audit()

    assert inspection["episode_count"] == 1
    assert inspection["real_view"] == "camera_head"
    assert audit["status"] == "incomplete"
    assert audit["error_count"] >= 2


def test_detector_preflight_fails_before_video_or_paid_api(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    vlm = _VLM()
    report = _pipeline(
        dataset,
        tmp_path / "checkpoints",
        vlm,
        _UnavailableDetector(),
    ).run()

    assert report["status"] == "partial"
    assert report["api_requests"] == 0
    assert report["decoded_episodes"] == 0
    assert report["failures"][0]["stage"] == "reference-preflight"
    assert vlm.calls == 0


def test_transaction_marker_makes_dataset_not_ready(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    pipeline = _pipeline(
        dataset,
        tmp_path / "checkpoints",
        _VLM(),
        _Detector(),
    )
    assert pipeline.run()["annotations_ready"] is True
    (dataset / "meta/.sim2real-prompt.transaction.json").write_text("{}")

    audit = pipeline.audit()

    assert audit["status"] == "incomplete"
    assert audit["annotations_ready"] is False
    assert any("interrupted publication" in error for error in audit["errors"])


def test_interrupted_subset_cannot_be_hidden_by_a_different_subset(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path, episodes=2)
    output = tmp_path / "checkpoints"
    pipeline = _pipeline(dataset, output, _VLM(), _Detector())
    assert pipeline.run()["annotations_ready"] is True
    sample_id = discover_episodes(pipeline.config.dataset)[0].sample_id
    marker = dataset / "meta/.sim2real-prompt.transaction.json"
    marker.write_text(
        json.dumps({"schema_version": 1, "sample_ids": [sample_id]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be fully reprocessed"):
        pipeline.run(episodes="1", audit=False)

    recovery = pipeline.run(episodes="0", audit=False)
    assert recovery["status"] == "complete"
    assert not marker.exists()


def test_selection_audit_ignores_unselected_corrupt_rows(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=2)
    pipeline = _pipeline(
        dataset,
        tmp_path / "checkpoints",
        _VLM(),
        _Detector(),
    )
    assert pipeline.run()["annotations_ready"] is True
    prompt_path = dataset / "meta/episodes_prompt.jsonl"
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8") + "{not-json}\n",
        encoding="utf-8",
    )

    selection = pipeline.audit(episodes="0")
    full = pipeline.audit()

    assert selection["status"] == "complete"
    assert selection["scope"] == "selection"
    assert selection["annotations_ready"] is None
    assert full["status"] == "incomplete"


def test_run_status_includes_post_publish_audit(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    (dataset / "meta/episodes_prompt.jsonl").write_text(
        json.dumps({"episode_index": 999}) + "\n",
        encoding="utf-8",
    )
    (dataset / "meta/reference_images.jsonl").write_text(
        json.dumps({"episode_index": 999}) + "\n",
        encoding="utf-8",
    )

    report = _pipeline(
        dataset,
        tmp_path / "checkpoints",
        _VLM(),
        _Detector(),
    ).run()

    assert report["status"] == "partial"
    assert report["published_episodes"] == 1
    assert report["audit"]["status"] == "incomplete"


def test_non_memory_detector_error_is_not_recursively_retried(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=2)
    detector = _DeterministicFailureDetector()

    report = _pipeline(
        dataset,
        tmp_path / "checkpoints",
        _VLM(),
        detector,
    ).run(audit=False)

    assert report["status"] == "partial"
    assert report["failed_episodes"] == 2
    assert detector.calls == 1


def test_supplemental_metadata_is_included_in_prompt_input(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    config = PipelineConfig.model_validate({"dataset": {"root": dataset}})
    record = discover_episodes(config.dataset)[0]

    assert _prompt_metadata(
        record.model_copy(update={"metadata": {"fixture_color": "matte black"}})
    )["supplemental_metadata"] == {"fixture_color": "matte black"}


def test_openai_timeout_is_retried_and_attempts_are_reported(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    vlm = _FlakyVLM()
    report = _pipeline(
        dataset,
        tmp_path / "checkpoints",
        vlm,
        _Detector(),
        retries=1,
    ).run()

    assert report["status"] == "complete"
    assert report["api_requests"] == 2
    assert vlm.attempts == 2


def test_prompt_cache_binds_episode_timeline_metadata(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    config = PipelineConfig.model_validate({"dataset": {"root": dataset}})
    record = discover_episodes(config.dataset)[0]
    baseline = prompt_cache_key(record, config)

    assert (
        prompt_cache_key(
            record.model_copy(update={"episode_length": record.episode_length + 1}),
            config,
        )
        != baseline
    )
    assert (
        prompt_cache_key(
            record.model_copy(update={"fps": record.fps + 1}),
            config,
        )
        != baseline
    )


def test_empty_audit_selection_fails_closed(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    pipeline = Sim2RealPreprocessingPipeline({"dataset": {"root": dataset}})

    with pytest.raises(ValueError, match="No episodes matched"):
        pipeline.audit(episodes="999")
