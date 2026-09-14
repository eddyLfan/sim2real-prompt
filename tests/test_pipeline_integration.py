from __future__ import annotations

import fcntl
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
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
from sim2real_prompt_annotation.export import ArtifactStore
from sim2real_prompt_annotation.models import (
    PromptPayload,
    RobotMaskDiagnostic,
    SceneReferenceArtifact,
    SceneReferenceResult,
)
from sim2real_prompt_annotation.pipeline import (
    _LEGACY_PROMPT_CACHE_SCHEMA,
    _LEGACY_PROMPT_SYSTEM_SHA256,
    _file_identity,
    _prompt_metadata,
    prompt_cache_key,
    reference_cache_key,
)
from sim2real_prompt_annotation.qwen import ResponseParseError, VLMClient, VLMResponse
from sim2real_prompt_annotation.reference_branch import (
    NoValidReferenceError,
    SceneReferenceRuntimeError,
)


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
            real_key: {"dtype": "video", "shape": [96, 128, 3]},
            f"{real_key}_sim": {"dtype": "video", "shape": [96, 128, 3]},
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
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.requests: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> VLMResponse:
        self.calls += 1
        self.requests.append(kwargs)
        assert len(kwargs["images"]) == 8
        assert kwargs["images"][0].frame_index == 0
        if self.fail:
            raise ValueError("fake Prompt failure")
        payload = PromptPayload(
            prompt=(
                "A dual-arm robot places a red cup into a storage box on a "
                "workbench under diffuse lighting."
            )
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


class _TruncatedVLM(_VLM):
    def __init__(self) -> None:
        super().__init__()
        self.budgets: list[int] = []

    def generate(self, **kwargs: Any) -> VLMResponse:
        self.budgets.append(kwargs["max_tokens"])
        if len(self.budgets) == 1:
            raise ResponseParseError(
                "prompt returned truncated JSON",
                '{"prompt":"cut off',
                truncated=True,
            )
        return super().generate(**kwargs)


def _scene_result(request: Any) -> SceneReferenceResult:
    frame = request.frame0
    assert isinstance(frame, np.ndarray)
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[8 : height // 2, 10 : width // 3] = 1
    ys, xs = np.nonzero(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    mask_sha256 = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
    cleaned = frame.copy()
    cleaned[mask > 0] = (31, 63, 95)
    ok, encoded = cv2.imencode(".jpg", cleaned)
    assert ok
    jpeg = encoded.tobytes()
    digest = hashlib.sha256(jpeg).hexdigest()
    ok, mask_encoded = cv2.imencode(".png", mask * 255)
    assert ok
    area = float(np.count_nonzero(mask)) / float(mask.size)
    artifact = SceneReferenceArtifact(
        sample_id=request.sample_id,
        reference_id=f"sha256:{digest}",
        relative_path=(
            Path("Reference")
            / f"episode_{request.episode_index:06d}"
            / "reference_00.jpg"
        ),
        jpeg=jpeg,
        source_view=request.source_view,
        source_frame_index=0,
        scope="environment",
        reference_kind="robot_removed_scene",
        width=width,
        height=height,
        source_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
        mask_sha256=mask_sha256,
        mask_area_fraction=area,
        sha256=digest,
        provenance={
            "operation": "robot_removal_inpainting",
            "segmenter": {"backend": "fake-robot-segmenter"},
            "final_mask": {"sha256": mask_sha256, "area_fraction": area},
            "inpainter": {"backend": "fake-inpainter"},
            "quality_control": {
                "outside_mask_unchanged": True,
                "residual_check_enabled": False,
                "residual_mask_area_fraction": 0.0,
                "max_residual_area_fraction": 0.002,
            },
            "residual_qa": {
                "enabled": False,
                "detector": None,
                "queries": ["robot", "robot arm", "robot gripper"],
                "area_fraction": 0.0,
                "threshold": 0.002,
                "pass": True,
            },
        },
    )
    return SceneReferenceResult(
        sample_id=request.sample_id,
        artifact=artifact,
        robot_masks=(
            RobotMaskDiagnostic(
                backend="fake-robot-segmenter",
                query="robot",
                confidence=0.99,
                bbox_xyxy=bbox,
                mask_sha256=mask_sha256,
                mask_area_fraction=area,
            ),
        ),
        removal_mask_png=mask_encoded.tobytes(),
        input_fingerprint=f"sha256:{hashlib.sha256(frame.tobytes()).hexdigest()}",
    )


class _ReferenceBranch:
    def __init__(
        self,
        *,
        fail: bool = False,
        runtime_fail: bool = False,
        unavailable: bool = False,
    ) -> None:
        self.calls = 0
        self.ready_calls = 0
        self.fail = fail
        self.runtime_fail = runtime_fail
        self.unavailable = unavailable

    def cache_identity(self) -> dict[str, object]:
        # Failure mode is deliberately not semantic, allowing resume tests to swap
        # runtime availability without changing the cache key.
        return {"backend": "fake-scene-reference", "version": 1}

    def ensure_ready(self) -> None:
        self.ready_calls += 1
        if self.unavailable:
            raise RuntimeError("missing robot-removal weights")

    def process_batch(self, requests: list[Any]) -> list[SceneReferenceResult]:
        self.calls += 1
        if self.fail:
            raise NoValidReferenceError(
                "no valid robot-removed scene",
                diagnostics={
                    "raw_masks": [],
                    "final_mask": None,
                    "reason": "fake deterministic failure",
                },
            )
        if self.runtime_fail:
            raise SceneReferenceRuntimeError(
                "temporary CUDA runtime failure",
                diagnostics={"stage": "segmentation"},
            )
        for request in requests:
            assert request.source_view == "camera_head"
            assert request.frame0.shape == (96, 128, 3)
        return [_scene_result(request) for request in requests]


def _pipeline(
    dataset: Path,
    output: Path,
    vlm: VLMClient,
    reference: _ReferenceBranch,
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
    return Sim2RealPreprocessingPipeline(
        config,
        vlm_client=vlm,
        reference_branch=reference,  # type: ignore[arg-type]
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_source_file_identity_binds_metadata_change_signals(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")

    identity = _file_identity(path)

    assert identity == {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "ctime_ns": path.stat().st_ctime_ns,
        "device": path.stat().st_dev,
        "inode": path.stat().st_ino,
    }


def test_end_to_end_schema_v3_exact_one_and_resume(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    output = tmp_path / "checkpoints"
    vlm = _VLM()
    reference = _ReferenceBranch()

    first = _pipeline(dataset, output, vlm, reference).run()

    assert first["status"] == "complete"
    assert first["schema_version"] == 3
    assert first["selected_episodes"] == 2
    assert first["api_requests"] == 2
    assert first["reference_requests"] == 2
    assert first["audit"]["status"] == "complete"
    assert vlm.calls == 2
    assert reference.calls == 1

    prompt_rows = _jsonl(dataset / "meta/episodes_prompt.jsonl")
    reference_rows = _jsonl(dataset / "meta/reference_images.jsonl")
    assert len(prompt_rows) == len(reference_rows) == 2
    assert set(prompt_rows[0]) == {"episode_index", "prompt", "reference_ids"}
    assert reference_rows[0]["schema_version"] == 3
    assert len(reference_rows[0]["references"]) == 1
    item = reference_rows[0]["references"][0]
    assert item["source_frame_index"] == 0
    assert item["source_view"] == "camera_head"
    assert item["scope"] == "environment"
    assert item["reference_kind"] == "robot_removed_scene"
    assert (item["width"], item["height"]) == (128, 96)
    assert item["reference_path"] == "Reference/episode_000000/reference_00.jpg"
    assert prompt_rows[0]["reference_ids"] == [item["reference_id"]]
    assert (dataset / item["reference_path"]).is_file()
    assert len(list((output / "reference_images").glob("**/removal_mask.png"))) == 2

    episode_directory = dataset / "Reference/episode_000000"
    for position in (1, 2, 99):
        (episode_directory / f"reference_{position:02d}.jpg").write_bytes(b"stale")
    no_call_vlm = _VLM(fail=True)
    no_call_reference = _ReferenceBranch(unavailable=True)
    second = _pipeline(dataset, output, no_call_vlm, no_call_reference).run()

    assert second["status"] == "complete"
    assert second["prompt_cache_hits"] == 2
    assert second["reference_cache_hits"] == 2
    assert second["decoded_episodes"] == 0
    assert second["api_requests"] == 0
    assert second["reference_requests"] == 0
    assert no_call_vlm.calls == 0
    assert no_call_reference.ready_calls == 0
    assert sorted(path.name for path in episode_directory.glob("reference_*.jpg")) == [
        "reference_00.jpg"
    ]


def test_reference_preflight_failure_does_not_block_prompt(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    output = tmp_path / "checkpoints"
    vlm = _VLM()

    failed = _pipeline(dataset, output, vlm, _ReferenceBranch(unavailable=True)).run()

    assert failed["status"] == "partial"
    assert failed["published_episodes"] == 0
    assert failed["api_requests"] == 1
    assert vlm.calls == 1
    assert any(row["stage"] == "reference-preflight" for row in failed["failures"])
    assert len(list((output / "prompt").glob("*/*.json"))) == 1

    no_call_vlm = _VLM(fail=True)
    recovered_reference = _ReferenceBranch()
    recovered = _pipeline(dataset, output, no_call_vlm, recovered_reference).run()
    assert recovered["status"] == "complete"
    assert recovered["prompt_cache_hits"] == 1
    assert recovered["reference_only_decode_episodes"] == 1
    assert no_call_vlm.calls == 0
    assert recovered_reference.calls == 1


def test_prompt_failure_does_not_block_reference_or_its_cache(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    output = tmp_path / "checkpoints"
    reference = _ReferenceBranch()

    failed = _pipeline(dataset, output, _VLM(fail=True), reference).run()

    assert failed["status"] == "partial"
    assert failed["published_episodes"] == 0
    assert reference.calls == 1
    assert len(list((output / "reference").glob("*/*.json"))) == 1

    unavailable_reference = _ReferenceBranch(unavailable=True)
    recovered = _pipeline(dataset, output, _VLM(), unavailable_reference).run()
    assert recovered["status"] == "complete"
    assert recovered["reference_cache_hits"] == 1
    assert recovered["api_requests"] == 1
    assert unavailable_reference.ready_calls == 0


def test_legacy_prompt_checkpoint_extracts_current_fields(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    output = tmp_path / "checkpoints"
    config = PipelineConfig.model_validate(
        {"dataset": {"root": dataset}, "output": {"root": output}}
    )
    record = discover_episodes(config.dataset)[0]
    legacy_config = config.model_copy(deep=True)
    legacy_config.prompt.max_tokens = 1024
    provider = legacy_config.prompt.provider
    key = prompt_cache_key(
        record,
        legacy_config,
        branch_identity={
            "system_prompt_sha256": _LEGACY_PROMPT_SYSTEM_SHA256,
            "provider": {
                "name": provider.name,
                "model": provider.model,
                "endpoint": provider.base_url or os.getenv(provider.base_url_env),
                "response_format": provider.response_format,
                "enable_thinking": provider.enable_thinking,
            },
            "temperature": legacy_config.prompt.temperature,
            "max_tokens": legacy_config.prompt.max_tokens,
        },
        _schema=_LEGACY_PROMPT_CACHE_SCHEMA,
    )
    path = ArtifactStore(output).prompt_path(record.sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cache_key": key,
                "result": {
                    "sample_id": record.sample_id,
                    "prompt": "A robot moves a cup across a bright workbench.",
                    "reference_queries": [
                        {"query": "cup", "role": "primary", "required": True}
                    ],
                    "model": "legacy-vlm",
                    "request_id": None,
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "frame_indices": [0, 11, 23, 34, 46, 57, 69, 80],
                    "input_fingerprint": "legacy-compatible",
                },
            }
        ),
        encoding="utf-8",
    )

    no_call_vlm = _VLM(fail=True)
    report = _pipeline(dataset, output, no_call_vlm, _ReferenceBranch()).run()

    assert report["status"] == "complete"
    assert report["prompt_cache_hits"] == 1
    assert report["prompt_cache_migrations"] == 1
    assert no_call_vlm.calls == 0
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["cache_key"] == prompt_cache_key(record, config)
    assert "reference_queries" not in migrated["result"]


def test_schema_v2_reference_checkpoint_is_invalidated(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    output = tmp_path / "checkpoints"
    assert _pipeline(dataset, output, _VLM(), _ReferenceBranch()).run()["status"] == (
        "complete"
    )
    path = next((output / "reference").glob("*/*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    reference = _ReferenceBranch()
    report = _pipeline(dataset, output, _VLM(fail=True), reference).run()

    assert report["prompt_cache_hits"] == 1
    assert report["reference_cache_hits"] == 0
    assert reference.calls == 1
    assert report["status"] == "complete"


def test_branch_cache_keys_are_independent(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    config = PipelineConfig.model_validate({"dataset": {"root": dataset}})
    record = discover_episodes(config.dataset)[0]
    prompt_key = prompt_cache_key(record, config)
    reference_key = reference_cache_key(record, config)

    changed_task = record.model_copy(update={"task": "move another object"})
    assert prompt_cache_key(changed_task, config) != prompt_key
    assert reference_cache_key(changed_task, config) == reference_key

    changed_reference = config.model_copy(deep=True)
    changed_reference.reference.mask_dilation_pixels += 1
    assert prompt_cache_key(record, changed_reference) == prompt_key
    assert reference_cache_key(record, changed_reference) != reference_key

    changed_geometry = record.model_copy(update={"real_frame_width": 127})
    assert reference_cache_key(changed_geometry, config) != reference_key


def test_failed_force_reprocessing_removes_rows_and_all_references(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    output = tmp_path / "checkpoints"
    assert _pipeline(dataset, output, _VLM(), _ReferenceBranch()).run()["status"] == (
        "complete"
    )
    directory = dataset / "Reference/episode_000000"
    (directory / "reference_01.jpg").write_bytes(b"stale")

    report = _pipeline(dataset, output, _VLM(), _ReferenceBranch(fail=True)).run(
        force=True
    )

    assert report["status"] == "partial"
    assert _jsonl(dataset / "meta/episodes_prompt.jsonl") == []
    assert _jsonl(dataset / "meta/reference_images.jsonl") == []
    assert list(directory.glob("reference_*.jpg")) == []
    failure_path = next((output / "reference_failures").glob("*/*.json"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["schema_version"] == 3
    assert "raw_masks" in failure["failure"]["diagnostics"]

    no_call_reference = _ReferenceBranch(unavailable=True)
    resumed = _pipeline(dataset, output, _VLM(fail=True), no_call_reference).run()
    assert resumed["status"] == "partial"
    assert resumed["reference_failure_cache_hits"] == 1
    assert resumed["prompt_cache_hits"] == 1
    assert no_call_reference.ready_calls == 0
    assert _jsonl(dataset / "meta/episodes_prompt.jsonl") == []


def test_reference_runtime_failure_is_retried_and_never_negative_cached(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    output = tmp_path / "checkpoints"

    failed = _pipeline(
        dataset,
        output,
        _VLM(),
        _ReferenceBranch(runtime_fail=True),
    ).run()

    assert failed["status"] == "partial"
    assert failed["reference_failure_cache_hits"] == 0
    assert not list((output / "reference_failures").glob("*/*.json"))

    recovered_reference = _ReferenceBranch()
    recovered = _pipeline(
        dataset,
        output,
        _VLM(fail=True),
        recovered_reference,
    ).run()
    assert recovered["status"] == "complete"
    assert recovered["reference_failure_cache_hits"] == 0
    assert recovered["prompt_cache_hits"] == 1
    assert recovered_reference.calls == 1


def test_inspect_and_audit_need_no_provider_or_models(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    pipeline = Sim2RealPreprocessingPipeline(
        {"dataset": {"root": dataset}, "output": {"root": tmp_path / "out"}}
    )

    inspection = pipeline.inspect()
    audit = pipeline.audit()

    assert inspection["episode_count"] == 1
    assert inspection["real_view"] == "camera_head"
    assert audit["status"] == "incomplete"
    assert audit["schema_version"] == 3


def test_transaction_marker_makes_dataset_not_ready(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    pipeline = _pipeline(dataset, tmp_path / "out", _VLM(), _ReferenceBranch())
    assert pipeline.run()["annotations_ready"] is True
    (dataset / "meta/.sim2real-prompt.transaction.json").write_text("{}")

    audit = pipeline.audit()

    assert audit["status"] == "incomplete"
    assert audit["annotations_ready"] is False
    assert any("interrupted publication" in error for error in audit["errors"])


def test_selection_audit_ignores_unselected_corrupt_rows(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=2)
    pipeline = _pipeline(dataset, tmp_path / "out", _VLM(), _ReferenceBranch())
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


def test_audit_rejects_incomplete_or_inconsistent_residual_qa(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    pipeline = _pipeline(dataset, tmp_path / "out", _VLM(), _ReferenceBranch())
    assert pipeline.run()["annotations_ready"] is True
    reference_path = dataset / "meta/reference_images.jsonl"
    original = _jsonl(reference_path)[0]

    missing_field = json.loads(json.dumps(original))
    del missing_field["references"][0]["provenance"]["residual_qa"]["detector"]
    reference_path.write_text(json.dumps(missing_field) + "\n", encoding="utf-8")
    incomplete = pipeline.audit()
    assert incomplete["status"] == "incomplete"
    assert any("residual_qa fields" in error for error in incomplete["errors"])

    inconsistent = json.loads(json.dumps(original))
    inconsistent["references"][0]["provenance"]["residual_qa"]["area_fraction"] = 0.001
    reference_path.write_text(json.dumps(inconsistent) + "\n", encoding="utf-8")
    mismatched = pipeline.audit()
    assert mismatched["status"] == "incomplete"
    assert any(
        "residual_qa differs from quality_control" in error
        for error in mismatched["errors"]
    )

    non_finite = json.loads(json.dumps(original))
    provenance = non_finite["references"][0]["provenance"]
    provenance["residual_qa"]["threshold"] = float("inf")
    provenance["quality_control"]["max_residual_area_fraction"] = float("inf")
    reference_path.write_text(json.dumps(non_finite) + "\n", encoding="utf-8")
    invalid_metrics = pipeline.audit()
    assert invalid_metrics["status"] == "incomplete"
    assert any(
        "residual_qa metrics are invalid" in error
        for error in invalid_metrics["errors"]
    )


def test_audit_holds_shared_lock_against_dataset_publication(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    pipeline = _pipeline(dataset, tmp_path / "out", _VLM(), _ReferenceBranch())
    assert pipeline.run()["annotations_ready"] is True
    lock_path = dataset / "meta/.sim2real-prompt.lock"

    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pipeline.audit)
            try:
                with pytest.raises(FutureTimeoutError):
                    future.result(timeout=0.1)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            report = future.result(timeout=5)

    assert report["status"] == "complete"


def test_openai_timeout_and_truncated_json_are_retried(tmp_path: Path) -> None:
    first_dataset = _write_dataset(tmp_path / "first", episodes=1)
    flaky = _FlakyVLM()
    timeout_report = _pipeline(
        first_dataset,
        tmp_path / "first-out",
        flaky,
        _ReferenceBranch(),
        retries=1,
    ).run()
    assert timeout_report["status"] == "complete"
    assert timeout_report["api_requests"] == 2
    assert flaky.attempts == 2

    second_dataset = _write_dataset(tmp_path / "second", episodes=1)
    truncated = _TruncatedVLM()
    truncated_report = _pipeline(
        second_dataset,
        tmp_path / "second-out",
        truncated,
        _ReferenceBranch(),
        retries=1,
    ).run()
    assert truncated_report["status"] == "complete"
    assert truncated_report["api_requests"] == 2
    assert truncated.budgets == [256, 1024]


def test_supplemental_metadata_and_empty_audit_fail_closed(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, episodes=1)
    config = PipelineConfig.model_validate({"dataset": {"root": dataset}})
    record = discover_episodes(config.dataset)[0]
    assert _prompt_metadata(
        record.model_copy(update={"metadata": {"fixture_color": "matte black"}})
    )["supplemental_metadata"] == {"fixture_color": "matte black"}

    pipeline = Sim2RealPreprocessingPipeline({"dataset": {"root": dataset}})
    with pytest.raises(ValueError, match="No episodes matched"):
        pipeline.audit(episodes="999")
