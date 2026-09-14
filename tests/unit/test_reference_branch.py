from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import sim2real_prompt_annotation.inpainting as inpainting_module
from sim2real_prompt_annotation.config import ReferenceConfig
from sim2real_prompt_annotation.inpainting import (
    BigLamaInpainter,
    InpaintingError,
    InpaintingUnavailableError,
)
from sim2real_prompt_annotation.models import (
    RobotMaskPrediction,
    SceneReferenceResult,
)
from sim2real_prompt_annotation.reference_branch import (
    NoValidSceneReferenceError,
    ReferenceBranch,
    ReferenceBranchInput,
    SceneReferenceRuntimeError,
)
from sim2real_prompt_annotation.robot_mask import mask_bbox
from sim2real_prompt_annotation.robotseg import (
    RobotSegSegmenter,
    RobotSegUnavailableError,
)
from sim2real_prompt_annotation.validation import validate_reference_result

WIDTH = 96
HEIGHT = 64


def _frame(value: int | None = None) -> np.ndarray:
    if value is not None:
        return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)
    y, x = np.indices((HEIGHT, WIDTH))
    return np.stack((x, y * 2, (x + y) % 255), axis=-1).astype(np.uint8)


def _prediction(
    mask: np.ndarray,
    *,
    backend: str = "robotseg",
    query: str = "robot",
) -> RobotMaskPrediction:
    bbox = mask_bbox(mask)
    assert bbox is not None
    return RobotMaskPrediction(
        backend=backend,
        query=query,
        confidence=0.9,
        bbox_xyxy=bbox,
        image_width=mask.shape[1],
        image_height=mask.shape[0],
        mask=mask,
    )


def _robot_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[16:48, 34:62] = 1
    return mask


class _FakeSegmenter:
    def __init__(self, masks: Sequence[np.ndarray] | None = None) -> None:
        self.masks = list(masks if masks is not None else [_robot_mask()])
        self.predict_calls = 0
        self.batch_sizes: list[int] = []
        self.ready_calls = 0

    def ensure_ready(self) -> None:
        self.ready_calls += 1

    def cache_identity(self) -> dict[str, object]:
        return {"backend": "fake-robotseg", "version": 1}

    def predict(
        self,
        frames: Sequence[np.ndarray],
        queries: Sequence[str] = ("robot",),
    ) -> list[list[RobotMaskPrediction]]:
        del queries
        self.predict_calls += 1
        self.batch_sizes.append(len(frames))
        return [[_prediction(mask) for mask in self.masks] for _frame_value in frames]


class _FakeInpainter:
    def __init__(self, *, fill: int = 240, unchanged: bool = False) -> None:
        self.fill = fill
        self.unchanged = unchanged
        self.batch_sizes: list[int] = []
        self.ready_calls = 0

    def ensure_ready(self) -> None:
        self.ready_calls += 1

    def cache_identity(self) -> dict[str, object]:
        return {"backend": "fake-lama", "version": 1}

    def inpaint(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        self.batch_sizes.append(len(images))
        output = []
        for image, mask in zip(images, masks, strict=True):
            generated = image.copy()
            if not self.unchanged:
                generated[mask.astype(bool)] = self.fill
            output.append(generated)
        return output


def _config(**updates: object) -> ReferenceConfig:
    values: dict[str, object] = {
        "device": "cpu",
        "inpainting_device": "cpu",
        "residual_check": False,
        "mask_close_kernel": 1,
        "mask_dilation_pixels": 0,
        "min_mask_area_fraction": 0.001,
        "max_mask_area_fraction": 0.8,
        "min_inpaint_change_fraction": 0.01,
        "jpeg_quality": 100,
    }
    values.update(updates)
    return ReferenceConfig.model_validate(values)


def test_builds_exactly_one_full_scene_frame_zero_reference() -> None:
    source = _frame()
    branch = ReferenceBranch(_FakeSegmenter(), _FakeInpainter(), _config())

    result = branch.process(
        sample_id="dataset:episode_0",
        episode_index=0,
        frame0=source,
        source_view="camera_head",
    )

    assert len(result.selected_artifacts) == 1
    artifact = result.artifact
    assert artifact.relative_path == Path("Reference/episode_000000/reference_00.jpg")
    assert artifact.source_frame_index == 0
    assert artifact.source_view == "camera_head"
    assert artifact.scope == "environment"
    assert artifact.reference_kind == "robot_removed_scene"
    assert artifact.reference_id == f"sha256:{artifact.sha256}"
    assert (
        artifact.source_frame_sha256
        == hashlib.sha256(source.tobytes(order="C")).hexdigest()
    )
    assert (artifact.width, artifact.height) == (WIDTH, HEIGHT)
    decoded = cv2.imdecode(np.frombuffer(artifact.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == source.shape
    assert artifact.provenance["operation"] == "robot_removal_inpainting"
    assert artifact.provenance["quality_control"]["outside_mask_unchanged"] is True
    assert artifact.provenance["residual_qa"] == {
        "enabled": False,
        "detector": None,
        "queries": ["robot", "robot arm", "robot gripper"],
        "area_fraction": 0.0,
        "threshold": 0.002,
        "pass": True,
    }
    assert validate_reference_result(result) is result


def test_removal_mask_png_uses_raw_binary_pixel_hash_semantics() -> None:
    result = ReferenceBranch(
        _FakeSegmenter(), _FakeInpainter(), _config(mask_dilation_pixels=2)
    ).process(
        sample_id="mask-sidecar",
        episode_index=3,
        frame0=_frame(),
    )

    decoded = cv2.imdecode(
        np.frombuffer(result.removal_mask_png, np.uint8), cv2.IMREAD_GRAYSCALE
    )
    assert decoded is not None
    binary = np.ascontiguousarray(decoded > 0, dtype=np.uint8)
    assert hashlib.sha256(binary.tobytes(order="C")).hexdigest() == (
        result.artifact.mask_sha256
    )
    assert np.count_nonzero(binary) > np.count_nonzero(_robot_mask())

    bad = np.zeros_like(binary)
    ok, encoded = cv2.imencode(".png", bad)
    assert ok
    payload = result.model_dump()
    payload["removal_mask_png"] = encoded.tobytes()
    with pytest.raises(ValueError, match="non-empty binary mask"):
        SceneReferenceResult.model_validate(payload)


def test_no_robot_mask_fails_without_using_yoloe_as_a_mask_fallback() -> None:
    primary = _FakeSegmenter(masks=[])
    residual = _FakeSegmenter()
    branch = ReferenceBranch(
        primary,
        _FakeInpainter(),
        _config(residual_check=True),
        residual_detector=residual,
    )

    with pytest.raises(NoValidSceneReferenceError, match="found no robot mask") as exc:
        branch.process(sample_id="no-mask", episode_index=0, frame0=_frame())

    assert exc.value.diagnostics["raw_masks"] == []
    assert primary.predict_calls == 1
    assert residual.predict_calls == 0


def test_mask_area_and_unchanged_inpainting_fail_closed() -> None:
    with pytest.raises(NoValidSceneReferenceError, match="outside configured bounds"):
        ReferenceBranch(
            _FakeSegmenter(),
            _FakeInpainter(),
            _config(max_mask_area_fraction=0.05),
        ).process(sample_id="too-large", episode_index=0, frame0=_frame())

    with pytest.raises(NoValidSceneReferenceError, match="changed too little"):
        ReferenceBranch(
            _FakeSegmenter(),
            _FakeInpainter(unchanged=True),
            _config(),
        ).process(sample_id="unchanged", episode_index=0, frame0=_frame())


def test_residual_yoloe_is_qc_only_and_rejects_visible_robot() -> None:
    residual = _FakeSegmenter()
    branch = ReferenceBranch(
        _FakeSegmenter(),
        _FakeInpainter(),
        _config(residual_check=True, max_residual_area_fraction=0.01),
        residual_detector=residual,
    )

    with pytest.raises(NoValidSceneReferenceError, match="exceeds the QC limit"):
        branch.process(sample_id="residual", episode_index=0, frame0=_frame())

    assert residual.predict_calls == 1


def test_residual_qa_records_detector_identity_and_validates_consistency() -> None:
    residual = _FakeSegmenter(masks=[])
    result = ReferenceBranch(
        _FakeSegmenter(),
        _FakeInpainter(),
        _config(residual_check=True, max_residual_area_fraction=0.01),
        residual_detector=residual,
    ).process(sample_id="residual-pass", episode_index=0, frame0=_frame())

    residual_qa = result.artifact.provenance["residual_qa"]
    assert residual_qa["enabled"] is True
    assert residual_qa["detector"] == {
        "backend": "fake-robotseg",
        "version": 1,
    }
    assert residual_qa["area_fraction"] == 0.0
    assert residual_qa["threshold"] == 0.01
    assert residual_qa["pass"] is True
    assert validate_reference_result(result) is result

    provenance = result.artifact.provenance.copy()
    provenance["residual_qa"] = {
        **residual_qa,
        "area_fraction": 0.005,
    }
    artifact = result.artifact.model_copy(update={"provenance": provenance})
    inconsistent = result.model_copy(update={"artifact": artifact})
    with pytest.raises(ValueError, match="differs from quality control"):
        validate_reference_result(inconsistent)

    invalid_identity = result.artifact.provenance.copy()
    invalid_identity["segmenter"] = "fake-robotseg"
    artifact = result.artifact.model_copy(update={"provenance": invalid_identity})
    with pytest.raises(ValueError, match="segmenter identity"):
        validate_reference_result(result.model_copy(update={"artifact": artifact}))

    non_finite = result.artifact.provenance.copy()
    non_finite["residual_qa"] = {**residual_qa, "threshold": float("inf")}
    non_finite["quality_control"] = {
        **non_finite["quality_control"],
        "max_residual_area_fraction": float("inf"),
    }
    artifact = result.artifact.model_copy(update={"provenance": non_finite})
    with pytest.raises(ValueError, match="metrics are invalid"):
        validate_reference_result(result.model_copy(update={"artifact": artifact}))


def test_process_batch_batches_segmentation_and_inpainting() -> None:
    segmenter = _FakeSegmenter()
    inpainter = _FakeInpainter()
    branch = ReferenceBranch(segmenter, inpainter, _config())

    results = branch.process_batch(
        [
            ReferenceBranchInput("one", 1, _frame(20), "head"),
            ReferenceBranchInput("two", 2, _frame(30), "head"),
        ]
    )

    assert [result.sample_id for result in results] == ["one", "two"]
    assert segmenter.batch_sizes == [2]
    assert inpainter.batch_sizes == [2]


def test_backend_failures_are_runtime_errors_not_deterministic_rejections() -> None:
    class CrashingSegmenter(_FakeSegmenter):
        def predict(self, *args: object, **kwargs: object):
            raise RuntimeError("CUDA out of memory")

    class CrashingInpainter(_FakeInpainter):
        def inpaint(self, *args: object, **kwargs: object):
            raise OSError("temporary checkpoint I/O failure")

    class CrashingResidual(_FakeSegmenter):
        def predict(self, *args: object, **kwargs: object):
            raise RuntimeError("temporary YOLO runtime crash")

    cases = (
        (
            ReferenceBranch(CrashingSegmenter(), _FakeInpainter(), _config()),
            "segmentation",
        ),
        (
            ReferenceBranch(_FakeSegmenter(), CrashingInpainter(), _config()),
            "inpainting",
        ),
        (
            ReferenceBranch(
                _FakeSegmenter(),
                _FakeInpainter(),
                _config(residual_check=True),
                residual_detector=CrashingResidual(),
            ),
            "residual_qc",
        ),
    )
    for branch, stage in cases:
        with pytest.raises(SceneReferenceRuntimeError) as exc:
            branch.process(sample_id=stage, episode_index=0, frame0=_frame())
        assert exc.value.diagnostics["stage"] == stage
        assert not isinstance(exc.value, NoValidSceneReferenceError)
        assert not isinstance(exc.value, ValueError)


def test_cache_fingerprint_tracks_frame_and_all_components() -> None:
    branch = ReferenceBranch(_FakeSegmenter(), _FakeInpainter(), _config())
    first = branch.process(sample_id="stable", episode_index=1, frame0=_frame(1))
    repeated = branch.process(sample_id="stable", episode_index=1, frame0=_frame(1))
    changed = branch.process(sample_id="stable", episode_index=1, frame0=_frame(2))

    assert first.input_fingerprint == repeated.input_fingerprint
    assert first.input_fingerprint != changed.input_fingerprint
    assert branch.cache_identity()["source"] == "real-frame-0"


class _FakeRobotSegPredictor:
    def __init__(self) -> None:
        self.saw_frame = False
        self.reset_calls = 0

    def init_state(self, *, video_path: str, **_: object) -> dict[str, object]:
        self.saw_frame = (Path(video_path) / "00000.jpg").is_file()
        return {"video_path": video_path}

    def add_new_robot(self, **_: object) -> tuple[int, list[int], np.ndarray]:
        logits = np.full((1, 1, HEIGHT, WIDTH), -2.0, dtype=np.float32)
        logits[:, :, 10:30, 20:50] = 2.0
        return 0, [0], logits

    def reset_state(self, _: object) -> None:
        self.reset_calls += 1


def test_robotseg_official_adapter_is_lazy_and_uses_jpeg_directory() -> None:
    predictor = _FakeRobotSegPredictor()
    factory_calls: list[tuple[str, str, str]] = []

    def factory(config: str, checkpoint: str, device: str) -> Any:
        factory_calls.append((config, checkpoint, device))
        return predictor

    segmenter = RobotSegSegmenter(
        "not-needed-by-fake.pt",
        device="cpu",
        predictor_factory=factory,
    )

    outputs = segmenter.predict([_frame()], ["robot"])

    assert factory_calls == [("configs/robotseg-infer", "not-needed-by-fake.pt", "cpu")]
    assert predictor.saw_frame is True
    assert predictor.reset_calls == 1
    assert outputs[0][0].mask.shape == (HEIGHT, WIDTH)
    assert outputs[0][0].bbox_xyxy == (20, 10, 50, 30)


def test_robotseg_cache_identity_tracks_actual_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "robotseg.pt"
    checkpoint.write_bytes(b"first")
    segmenter = RobotSegSegmenter(
        checkpoint,
        device="cpu",
        predictor_factory=lambda *_: _FakeRobotSegPredictor(),
    )

    first = segmenter.cache_identity()["model"]
    assert isinstance(first, dict)
    assert first["file"]["sha256"] == hashlib.sha256(b"first").hexdigest()

    checkpoint.write_bytes(b"second-version")
    second = segmenter.cache_identity()["model"]
    assert isinstance(second, dict)
    assert second["file"]["sha256"] == hashlib.sha256(b"second-version").hexdigest()


def test_missing_production_checkpoints_fail_with_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(RobotSegUnavailableError, match="exact local file"):
        RobotSegSegmenter(tmp_path / "missing.pt", device="cpu").ensure_ready()
    with pytest.raises(InpaintingUnavailableError, match="official local model"):
        BigLamaInpainter(tmp_path / "missing-lama", device="cpu").ensure_ready()


def test_big_lama_cache_identity_tracks_official_config_contents(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "big-lama"
    (model_directory / "models").mkdir(parents=True)
    config_path = model_directory / "config.yaml"
    config_path.write_bytes(b"training_model:\n  kind: default\n")
    (model_directory / "models/best.ckpt").write_bytes(b"checkpoint")
    inpainter = BigLamaInpainter(
        model_directory,
        device="cpu",
        model_factory=lambda *_: lambda *_: None,
    )

    first = inpainter.cache_identity()
    first_config = first["config"]
    assert isinstance(first_config, dict)
    assert first_config["file"]["size"] == config_path.stat().st_size
    assert (
        first_config["file"]["sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )

    config_path.write_bytes(b"training_model:\n  kind: changed\n")
    second = inpainter.cache_identity()
    assert second["config"] != first_config


def test_big_lama_cache_identity_does_not_rehash_unchanged_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = tmp_path / "big-lama"
    (model_directory / "models").mkdir(parents=True)
    (model_directory / "config.yaml").write_text("training_model: {}\n")
    checkpoint = model_directory / "models/best.ckpt"
    checkpoint.write_bytes(b"checkpoint-v1")
    real_sha256_file = inpainting_module.sha256_file
    hashed_paths: list[Path] = []

    def tracked_sha256_file(path: Path) -> str:
        hashed_paths.append(Path(path).resolve())
        return real_sha256_file(path)

    monkeypatch.setattr(inpainting_module, "sha256_file", tracked_sha256_file)
    inpainter = BigLamaInpainter(
        model_directory,
        device="cpu",
        model_factory=lambda *_: lambda *_: None,
    )

    inpainter.cache_identity()
    inpainter.cache_identity()
    assert hashed_paths.count(checkpoint.resolve()) == 1

    checkpoint.write_bytes(b"checkpoint-v2-longer")
    identity = inpainter.cache_identity()
    assert hashed_paths.count(checkpoint.resolve()) == 2
    assert (
        identity["checkpoint"]["sha256"]
        == hashlib.sha256(b"checkpoint-v2-longer").hexdigest()
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_big_lama_rejects_non_finite_output_before_uint8_conversion(
    value: float,
) -> None:
    pytest.importorskip("torch")

    def model(image, mask):
        del mask
        output = image.clone()
        output[0, 0, 0, 0] = value
        return output

    inpainter = BigLamaInpainter(
        "fake-lama.pt",
        device="cpu",
        model_factory=lambda *_: model,
    )
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[1:4, 1:4] = 1

    with pytest.raises(InpaintingError, match="NaN or infinite"):
        inpainter.inpaint([_frame()], [mask])
