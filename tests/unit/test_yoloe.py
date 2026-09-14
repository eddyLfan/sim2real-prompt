from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pytest

from sim2real_prompt_annotation.yoloe import (
    DetectionRequest,
    YOLOEDetector,
    YOLOEUnavailableError,
    normalize_queries,
)


@dataclass
class _Boxes:
    conf: np.ndarray
    cls: np.ndarray


@dataclass
class _Masks:
    data: np.ndarray


@dataclass
class _Result:
    boxes: _Boxes
    masks: _Masks


class _FakeYOLOE:
    task = "segment"

    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.set_calls: list[tuple[list[str], object]] = []
        self.predict_batch_sizes: list[int] = []
        self.active_classes: list[str] = []

    def get_text_pe(self, classes: list[str]) -> object:
        self.text_calls.append(classes.copy())
        return tuple(f"embedding:{name}" for name in classes)

    def set_classes(self, classes: list[str], embeddings: object) -> None:
        self.active_classes = classes.copy()
        self.set_calls.append((classes.copy(), embeddings))

    def predict(self, source: list[np.ndarray], **kwargs: object) -> list[_Result]:
        assert kwargs["verbose"] is False
        assert kwargs["retina_masks"] is False
        self.predict_batch_sizes.append(len(source))
        results = []
        for frame in source:
            height, width = frame.shape[:2]
            mask = np.zeros((1, height // 2, width // 2), dtype=np.float32)
            mask[:, 2:-2, 2:-2] = 1.0
            results.append(
                _Result(
                    boxes=_Boxes(
                        conf=np.asarray([0.91], dtype=float),
                        cls=np.asarray([0], dtype=float),
                    ),
                    masks=_Masks(mask),
                )
            )
        return results


class _FakeTextModel:
    def tokenize(self, texts: list[str]) -> list[str]:
        return texts

    def encode_text(self, tokens: list[str]) -> list[str]:
        return tokens


class _FakeParameter:
    device = "cpu"


class _OfficialInnerModel:
    def __init__(self) -> None:
        self.clip_model: object | None = None
        self.text_calls: list[tuple[list[str], bool]] = []

    def parameters(self):
        return iter([_FakeParameter()])

    def get_text_pe(
        self, classes: list[str], *, cache_clip_model: bool = False
    ) -> object:
        assert self.clip_model is not None
        self.text_calls.append((classes.copy(), cache_clip_model))
        return tuple(f"local:{name}" for name in classes)


class _OfficialStyleYOLOE(_FakeYOLOE):
    def __init__(self) -> None:
        super().__init__()
        self.model = _OfficialInnerModel()


def _frame(value: int = 0) -> np.ndarray:
    return np.full((20, 30, 3), value, dtype=np.uint8)


def test_normalize_queries_accepts_strings_mappings_and_deduplicates() -> None:
    queries = normalize_queries([" robot ", {"query": "Robot"}, {"label": "robot arm"}])

    assert queries == ("robot", "robot arm")


def test_detector_is_lazy_and_returns_full_resolution_raster_masks() -> None:
    fake = _FakeYOLOE()
    factory_calls: list[str] = []

    def factory(path: str) -> _FakeYOLOE:
        factory_calls.append(path)
        return fake

    detector = YOLOEDetector(device="cpu", model_factory=factory)
    assert detector.loaded is False

    first = detector.predict([_frame()], ["robot", "robot arm"])
    second = detector.predict([_frame(1)], ["robot", "robot arm"])

    assert factory_calls == ["yoloe-11s-seg.pt"]
    assert fake.text_calls == [["robot", "robot arm"]]
    assert len(fake.set_calls) == 1
    assert fake.predict_batch_sizes == [1, 1]
    assert first[0][0].query == "robot"
    assert first[0][0].mask.shape == (20, 30)
    assert first[0][0].mask.dtype == np.uint8
    assert first[0][0].bbox_xyxy == (4, 4, 26, 16)
    assert second[0][0].image_width == 30


def test_embedding_cache_is_lru_bounded() -> None:
    fake = _FakeYOLOE()
    detector = YOLOEDetector(
        device="cpu",
        model_factory=lambda _: fake,
        embedding_cache_size=1,
    )

    detector.predict([_frame()], ["robot"])
    detector.predict([_frame()], ["robot arm"])
    detector.predict([_frame()], ["robot"])

    assert fake.text_calls == [["robot"], ["robot arm"], ["robot"]]
    assert detector.cached_query_signatures == (("robot",),)


def test_readiness_rejects_non_segmentation_checkpoint() -> None:
    fake = _FakeYOLOE()
    fake.task = "detect"
    detector = YOLOEDetector(device="cpu", model_factory=lambda _: fake)

    with pytest.raises(YOLOEUnavailableError, match="segmentation model"):
        detector.ensure_ready()

    assert fake.text_calls == []


def test_readiness_rejects_wrong_checkpoint_hash(tmp_path) -> None:
    checkpoint = tmp_path / "yoloe.pt"
    checkpoint.write_bytes(b"not-the-expected-checkpoint")
    fake = _FakeYOLOE()
    detector = YOLOEDetector(
        checkpoint,
        model_sha256="0" * 64,
        device="cpu",
        model_factory=lambda _: fake,
    )

    with pytest.raises(YOLOEUnavailableError, match="SHA-256 mismatch"):
        detector.ensure_ready()


def test_production_preflight_rejects_missing_mobileclip_without_download(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "yoloe.pt"
    checkpoint.write_bytes(b"placeholder")
    detector = YOLOEDetector(
        checkpoint,
        text_model_path=tmp_path / "missing-mobileclip.ts",
        device="cpu",
    )

    with pytest.raises(YOLOEUnavailableError, match="implicit downloads are disabled"):
        detector.ensure_ready()


def test_explicit_local_mobileclip_is_injected_and_bound_to_cache_identity(
    tmp_path,
) -> None:
    text_path = tmp_path / "mobileclip_blt.ts"
    text_path.write_bytes(b"local-mobileclip")
    expected_sha = hashlib.sha256(b"local-mobileclip").hexdigest()
    fake = _OfficialStyleYOLOE()
    factory_calls: list[tuple[str, object]] = []

    def text_factory(path: str, device: object) -> _FakeTextModel:
        factory_calls.append((path, device))
        return _FakeTextModel()

    detector = YOLOEDetector(
        device="cpu",
        text_model_path=text_path,
        text_model_sha256=expected_sha,
        model_factory=lambda _: fake,
        text_model_factory=text_factory,
    )

    detector.ensure_ready()

    assert factory_calls == [(str(text_path.resolve()), "cpu")]
    assert isinstance(fake.model.clip_model, _FakeTextModel)
    assert fake.model.text_calls == [(["robot"], True)]
    identity = detector.cache_identity()
    assert identity["text_model"]["file"]["sha256"] == expected_sha
    assert identity["configured_text_model_sha256"] == expected_sha
    assert identity["text_asset_policy"] == "explicit-local-only-v1"


def test_mobileclip_hash_mismatch_fails_before_text_model_factory(tmp_path) -> None:
    text_path = tmp_path / "mobileclip_blt.ts"
    text_path.write_bytes(b"wrong")
    calls = 0

    def text_factory(_: str, __: object) -> _FakeTextModel:
        nonlocal calls
        calls += 1
        return _FakeTextModel()

    detector = YOLOEDetector(
        device="cpu",
        text_model_path=text_path,
        text_model_sha256="0" * 64,
        model_factory=lambda _: _OfficialStyleYOLOE(),
        text_model_factory=text_factory,
    )

    with pytest.raises(YOLOEUnavailableError, match="MobileCLIPTS SHA-256 mismatch"):
        detector.ensure_ready()
    assert calls == 0


def test_predict_requests_batches_identical_residual_vocabularies() -> None:
    fake = _FakeYOLOE()
    detector = YOLOEDetector(device="cpu", model_factory=lambda _: fake)

    outputs = detector.predict_requests(
        [
            DetectionRequest(_frame(1), ["robot"], "one"),
            DetectionRequest(_frame(2), ["robot arm"], "two"),
            DetectionRequest(_frame(3), ["robot"], "three"),
        ]
    )

    assert [output[0].query for output in outputs] == [
        "robot",
        "robot arm",
        "robot",
    ]
    assert fake.predict_batch_sizes == [2, 1]


def test_detector_rejects_empty_queries_before_loading_model() -> None:
    factory_called = False

    def factory(_: str) -> _FakeYOLOE:
        nonlocal factory_called
        factory_called = True
        return _FakeYOLOE()

    detector = YOLOEDetector(device="cpu", model_factory=factory)
    with pytest.raises(ValueError, match="at least one"):
        detector.predict([_frame()], [])
    assert factory_called is False
