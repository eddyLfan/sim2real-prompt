from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sim2real_prompt_annotation.yoloe import (
    DetectionRequest,
    YOLOEDetector,
    YOLOEQuery,
    YOLOEUnavailableError,
    normalize_queries,
)


@dataclass
class _Boxes:
    xyxy: np.ndarray
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
        assert kwargs["agnostic_nms"] is False
        self.predict_batch_sizes.append(len(source))
        results = []
        for frame in source:
            height, width = frame.shape[:2]
            mask = np.zeros((1, height // 2, width // 2), dtype=np.float32)
            mask[:, 1:-1, 1:-1] = 1.0
            results.append(
                _Result(
                    boxes=_Boxes(
                        xyxy=np.asarray([[1, 2, width - 1, height - 2]], dtype=float),
                        conf=np.asarray([0.91], dtype=float),
                        cls=np.asarray([0], dtype=float),
                    ),
                    masks=_Masks(mask),
                )
            )
        return results


class _InnerTextModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def get_text_pe(
        self, classes: list[str], *, cache_clip_model: bool = False
    ) -> object:
        self.calls.append((classes.copy(), cache_clip_model))
        return tuple(f"inner:{name}" for name in classes)


class _OfficialStyleYOLOE(_FakeYOLOE):
    def __init__(self) -> None:
        super().__init__()
        self.model = _InnerTextModel()


def _frame(value: int = 0) -> np.ndarray:
    return np.full((20, 30, 3), value, dtype=np.uint8)


def test_normalize_queries_accepts_dtos_and_merges_duplicate_labels() -> None:
    queries = normalize_queries(
        [
            {"text": " red   cup ", "role": "secondary"},
            YOLOEQuery(query="Red Cup", role="primary", required=True),
            {"label": "basket", "role": "destination"},
        ]
    )

    assert [query.query for query in queries] == ["red cup", "basket"]
    assert queries[0].role == "primary"
    assert queries[0].primary is True
    assert queries[0].required is True


def test_detector_is_lazy_and_reuses_cached_text_embeddings() -> None:
    fake = _FakeYOLOE()
    factory_calls: list[str] = []

    def factory(path: str) -> _FakeYOLOE:
        factory_calls.append(path)
        return fake

    detector = YOLOEDetector(model_factory=factory)
    assert detector.loaded is False

    queries = [
        {"query": "red cup", "role": "primary", "required": True},
        {"query": "basket", "role": "destination"},
    ]
    first = detector.predict([_frame()], queries)
    second = detector.predict([_frame(1)], queries)

    assert factory_calls == ["yoloe-11s-seg.pt"]
    assert fake.text_calls == [["red cup", "basket"]]
    assert len(fake.set_calls) == 1
    assert fake.predict_batch_sizes == [1, 1]
    assert first[0][0].query == "red cup"
    assert first[0][0].required is True
    assert first[0][0].mask_polygon is not None
    assert first[0][0].bbox_xyxy == (2.0, 2.0, 28.0, 18.0)
    assert second[0][0].image_width == 30

    detector.predict([_frame()], [{"query": "green tray"}])
    detector.predict([_frame()], queries)
    assert len(fake.text_calls) == 2
    assert len(fake.set_calls) == 3
    assert detector.cached_query_signatures == (
        ("green tray",),
        ("red cup", "basket"),
    )


def test_embedding_cache_is_lru_bounded() -> None:
    fake = _FakeYOLOE()
    detector = YOLOEDetector(
        model_factory=lambda _: fake,
        embedding_cache_size=1,
    )

    detector.predict([_frame()], [{"query": "cup"}])
    detector.predict([_frame()], [{"query": "tray"}])
    detector.predict([_frame()], [{"query": "cup"}])

    assert fake.text_calls == [["cup"], ["tray"], ["cup"]]
    assert detector.cached_query_signatures == (("cup",),)


def test_detector_keeps_supported_ultralytics_text_encoder_resident() -> None:
    fake = _OfficialStyleYOLOE()
    detector = YOLOEDetector(model_factory=lambda _: fake)

    detector.predict([_frame()], [{"query": "cup"}])
    detector.predict([_frame()], [{"query": "tray"}])

    assert fake.model.calls == [(["cup"], True), (["tray"], True)]
    assert fake.text_calls == []


def test_readiness_prewarms_text_prompting_only_once() -> None:
    fake = _FakeYOLOE()
    detector = YOLOEDetector(device="cpu", model_factory=lambda _: fake)

    detector.ensure_ready()
    detector.ensure_ready()

    assert fake.text_calls == [["object"]]
    assert fake.active_classes == ["object"]


def test_readiness_rejects_non_segmentation_checkpoint_before_prompting() -> None:
    fake = _FakeYOLOE()
    fake.task = "detect"
    detector = YOLOEDetector(device="cpu", model_factory=lambda _: fake)

    with pytest.raises(YOLOEUnavailableError, match="segmentation model"):
        detector.ensure_ready()

    assert fake.text_calls == []


def test_cache_identity_binds_ultralytics_runtime_version() -> None:
    detector = YOLOEDetector(device="cpu", model_factory=lambda _: _FakeYOLOE())

    assert "ultralytics_version" in detector.cache_identity()


def test_predict_requests_batches_frames_with_identical_query_signature() -> None:
    fake = _FakeYOLOE()
    detector = YOLOEDetector(model_factory=lambda _: fake)
    cup = [{"query": "cup", "primary": True}]
    tray = [{"query": "tray", "required": True}]

    outputs = detector.predict_requests(
        [
            DetectionRequest(_frame(1), cup, "one"),
            DetectionRequest(_frame(2), tray, "two"),
            DetectionRequest(_frame(3), cup, "three"),
        ]
    )

    assert [output[0].query for output in outputs] == ["cup", "tray", "cup"]
    assert fake.predict_batch_sizes == [2, 1]
    assert len(fake.text_calls) == 2


def test_predict_requests_separates_same_text_with_different_semantics() -> None:
    fake = _FakeYOLOE()
    detector = YOLOEDetector(model_factory=lambda _: fake)

    outputs = detector.predict_requests(
        [
            DetectionRequest(
                _frame(1),
                [{"query": "cup", "role": "primary", "required": True}],
                "primary",
            ),
            DetectionRequest(
                _frame(2),
                [{"query": "cup", "role": "secondary", "required": False}],
                "secondary",
            ),
        ]
    )

    assert fake.predict_batch_sizes == [1, 1]
    assert [output[0].role for output in outputs] == ["primary", "secondary"]
    assert [output[0].required for output in outputs] == [True, False]
    # Text embeddings are still shared because the detector vocabulary is identical.
    assert fake.text_calls == [["cup"]]


def test_detector_rejects_empty_queries_before_loading_model() -> None:
    factory_called = False

    def factory(_: str) -> _FakeYOLOE:
        nonlocal factory_called
        factory_called = True
        return _FakeYOLOE()

    detector = YOLOEDetector(model_factory=factory)
    with pytest.raises(ValueError, match="at least one"):
        detector.predict([_frame()], [])
    assert factory_called is False


def test_original_image_polygon_wins_over_letterboxed_mask_tensor() -> None:
    frame = _frame()
    result = _Result(
        boxes=_Boxes(
            xyxy=np.asarray([[0, 0, 30, 20]], dtype=float),
            conf=np.asarray([0.9], dtype=float),
            cls=np.asarray([0], dtype=float),
        ),
        masks=_Masks(np.ones((1, 8, 8), dtype=np.float32)),
    )
    result.masks.xy = [
        np.asarray([[5, 6], [10, 6], [10, 12], [5, 12]], dtype=np.float32)
    ]

    detections = YOLOEDetector._parse_result(
        result,
        frame=frame,
        queries=(YOLOEQuery(query="cup", role="primary", required=True),),
    )

    assert detections[0].bbox_xyxy == (5.0, 6.0, 11.0, 13.0)
