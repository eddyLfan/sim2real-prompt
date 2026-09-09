from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import pytest

from sim2real_prompt_annotation.config import ReferenceConfig
from sim2real_prompt_annotation.models import Detection, ReferenceQuery
from sim2real_prompt_annotation.reference_branch import (
    NoValidReferenceError,
    ReferenceBranch,
    ReferenceBranchInput,
)

WIDTH = 200
HEIGHT = 100


def _frame() -> np.ndarray:
    y, x = np.indices((HEIGHT, WIDTH))
    return np.stack((x % 251, y % 251, (x + y) % 251), axis=-1).astype(np.uint8)


def _detection(
    query: str,
    box: tuple[int, int, int, int],
    *,
    role: str = "secondary",
    required: bool = False,
    confidence: float = 0.9,
    mask: bool = True,
) -> Detection:
    x1, y1, x2, y2 = box
    polygon = (
        (
            (float(x1), float(y1)),
            (float(x2), float(y1)),
            (float(x2), float(y2)),
            (float(x1), float(y2)),
        )
        if mask
        else None
    )
    return Detection(
        query=query,
        role=role,
        required=required,
        confidence=confidence,
        bbox_xyxy=tuple(float(value) for value in box),
        image_width=WIDTH,
        image_height=HEIGHT,
        mask_polygon=polygon,
    )


class _FakeDetector:
    def __init__(self, detections: Sequence[Detection]) -> None:
        self.detections = list(detections)
        self.predict_calls = 0
        self.batch_calls = 0

    def predict(
        self, frames: Sequence[np.ndarray], queries: Sequence[object]
    ) -> list[list[Detection]]:
        self.predict_calls += 1
        return [self.detections.copy() for _ in frames]

    def predict_requests(self, requests: Sequence[object]) -> list[list[Detection]]:
        self.batch_calls += 1
        return [self.detections.copy() for _ in requests]


def _config(**changes: object) -> ReferenceConfig:
    values: dict[str, object] = {
        "confidence": 0.25,
        "crop_padding": 0.10,
        "candidate_pool_size": 8,
        "min_images": 1,
        "max_images": 3,
        "selection_seed": 42,
        "jpeg_quality": 95,
    }
    values.update(changes)
    return ReferenceConfig.model_validate(values)


def test_builds_rectangular_jpeg_and_schema_ready_provenance() -> None:
    detections = [
        _detection("red cup", (20, 20, 60, 60), role="primary", required=True),
        _detection("basket", (120, 20, 170, 70), role="destination"),
    ]
    branch = ReferenceBranch(
        _FakeDetector(detections), _config(max_images=1), duplicate_iou=0.7
    )

    result = branch.process(
        sample_id="dataset:episode_0",
        episode_index=0,
        frame0=_frame(),
        queries=[
            ReferenceQuery(query="red cup", role="primary", required=True),
            ReferenceQuery(query="basket", role="destination"),
        ],
    )

    assert len(result.candidate_pool) == 2  # not truncated to selected 1
    assert len(result.selected_artifacts) == 1
    artifact = result.selected_artifacts[0]
    assert artifact.query == "red cup"
    assert artifact.relative_path.as_posix() == (
        "Reference/episode_000000/reference_00.jpg"
    )
    assert artifact.crop_xyxy == (16, 16, 64, 64)
    assert artifact.provenance["backend"] == "yoloe"
    assert artifact.provenance["required"] is True
    assert artifact.reference_id == f"sha256:{artifact.sha256}"
    decoded = cv2.imdecode(np.frombuffer(artifact.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (48, 48)
    assert artifact.row["source_frame_index"] == 0


def test_keeps_full_pool_and_selection_is_seeded_with_priority_coverage() -> None:
    detections = [
        _detection("main object", (5, 5, 30, 30), role="primary", required=True),
        _detection("target bin", (35, 5, 60, 30), role="destination", required=True),
        _detection("blue block", (65, 5, 90, 30)),
        _detection("green block", (95, 5, 120, 30)),
        _detection("robot arm", (125, 5, 150, 30), role="robot"),
        _detection("work table", (155, 5, 190, 30), role="workspace"),
    ]
    queries = [
        {
            "query": detection.query,
            "role": detection.role,
            "required": detection.required,
        }
        for detection in detections
    ]
    branch = ReferenceBranch(_FakeDetector(detections), _config(selection_seed=7))

    first = branch.process(
        sample_id="stable-sample",
        episode_index=12,
        frame0=_frame(),
        queries=queries,
    )
    second = branch.process(
        sample_id="stable-sample",
        episode_index=12,
        frame0=_frame(),
        queries=queries,
    )

    assert len(first.candidate_pool) == 6
    assert 2 <= len(first.selected_artifacts) <= 3
    selected_queries = {artifact.query for artifact in first.selected_artifacts}
    assert {"main object", "target bin"} <= selected_queries
    assert [item.reference_id for item in first.selected_artifacts] == [
        item.reference_id for item in second.selected_artifacts
    ]
    assert first.input_fingerprint == second.input_fingerprint


def test_overflowing_important_candidates_are_seeded_dropped_with_primary_first() -> (
    None
):
    detections = [
        _detection(
            f"primary {name}",
            (2 + index * 24, 5, 20 + index * 24, 35),
            role="primary",
            required=True,
        )
        for index, name in enumerate(("a", "b", "c", "d"))
    ]
    detections.extend(
        _detection(
            f"required {name}",
            (2 + index * 24, 55, 20 + index * 24, 85),
            role="destination",
            required=True,
        )
        for index, name in enumerate(("a", "b", "c", "d"))
    )
    queries = [
        {
            "query": detection.query,
            "role": detection.role,
            "required": detection.required,
        }
        for detection in detections
    ]
    branch = ReferenceBranch(
        _FakeDetector(detections),
        _config(min_images=3, max_images=3, selection_seed=19),
    )

    selections: list[tuple[str, ...]] = []
    for index in range(8):
        result = branch.process(
            sample_id=f"overflow-{index}",
            episode_index=index,
            frame0=_frame(),
            queries=queries,
        )
        selected = tuple(artifact.query for artifact in result.selected_artifacts)
        assert len(selected) == 3
        assert all(query.startswith("primary ") for query in selected)
        selections.append(selected)

    repeated = branch.process(
        sample_id="overflow-0",
        episode_index=0,
        frame0=_frame(),
        queries=queries,
    )
    assert tuple(item.query for item in repeated.selected_artifacts) == selections[0]
    assert len(set(selections)) > 1
    assert any(
        set(selection) != {"primary a", "primary b", "primary c"}
        for selection in selections
    )


def test_confidence_area_mask_and_iou_filters_are_auditable() -> None:
    kept = _detection("cup", (20, 20, 60, 60), role="primary", required=True)
    detections = [
        kept,
        _detection("cup", (21, 21, 61, 61), confidence=0.89),
        _detection("low confidence", (80, 10, 110, 40), confidence=0.1),
        _detection("no mask", (115, 10, 145, 40), mask=False),
        _detection("too small", (150, 10, 151, 11)),
        _detection("too large", (0, 0, WIDTH, HEIGHT)),
    ]
    branch = ReferenceBranch(
        _FakeDetector(detections),
        _config(max_images=1),
        min_area_fraction=0.001,
        max_area_fraction=0.90,
        duplicate_iou=0.5,
    )

    result = branch.process(
        sample_id="filter-test",
        episode_index=1,
        frame0=_frame(),
        queries=[{"query": item.query, "role": item.role} for item in detections],
    )

    assert result.candidate_pool == (kept,)
    reasons = " | ".join(result.rejected_reasons)
    assert "duplicate overlap" in reasons
    assert "confidence below threshold" in reasons
    assert "segmentation mask is missing" in reasons
    assert "segmented area is too small" in reasons
    assert "segmented area is too large" in reasons


def test_overlap_dedup_is_query_local_and_pool_covers_required_queries() -> None:
    detections = [
        _detection(
            "cup", (20, 20, 60, 60), role="primary", required=True, confidence=0.99
        ),
        _detection(
            "cup", (80, 20, 120, 60), role="primary", required=True, confidence=0.98
        ),
        _detection(
            "box",
            (21, 21, 61, 61),
            role="destination",
            required=True,
            confidence=0.70,
        ),
    ]
    branch = ReferenceBranch(
        _FakeDetector(detections),
        _config(candidate_pool_size=2, min_images=2, max_images=2),
    )

    result = branch.process(
        sample_id="query-local-overlap",
        episode_index=4,
        frame0=_frame(),
        queries=[
            {"query": "cup", "role": "primary", "required": True},
            {"query": "box", "role": "destination", "required": True},
        ],
    )

    assert [item.query for item in result.candidate_pool] == ["cup", "box"]


def test_no_detection_raises_without_whole_frame_fallback() -> None:
    branch = ReferenceBranch(_FakeDetector([]), _config())

    with pytest.raises(NoValidReferenceError, match="no valid YOLOE"):
        branch.process(
            sample_id="missing-object",
            episode_index=2,
            frame0=_frame(),
            queries=[{"query": "missing cup", "role": "primary"}],
        )


def test_non_primary_detections_cannot_publish_a_reference_set() -> None:
    detection = _detection(
        "target basket", (20, 20, 60, 60), role="destination", required=True
    )
    branch = ReferenceBranch(_FakeDetector([detection]), _config())

    with pytest.raises(NoValidReferenceError, match="no valid primary-object"):
        branch.process(
            sample_id="missing-primary",
            episode_index=3,
            frame0=_frame(),
            queries=[
                {
                    "query": detection.query,
                    "role": detection.role,
                    "required": True,
                }
            ],
        )


def test_batch_path_uses_detector_batch_api_and_accepts_jpeg_frame0() -> None:
    detection = _detection("cup", (20, 20, 60, 60), role="primary")
    detector = _FakeDetector([detection])
    branch = ReferenceBranch(detector, _config(max_images=1))
    success, encoded = cv2.imencode(".jpg", _frame())
    assert success

    results = branch.process_batch(
        [
            ReferenceBranchInput(
                sample_id="one",
                episode_index=1,
                frame0=encoded.tobytes(),
                queries=[{"query": "cup", "role": "primary"}],
            ),
            ReferenceBranchInput(
                sample_id="two",
                episode_index=2,
                frame0=encoded.tobytes(),
                queries=[{"query": "cup", "role": "primary"}],
            ),
        ]
    )

    assert detector.batch_calls == 1
    assert detector.predict_calls == 0
    assert [result.sample_id for result in results] == ["one", "two"]
