"""Thin, testable YOLOE-11s-seg inference adapter.

The adapter deliberately keeps ``ultralytics`` behind a lazy import.  Importing the
data-processing package therefore does not initialize CUDA (or require the optional
runtime dependency), and unit tests can inject a small fake model.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from .io_utils import sha256_file
from .models import Detection, ReferenceQuery


class YOLOEError(RuntimeError):
    """Base error raised by the YOLOE adapter."""


class YOLOEUnavailableError(YOLOEError):
    """Raised when the optional YOLOE runtime is not installed."""


@runtime_checkable
class YOLOEModelProtocol(Protocol):
    """Minimum Ultralytics YOLOE API used by :class:`YOLOEDetector`."""

    def get_text_pe(self, classes: list[str]) -> Any:
        """Encode a class vocabulary once."""

    def set_classes(self, classes: list[str], embeddings: Any) -> Any:
        """Activate a class vocabulary and its cached embeddings."""

    def predict(self, source: Sequence[np.ndarray], **kwargs: Any) -> Any:
        """Run batched segmentation inference."""


# Backwards-friendly names for callers that discuss the detector-specific concepts.
YOLOEQuery = ReferenceQuery
YOLOEDetection = Detection


@dataclass(frozen=True, slots=True)
class DetectionRequest:
    """One frame and the task-specific open-vocabulary queries applied to it."""

    frame: np.ndarray
    queries: Sequence[object]
    request_id: str = ""


def _read_value(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def normalize_queries(queries: Sequence[object]) -> tuple[ReferenceQuery, ...]:
    """Normalize VLM DTOs/mappings and merge duplicate textual classes.

    The VLM contract may call the textual field ``query``, ``text``, ``label``, or
    ``name``.  Supporting these aliases keeps this runtime adapter independent of a
    concrete serialization library while still rejecting malformed input early.
    """

    normalized: list[ReferenceQuery] = []
    positions: dict[str, int] = {}
    for raw in queries:
        text = str(_read_value(raw, "query", "text", "label", "name", default="") or "")
        text = " ".join(text.replace("\x00", " ").split()).strip(" ,.;:")
        if not text:
            raise ValueError("YOLOE reference query must contain non-whitespace text")
        role = str(_read_value(raw, "role", default="secondary") or "secondary")
        role = role.strip() or "secondary"
        required = bool(_read_value(raw, "required", default=False))
        primary = bool(_read_value(raw, "primary", "is_primary", default=False))
        if primary:
            role = "primary"

        key = text.casefold()
        existing_index = positions.get(key)
        if existing_index is None:
            positions[key] = len(normalized)
            normalized.append(
                ReferenceQuery(
                    query=text,
                    role=role,
                    required=required,
                )
            )
            continue

        existing = normalized[existing_index]
        use_new_role = primary and not existing.primary
        normalized[existing_index] = ReferenceQuery(
            query=existing.query,
            role=role if use_new_role else existing.role,
            required=existing.required or required,
        )

    if not normalized:
        raise ValueError("at least one YOLOE reference query is required")
    return tuple(normalized)


def query_signature(queries: Sequence[object]) -> tuple[str, ...]:
    """Return the order-sensitive vocabulary signature used by YOLOE."""

    return tuple(query.query.casefold() for query in normalize_queries(queries))


def request_signature(
    queries: Sequence[object],
) -> tuple[tuple[str, str, bool], ...]:
    """Include DTO semantics when grouping requests that share model outputs."""

    return tuple(
        (query.query.casefold(), query.role, query.required)
        for query in normalize_queries(queries)
    )


def _as_numpy(value: Any) -> np.ndarray:
    """Convert a Torch-like tensor without importing Torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YOLOEDetector:
    """Lazy, single-owner adapter for the YOLOE-11s segmentation model.

    YOLOE mutates active class state through ``set_classes``.  Loading, vocabulary
    activation, and prediction are consequently guarded by one re-entrant lock.  A
    signature cache avoids recomputing text embeddings when repeated tasks share the
    same object vocabulary.
    """

    def __init__(
        self,
        model_path: str | Path = "yoloe-11s-seg.pt",
        *,
        device: str = "cuda:0",
        image_size: int = 640,
        confidence: float = 0.15,
        iou_threshold: float = 0.50,
        embedding_cache_size: int = 64,
        model_factory: Callable[[str], YOLOEModelProtocol] | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if embedding_cache_size < 1:
            raise ValueError("embedding_cache_size must be positive")
        self.model_path = str(model_path)
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.embedding_cache_size = embedding_cache_size
        self._model_factory = model_factory
        self._model: YOLOEModelProtocol | None = None
        self._embedding_cache: OrderedDict[tuple[str, ...], Any] = OrderedDict()
        self._active_signature: tuple[str, ...] | None = None
        self._weight_identity: tuple[tuple[str, int, int], dict[str, Any]] | None = None
        self._text_runtime_ready = False
        self._lock = RLock()

    @property
    def loaded(self) -> bool:
        """Whether this detector has initialized its model."""

        return self._model is not None

    @property
    def cached_query_signatures(self) -> tuple[tuple[str, ...], ...]:
        """Expose immutable cache keys for diagnostics and tests."""

        with self._lock:
            return tuple(self._embedding_cache)

    def _load_model(self) -> YOLOEModelProtocol:
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            model = self._model_factory(self.model_path)
        else:
            try:
                from ultralytics import YOLOE  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - depends on optional runtime
                raise YOLOEUnavailableError(
                    "YOLOE runtime is unavailable; install the project's YOLOE "
                    "optional dependency and prepare yoloe-11s-seg.pt"
                ) from exc
            model = YOLOE(self.model_path)
        self._model = model
        return model

    def ensure_ready(self) -> None:
        """Fail before paid VLM work if weights, device, or text encoder are absent."""

        with self._lock:
            if self.device.startswith("cuda"):
                try:
                    import torch
                except ImportError as error:  # pragma: no cover - ultralytics needs it
                    raise YOLOEUnavailableError(
                        "CUDA YOLOE requires a working PyTorch installation"
                    ) from error
                if not torch.cuda.is_available():
                    raise YOLOEUnavailableError(
                        f"YOLOE device {self.device!r} requested but CUDA is "
                        "unavailable"
                    )
                _, _, index_text = self.device.partition(":")
                if index_text:
                    try:
                        index = int(index_text)
                    except ValueError as error:
                        raise YOLOEUnavailableError(
                            f"Invalid YOLOE CUDA device {self.device!r}"
                        ) from error
                    if index < 0 or index >= torch.cuda.device_count():
                        raise YOLOEUnavailableError(
                            f"YOLOE device {self.device!r} does not exist"
                        )
            model = self._load_model()
            task = getattr(model, "task", None)
            if task is None:
                task = getattr(getattr(model, "model", None), "task", None)
            if task is not None and str(task).casefold() != "segment":
                raise YOLOEUnavailableError(
                    f"YOLOE checkpoint must be a segmentation model; got task={task!r}"
                )
            if not self._text_runtime_ready:
                try:
                    self._activate(
                        model,
                        (
                            ReferenceQuery(
                                query="object",
                                role="primary",
                                required=True,
                            ),
                        ),
                    )
                except Exception as error:  # pragma: no cover - external runtime
                    raise YOLOEUnavailableError(
                        "YOLOE text prompting is unavailable; pre-download the "
                        "MobileCLIP text encoder and its tokenizer dependencies"
                    ) from error
                self._text_runtime_ready = True

    def cache_identity(self) -> dict[str, Any]:
        """Hash weights once; omit operational device and batching settings."""

        path = Path(self.model_path).expanduser()
        model: dict[str, Any] = {"locator": self.model_path}
        if path.is_file():
            resolved = path.resolve()
            stat = resolved.stat()
            key = (str(resolved), stat.st_size, stat.st_mtime_ns)
            if self._weight_identity is None or self._weight_identity[0] != key:
                self._weight_identity = (
                    key,
                    {
                        "path": str(resolved),
                        "size": stat.st_size,
                        "sha256": sha256_file(resolved),
                    },
                )
            model["file"] = self._weight_identity[1]
        try:
            ultralytics_version: str | None = version("ultralytics")
        except PackageNotFoundError:
            ultralytics_version = None
        return {
            "model": model,
            "ultralytics_version": ultralytics_version,
            "image_size": self.image_size,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold,
            "text_embedding_strategy": "persistent-encoder-v1",
        }

    @staticmethod
    def _get_text_embeddings(
        model: YOLOEModelProtocol,
        classes: list[str],
    ) -> Any:
        """Keep Ultralytics' large text encoder resident when its API supports it.

        The public ``YOLOE.get_text_pe`` API rebuilds the MobileCLIP wrapper for
        every new vocabulary. Current Ultralytics releases expose an explicit
        ``cache_clip_model`` switch on the underlying YOLOE model; feature-detect
        that optimization and retain the public API as the compatibility path.
        """

        inner_model = getattr(model, "model", None)
        inner_get_text_pe = getattr(inner_model, "get_text_pe", None)
        if callable(inner_get_text_pe):
            try:
                parameters = signature(inner_get_text_pe).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "cache_clip_model" in parameters:
                return inner_get_text_pe(classes, cache_clip_model=True)
        return model.get_text_pe(classes)

    def _activate(
        self,
        model: YOLOEModelProtocol,
        queries: tuple[ReferenceQuery, ...],
    ) -> None:
        signature = tuple(query.query.casefold() for query in queries)
        if signature == self._active_signature:
            return
        embeddings = self._embedding_cache.pop(signature, None)
        classes = [query.query for query in queries]
        if embeddings is None:
            embeddings = self._get_text_embeddings(model, classes)
        self._embedding_cache[signature] = embeddings
        while len(self._embedding_cache) > self.embedding_cache_size:
            self._embedding_cache.popitem(last=False)
        model.set_classes(classes, embeddings)
        self._active_signature = signature

    @staticmethod
    def _validate_frames(frames: Sequence[np.ndarray]) -> None:
        for index, frame in enumerate(frames):
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"frame {index} must be a numpy array")
            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                raise ValueError(f"frame {index} must have shape HxWx3 or HxWx4")
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                raise ValueError(f"frame {index} must be non-empty")

    @staticmethod
    def _mask_geometry(
        raw_mask: np.ndarray,
        *,
        width: int,
        height: int,
    ) -> (
        tuple[tuple[float, float, float, float], tuple[tuple[float, float], ...]] | None
    ):
        mask = np.asarray(raw_mask)
        if mask.ndim != 2:
            raise YOLOEError("each YOLOE segmentation mask must have shape [H, W]")
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        binary = np.asarray(mask) > 0.5
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        x1, y1, box_width, box_height = cv2.boundingRect(contour)
        x2 = x1 + box_width
        y2 = y1 + box_height
        epsilon = max(0.5, 0.002 * cv2.arcLength(contour, True))
        approximated = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approximated) < 3:
            polygon = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        else:
            polygon = tuple(
                (float(point[0]), float(point[1])) for point in approximated
            )
        return (float(x1), float(y1), float(x2), float(y2)), polygon

    @staticmethod
    def _polygon_geometry(
        raw_polygon: Any,
        *,
        width: int,
        height: int,
    ) -> (
        tuple[tuple[float, float, float, float], tuple[tuple[float, float], ...]] | None
    ):
        """Use Ultralytics' original-image polygon to avoid letterbox distortion."""

        polygon = _as_numpy(raw_polygon).astype(np.float32)
        if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3:
            return None
        if not np.isfinite(polygon).all():
            raise YOLOEError("YOLOE mask polygon contains non-finite coordinates")
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height)
        x1 = max(0, int(np.floor(polygon[:, 0].min())))
        y1 = max(0, int(np.floor(polygon[:, 1].min())))
        x2 = min(width, int(np.ceil(polygon[:, 0].max())) + 1)
        y2 = min(height, int(np.ceil(polygon[:, 1].max())) + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        contour = polygon.reshape(-1, 1, 2)
        epsilon = max(0.5, 0.002 * cv2.arcLength(contour, True))
        approximated = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approximated) < 3:
            approximated = np.asarray(
                ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
                dtype=np.float32,
            )
        points = tuple((float(x), float(y)) for x, y in approximated)
        return (float(x1), float(y1), float(x2), float(y2)), points

    @staticmethod
    def _parse_result(
        result: object,
        *,
        frame: np.ndarray,
        queries: tuple[ReferenceQuery, ...],
    ) -> list[Detection]:
        if isinstance(result, Sequence) and not isinstance(
            result, (str, bytes, np.ndarray)
        ):
            items = list(result)
            if all(isinstance(item, Detection) for item in items):
                return items

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = _as_numpy(getattr(boxes, "xyxy", np.empty((0, 4))))
        confidences = _as_numpy(getattr(boxes, "conf", np.empty((0,))))
        class_ids = _as_numpy(getattr(boxes, "cls", np.empty((0,))))
        if xyxy.ndim != 2 or xyxy.shape[1:] != (4,):
            raise YOLOEError("YOLOE boxes.xyxy must have shape [N, 4]")
        if len(confidences) != len(xyxy) or len(class_ids) != len(xyxy):
            raise YOLOEError("YOLOE boxes arrays have inconsistent lengths")

        mask_container = getattr(result, "masks", None)
        raw_polygons = getattr(mask_container, "xy", None)
        polygons = list(raw_polygons) if raw_polygons is not None else None
        # ``masks.xy`` is already mapped to the original image and is much smaller
        # than copying the complete [N,H,W] mask tensor from GPU to CPU.
        raw_masks = getattr(mask_container, "data", None) if polygons is None else None
        masks = _as_numpy(raw_masks) if raw_masks is not None else None
        if len(xyxy) and masks is None and polygons is None:
            raise YOLOEError(
                "YOLOE produced boxes without segmentation masks; "
                "use a yoloe-11s-seg checkpoint"
            )
        if masks is not None and len(masks) != len(xyxy):
            raise YOLOEError("YOLOE mask count does not match box count")
        if polygons is not None and len(polygons) != len(xyxy):
            raise YOLOEError("YOLOE mask polygon count does not match box count")

        height, width = frame.shape[:2]
        detections: list[Detection] = []
        for index, _raw_box in enumerate(xyxy):
            class_index = int(class_ids[index])
            if class_index < 0 or class_index >= len(queries):
                raise YOLOEError(
                    f"YOLOE class index {class_index} is outside active vocabulary"
                )
            query = queries[class_index]
            if polygons is not None:
                geometry = YOLOEDetector._polygon_geometry(
                    polygons[index], width=width, height=height
                )
            else:
                assert masks is not None
                geometry = YOLOEDetector._mask_geometry(
                    masks[index], width=width, height=height
                )
            if geometry is None:
                continue
            bbox_xyxy, mask_polygon = geometry
            detections.append(
                Detection(
                    query=query.query,
                    role=query.role,
                    required=query.required,
                    confidence=float(confidences[index]),
                    bbox_xyxy=bbox_xyxy,
                    mask_polygon=mask_polygon,
                    image_width=width,
                    image_height=height,
                )
            )
        return detections

    def predict(
        self,
        frames: Sequence[np.ndarray],
        queries: Sequence[object],
    ) -> list[list[Detection]]:
        """Run one batched prediction for frames sharing the same vocabulary."""

        frames = list(frames)
        if not frames:
            return []
        self._validate_frames(frames)
        normalized = normalize_queries(queries)
        with self._lock:
            model = self._load_model()
            self._activate(model, normalized)
            raw_results = list(
                model.predict(
                    source=frames,
                    device=self.device,
                    imgsz=self.image_size,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    # masks.xy still uses original-image coordinates. Avoiding
                    # retina masks keeps batched GPU/CPU memory and transfer cost low.
                    retina_masks=False,
                    # YOLOE defaults to class-agnostic NMS; references need to keep
                    # overlapping candidates belonging to different semantic queries.
                    agnostic_nms=False,
                    verbose=False,
                )
            )
            if len(raw_results) != len(frames):
                raise YOLOEError(
                    "YOLOE returned "
                    f"{len(raw_results)} results for {len(frames)} input frames"
                )
            return [
                self._parse_result(result, frame=frame, queries=normalized)
                for result, frame in zip(raw_results, frames, strict=True)
            ]

    def predict_requests(
        self, requests: Sequence[DetectionRequest]
    ) -> list[list[Detection]]:
        """Group requests by vocabulary so each group uses one GPU batch call."""

        requests = list(requests)
        if not requests:
            return []
        grouped: dict[tuple[tuple[str, str, bool], ...], list[int]] = defaultdict(list)
        for index, request in enumerate(requests):
            grouped[request_signature(request.queries)].append(index)

        outputs: list[list[Detection] | None] = [None] * len(requests)
        for indices in grouped.values():
            representative = requests[indices[0]]
            frames = [requests[index].frame for index in indices]
            detections = self.predict(frames, representative.queries)
            for index, result in zip(indices, detections, strict=True):
                outputs[index] = result
        return [output if output is not None else [] for output in outputs]
