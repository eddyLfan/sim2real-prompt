"""Lazy YOLOE-11s-seg adapter for residual-robot quality control."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from inspect import signature as inspect_signature
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from .io_utils import sha256_file
from .models import RobotMaskPrediction
from .robot_mask import mask_bbox, normalize_robot_queries, validate_bgr_frame


class YOLOEError(RuntimeError):
    """Base error raised by the YOLOE adapter."""


class YOLOEUnavailableError(YOLOEError):
    """The optional YOLOE runtime/checkpoint cannot be used."""


@runtime_checkable
class YOLOEModelProtocol(Protocol):
    def get_text_pe(self, classes: list[str]) -> Any: ...

    def set_classes(self, classes: list[str], embeddings: Any) -> Any: ...

    def predict(self, source: Sequence[np.ndarray], **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class DetectionRequest:
    frame: np.ndarray
    queries: Sequence[object]
    request_id: str = ""


TextModelFactory = Callable[[str, Any], Any]


def _mobileclip_factory(model_path: str, device: Any) -> Any:
    """Load MobileCLIPTS from an already validated absolute local path."""

    try:
        from ultralytics.nn.text_model import MobileCLIPTS
    except (ImportError, ModuleNotFoundError) as error:
        raise YOLOEUnavailableError(
            "YOLOE MobileCLIPTS runtime is unavailable; install the Reference extra"
        ) from error
    return MobileCLIPTS(device=device, weight=model_path)


def normalize_queries(queries: Sequence[object]) -> tuple[str, ...]:
    return normalize_robot_queries(queries)


def query_signature(queries: Sequence[object]) -> tuple[str, ...]:
    return tuple(query.casefold() for query in normalize_queries(queries))


def request_signature(queries: Sequence[object]) -> tuple[str, ...]:
    return query_signature(queries)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YOLOEDetector:
    """Batched full-resolution robot masks from an open-vocabulary segmenter."""

    def __init__(
        self,
        model_path: str | Path = "yoloe-11s-seg.pt",
        *,
        model_sha256: str | None = None,
        text_model_path: str | Path = "weights/mobileclip_blt.ts",
        text_model_sha256: str | None = None,
        device: str = "cuda:0",
        image_size: int = 640,
        confidence: float = 0.05,
        iou_threshold: float = 0.50,
        embedding_cache_size: int = 16,
        model_factory: Callable[[str], YOLOEModelProtocol] | None = None,
        text_model_factory: TextModelFactory | None = None,
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
        self.model_sha256 = model_sha256
        self.text_model_path = str(text_model_path)
        self.text_model_sha256 = text_model_sha256
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.embedding_cache_size = embedding_cache_size
        self._model_factory = model_factory
        self._text_model_factory = text_model_factory
        self._model: YOLOEModelProtocol | None = None
        self._embedding_cache: OrderedDict[tuple[str, ...], Any] = OrderedDict()
        self._active_signature: tuple[str, ...] | None = None
        self._weight_identity: (
            tuple[tuple[str, int, int, int, int, int], dict[str, Any]] | None
        ) = None
        self._text_weight_identity: (
            tuple[tuple[str, int, int, int, int, int], dict[str, Any]] | None
        ) = None
        self._text_runtime_ready = False
        self._lock = RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def cached_query_signatures(self) -> tuple[tuple[str, ...], ...]:
        with self._lock:
            return tuple(self._embedding_cache)

    def _checkpoint_identity(self, path: Path, *, text_model: bool) -> dict[str, Any]:
        """Hash a local YOLOE asset once per concrete filesystem identity."""

        resolved = path.resolve()
        stat = resolved.stat()
        key = (
            str(resolved),
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_dev,
            stat.st_ino,
        )
        cached = self._text_weight_identity if text_model else self._weight_identity
        if cached is None or cached[0] != key:
            cached = (
                key,
                {
                    "path": str(resolved),
                    "size": stat.st_size,
                    "sha256": sha256_file(resolved),
                },
            )
            if text_model:
                self._text_weight_identity = cached
            else:
                self._weight_identity = cached
        return cached[1]

    def _load_model(self) -> YOLOEModelProtocol:
        if self._model is not None:
            return self._model
        path = Path(self.model_path).expanduser()
        text_model_path = self._validate_text_model_asset()
        if self._model_factory is None or self.model_sha256 is not None:
            if not path.is_file():
                raise YOLOEUnavailableError(
                    f"YOLOE checkpoint is not an exact local file: {path}"
                )
            if self.model_sha256 is not None:
                actual = self._checkpoint_identity(path, text_model=False)["sha256"]
                if actual != self.model_sha256:
                    raise YOLOEUnavailableError(
                        "YOLOE checkpoint SHA-256 mismatch: "
                        f"expected={self.model_sha256}, actual={actual}"
                    )
        if self._model_factory is not None:
            model = self._model_factory(self.model_path)
        else:
            try:
                from ultralytics import YOLOE  # type: ignore[import-not-found]
            except ImportError as error:  # pragma: no cover - optional runtime
                raise YOLOEUnavailableError(
                    "YOLOE runtime is unavailable; install the 'reference' extra"
                ) from error
            model = YOLOE(self.model_path)
        if text_model_path is not None:
            self._install_text_model(model, text_model_path)
        self._model = model
        return model

    def _validate_text_model_asset(self) -> Path | None:
        """Resolve the text encoder before Ultralytics can attempt a download."""

        required = (
            self._model_factory is None
            or self._text_model_factory is not None
            or self.text_model_sha256 is not None
        )
        if not required:
            return None
        path = Path(self.text_model_path).expanduser()
        if not path.is_file():
            raise YOLOEUnavailableError(
                "YOLOE MobileCLIPTS must be an exact local file; implicit downloads "
                f"are disabled: {path}"
            )
        resolved = path.resolve()
        if self.text_model_sha256 is not None:
            actual = self._checkpoint_identity(resolved, text_model=True)["sha256"]
            if actual != self.text_model_sha256:
                raise YOLOEUnavailableError(
                    "YOLOE MobileCLIPTS SHA-256 mismatch: "
                    f"expected={self.text_model_sha256}, actual={actual}"
                )
        return resolved

    def _install_text_model(
        self,
        model: YOLOEModelProtocol,
        path: Path,
    ) -> None:
        inner_model = getattr(model, "model", None)
        if inner_model is None or not callable(
            getattr(inner_model, "get_text_pe", None)
        ):
            raise YOLOEUnavailableError(
                "YOLOE model does not expose the inner get_text_pe API needed for "
                "local MobileCLIPTS injection"
            )
        try:
            parameter = next(inner_model.parameters())
            device = parameter.device
        except (AttributeError, StopIteration, TypeError):
            device = self.device
        factory = self._text_model_factory or _mobileclip_factory
        try:
            text_model = factory(str(path), device)
        except YOLOEUnavailableError:
            raise
        except Exception as error:  # noqa: BLE001 - optional runtime boundary
            raise YOLOEUnavailableError(
                f"Failed to load local YOLOE MobileCLIPTS {path}: {error}"
            ) from error
        if not callable(getattr(text_model, "tokenize", None)) or not callable(
            getattr(text_model, "encode_text", None)
        ):
            raise YOLOEUnavailableError(
                "YOLOE text model must expose tokenize and encode_text"
            )
        inner_model.clip_model = text_model

    def _validate_device(self) -> None:
        if not self.device.casefold().startswith("cuda"):
            return
        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional runtime
            raise YOLOEUnavailableError(
                "CUDA YOLOE requires a working PyTorch installation"
            ) from error
        if not torch.cuda.is_available():
            raise YOLOEUnavailableError(
                f"YOLOE device {self.device!r} requested but CUDA is unavailable"
            )
        _, _, index_text = self.device.partition(":")
        if index_text:
            try:
                index = int(index_text)
            except ValueError as error:
                raise YOLOEUnavailableError(
                    f"Invalid YOLOE CUDA device {self.device!r}"
                ) from error
            if not 0 <= index < torch.cuda.device_count():
                raise YOLOEUnavailableError(
                    f"YOLOE device {self.device!r} does not exist"
                )

    def ensure_ready(self) -> None:
        with self._lock:
            self._validate_device()
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
                    self._activate(model, ("robot",))
                except Exception as error:  # pragma: no cover - external runtime
                    raise YOLOEUnavailableError(
                        "YOLOE text prompting is unavailable; prepare MobileCLIP "
                        "and tokenizer dependencies"
                    ) from error
                self._text_runtime_ready = True

    def cache_identity(self) -> dict[str, Any]:
        path = Path(self.model_path).expanduser()
        model: dict[str, Any] = {"locator": self.model_path}
        if path.is_file():
            model["file"] = self._checkpoint_identity(path, text_model=False)
        text_path = Path(self.text_model_path).expanduser()
        text_model: dict[str, Any] = {"locator": self.text_model_path}
        if text_path.is_file():
            text_model["file"] = self._checkpoint_identity(text_path, text_model=True)
        try:
            runtime_version: str | None = version("ultralytics")
        except PackageNotFoundError:
            runtime_version = None
        return {
            "backend": "yoloe",
            "model": model,
            "text_model": text_model,
            "configured_text_model_sha256": self.text_model_sha256,
            "ultralytics_version": runtime_version,
            "image_size": self.image_size,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold,
            "mask_representation": "full-resolution-binary-v1",
            "text_embedding_strategy": "persistent-encoder-v1",
            "text_asset_policy": "explicit-local-only-v1",
        }

    @staticmethod
    def _get_text_embeddings(model: YOLOEModelProtocol, classes: list[str]) -> Any:
        inner_model = getattr(model, "model", None)
        inner_get_text_pe = getattr(inner_model, "get_text_pe", None)
        if callable(inner_get_text_pe):
            try:
                parameters = inspect_signature(inner_get_text_pe).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "cache_clip_model" in parameters:
                return inner_get_text_pe(classes, cache_clip_model=True)
        return model.get_text_pe(classes)

    def _activate(self, model: YOLOEModelProtocol, queries: tuple[str, ...]) -> None:
        signature = tuple(query.casefold() for query in queries)
        if signature == self._active_signature:
            return
        embeddings = self._embedding_cache.pop(signature, None)
        classes = list(queries)
        if embeddings is None:
            embeddings = self._get_text_embeddings(model, classes)
        self._embedding_cache[signature] = embeddings
        while len(self._embedding_cache) > self.embedding_cache_size:
            self._embedding_cache.popitem(last=False)
        model.set_classes(classes, embeddings)
        self._active_signature = signature

    @staticmethod
    def _rasterize_polygon(value: Any, *, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.asarray(value, dtype=np.float32)
        if polygon.ndim == 2 and polygon.shape[0] >= 3 and polygon.shape[1] == 2:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        return mask

    @staticmethod
    def _parse_result(
        result: Any,
        *,
        frame: np.ndarray,
        queries: tuple[str, ...],
    ) -> list[RobotMaskPrediction]:
        height, width = frame.shape[:2]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        confidences = _as_numpy(getattr(boxes, "conf", [])).reshape(-1)
        classes = _as_numpy(getattr(boxes, "cls", [])).reshape(-1)
        if len(confidences) != len(classes):
            raise YOLOEError("YOLOE boxes arrays have inconsistent lengths")
        masks_container = getattr(result, "masks", None)
        raw_masks = getattr(masks_container, "data", None)
        masks = _as_numpy(raw_masks) if raw_masks is not None else None
        polygons = getattr(masks_container, "xy", None)
        if len(classes) and masks is None and polygons is None:
            raise YOLOEError("YOLOE produced boxes without segmentation masks")
        if masks is not None and len(masks) != len(classes):
            raise YOLOEError("YOLOE mask count does not match box count")
        if polygons is not None and len(polygons) != len(classes):
            raise YOLOEError("YOLOE polygon count does not match box count")

        output: list[RobotMaskPrediction] = []
        for index, (confidence, class_value) in enumerate(
            zip(confidences, classes, strict=True)
        ):
            class_index = int(class_value)
            if not 0 <= class_index < len(queries):
                raise YOLOEError(
                    f"YOLOE class index {class_index} is outside active vocabulary"
                )
            if masks is not None:
                raw = np.asarray(masks[index])
                while raw.ndim > 2:
                    raw = raw[0]
                if raw.ndim != 2:
                    raise YOLOEError("each YOLOE mask must have shape [H,W]")
                if raw.shape != (height, width):
                    raw = cv2.resize(
                        raw.astype(np.float32),
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mask = np.ascontiguousarray(raw > 0.5, dtype=np.uint8)
            else:
                mask = YOLOEDetector._rasterize_polygon(
                    polygons[index], width=width, height=height
                )
            bbox = mask_bbox(mask)
            if bbox is None:
                continue
            output.append(
                RobotMaskPrediction(
                    backend="yoloe",
                    query=queries[class_index],
                    confidence=float(confidence),
                    bbox_xyxy=bbox,
                    image_width=width,
                    image_height=height,
                    mask=mask,
                )
            )
        return output

    def predict(
        self,
        frames: Sequence[np.ndarray],
        queries: Sequence[object] = ("robot",),
    ) -> list[list[RobotMaskPrediction]]:
        frames = list(frames)
        normalized = normalize_queries(queries)
        for frame in frames:
            validate_bgr_frame(frame)
        if not frames:
            return []
        with self._lock:
            model = self._load_model()
            self._activate(model, normalized)
            try:
                results = model.predict(
                    source=frames,
                    device=self.device,
                    imgsz=self.image_size,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    retina_masks=False,
                    agnostic_nms=False,
                    verbose=False,
                )
            except Exception as error:  # noqa: BLE001 - external runtime boundary
                raise YOLOEError(f"YOLOE prediction failed: {error}") from error
            if len(results) != len(frames):
                raise YOLOEError(
                    f"YOLOE returned {len(results)} results for {len(frames)} frames"
                )
            return [
                self._parse_result(result, frame=frame, queries=normalized)
                for result, frame in zip(results, frames, strict=True)
            ]

    def predict_requests(
        self, requests: Sequence[DetectionRequest]
    ) -> list[list[RobotMaskPrediction]]:
        requests = list(requests)
        if not requests:
            return []
        grouped: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, request in enumerate(requests):
            grouped[request_signature(request.queries)].append(index)
        output: list[list[RobotMaskPrediction] | None] = [None] * len(requests)
        for indices in grouped.values():
            first = requests[indices[0]]
            batches = self.predict(
                [requests[index].frame for index in indices], first.queries
            )
            for index, values in zip(indices, batches, strict=True):
                output[index] = values
        if any(values is None for values in output):
            raise YOLOEError("YOLOE request batching returned an incomplete result")
        return [values for values in output if values is not None]


YOLOEDetection = RobotMaskPrediction
