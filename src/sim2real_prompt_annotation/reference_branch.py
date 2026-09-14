"""Frame-zero robot removal and single full-scene Reference construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ReferenceConfig
from .inpainting import BigLamaInpainter, Inpainter
from .models import (
    RobotMaskDiagnostic,
    RobotMaskPrediction,
    SceneReferenceArtifact,
    SceneReferenceResult,
)
from .robot_mask import RobotMaskSegmenter, validate_bgr_frame
from .robotseg import RobotSegSegmenter
from .yoloe import YOLOEDetector

REFERENCE_DIRECTORY = "Reference"
_CHANGE_THRESHOLD = 2


class NoValidSceneReferenceError(RuntimeError):
    """Robot removal could not produce a safe training Reference."""

    def __init__(self, message: str, *, diagnostics: dict[str, object] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class SceneReferenceRuntimeError(RuntimeError):
    """A non-deterministic backend or runtime failure; never negative-cache it."""

    def __init__(self, message: str, *, diagnostics: dict[str, object] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


# One-release compatibility name for callers that handled the old crop branch.
NoValidReferenceError = NoValidSceneReferenceError


@dataclass(frozen=True, slots=True)
class ReferenceBranchInput:
    """One independently identifiable Real-frame-zero Reference job."""

    sample_id: str
    episode_index: int
    frame0: np.ndarray | bytes
    source_view: str = "camera_head"


@dataclass(frozen=True, slots=True)
class _PreparedReference:
    request: ReferenceBranchInput
    frame: np.ndarray
    masks: tuple[RobotMaskPrediction, ...]
    diagnostics: tuple[RobotMaskDiagnostic, ...]
    final_mask: np.ndarray
    final_mask_sha256: str
    final_mask_area_fraction: float


def _decode_frame(frame0: np.ndarray | bytes) -> np.ndarray:
    if isinstance(frame0, bytes):
        frame = cv2.imdecode(np.frombuffer(frame0, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Real frame zero is not a decodable image")
        return frame
    if not isinstance(frame0, np.ndarray):
        raise TypeError("Real frame zero must be a numpy array or encoded image")
    return validate_bgr_frame(frame0)


def _identity(component: object) -> dict[str, object]:
    cache_identity = getattr(component, "cache_identity", None)
    if callable(cache_identity):
        value = cache_identity()
        if not isinstance(value, dict) or not value:
            raise TypeError(
                f"{type(component).__qualname__}.cache_identity() must return "
                "a non-empty object"
            )
        return value
    return {"backend": type(component).__qualname__}


def _mask_summary(mask: np.ndarray) -> dict[str, object]:
    binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    return {
        "sha256": hashlib.sha256(binary.tobytes(order="C")).hexdigest(),
        "area_fraction": float(np.count_nonzero(binary)) / float(binary.size),
        "width": int(binary.shape[1]),
        "height": int(binary.shape[0]),
    }


class ReferenceBranch:
    """Segment, inpaint, and fail closed on one complete frame-zero scene."""

    def __init__(
        self,
        segmenter: RobotMaskSegmenter,
        inpainter: Inpainter,
        config: ReferenceConfig,
        *,
        residual_detector: RobotMaskSegmenter | None = None,
    ) -> None:
        if config.backend != "robotseg":
            raise ValueError(f"unsupported Reference backend: {config.backend}")
        if config.residual_check and residual_detector is None:
            raise ValueError(
                "residual_check requires an explicit residual detector; it is QC, "
                "not a robot-mask fallback"
            )
        self.segmenter = segmenter
        self.inpainter = inpainter
        self.config = config
        self.residual_detector = residual_detector

    @classmethod
    def from_config(cls, config: ReferenceConfig) -> ReferenceBranch:
        """Create lazy local-only backends from one strict configuration."""

        segmenter = RobotSegSegmenter(
            config.model_path,
            model_sha256=config.model_sha256,
            config_name=config.robotseg_config,
            device=config.device,
            category=config.robot_category,
        )
        inpainter = BigLamaInpainter(
            config.inpainting_model_path,
            model_sha256=config.inpainting_model_sha256,
            device=config.inpainting_device or config.device,
            modulo=config.inpainting_modulo,
        )
        residual_detector: RobotMaskSegmenter | None = None
        if config.residual_check:
            residual_detector = YOLOEDetector(
                config.yoloe_model_path,
                model_sha256=config.yoloe_model_sha256,
                text_model_path=config.yoloe_text_model_path,
                text_model_sha256=config.yoloe_text_model_sha256,
                device=config.device,
                image_size=config.yoloe_image_size,
                confidence=config.yoloe_confidence,
                iou_threshold=config.yoloe_iou_threshold,
                embedding_cache_size=config.embedding_cache_size,
            )
        return cls(
            segmenter,
            inpainter,
            config,
            residual_detector=residual_detector,
        )

    def ensure_ready(self) -> None:
        """Preflight every configured production backend without processing data."""

        for component in (
            self.segmenter,
            self.inpainter,
            self.residual_detector,
        ):
            if component is None:
                continue
            ensure_ready = getattr(component, "ensure_ready", None)
            if callable(ensure_ready):
                ensure_ready()

    def cache_identity(self) -> dict[str, object]:
        """Bind cache reuse to models, mask processing, compositing, and QC."""

        return {
            "schema": "robot-removed-scene-reference-v3",
            "source": "real-frame-0",
            "segmenter": _identity(self.segmenter),
            "robot_queries": list(self.config.robot_queries),
            "mask_close_kernel": self.config.mask_close_kernel,
            "mask_dilation_pixels": self.config.mask_dilation_pixels,
            "min_mask_area_fraction": self.config.min_mask_area_fraction,
            "max_mask_area_fraction": self.config.max_mask_area_fraction,
            "inpainter": _identity(self.inpainter),
            "min_inpaint_change_fraction": self.config.min_inpaint_change_fraction,
            "inpaint_change_pixel_threshold": _CHANGE_THRESHOLD,
            "residual_check": self.config.residual_check,
            "residual_detector": (
                _identity(self.residual_detector)
                if self.residual_detector is not None
                else None
            ),
            "max_residual_area_fraction": self.config.max_residual_area_fraction,
            "jpeg_quality": self.config.jpeg_quality,
            "composition": "replace-final-mask-only-v1",
            "mask_hash_encoding": "full-resolution-uint8-binary-c-order",
        }

    def _fingerprint(self, request: ReferenceBranchInput, frame: np.ndarray) -> str:
        settings = {
            "sample_id": request.sample_id,
            "episode_index": request.episode_index,
            "source_view": request.source_view,
            "source_frame_index": 0,
            "cache_identity": self.cache_identity(),
        }
        digest = hashlib.sha256()
        digest.update(str(frame.shape).encode("ascii"))
        digest.update(str(frame.dtype).encode("ascii"))
        digest.update(frame.tobytes(order="C"))
        digest.update(
            json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return f"sha256:{digest.hexdigest()}"

    def _postprocess_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        mask = np.ascontiguousarray(raw_mask > 0, dtype=np.uint8)
        close_size = self.config.mask_close_kernel
        if close_size > 1:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_size, close_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        dilation = self.config.mask_dilation_pixels
        if dilation > 0:
            dilation_size = 2 * dilation + 1
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
            )
            mask = cv2.dilate(mask, dilation_kernel, iterations=1)
        return np.ascontiguousarray(mask > 0, dtype=np.uint8)

    def _prepare(
        self,
        request: ReferenceBranchInput,
        frame: np.ndarray,
        masks: Sequence[RobotMaskPrediction],
    ) -> _PreparedReference:
        try:
            validated_masks = tuple(
                item
                if isinstance(item, RobotMaskPrediction)
                else RobotMaskPrediction.model_validate(item)
                for item in masks
            )
            diagnostics = tuple(
                RobotMaskDiagnostic.from_prediction(item) for item in validated_masks
            )
        except Exception as error:
            raise SceneReferenceRuntimeError(
                f"{request.sample_id}: robot segmenter returned invalid mask data",
                diagnostics={"stage": "segmentation", "error": str(error)},
            ) from error
        base_diagnostics: dict[str, object] = {
            "sample_id": request.sample_id,
            "source_view": request.source_view,
            "source_frame_index": 0,
            "raw_masks": [item.model_dump(mode="json") for item in diagnostics],
        }
        if not validated_masks:
            raise NoValidSceneReferenceError(
                f"{request.sample_id}: RobotSeg found no robot mask in Real frame 0",
                diagnostics=base_diagnostics,
            )
        union = np.zeros(frame.shape[:2], dtype=np.uint8)
        for prediction in validated_masks:
            if prediction.mask.shape != frame.shape[:2]:
                raise SceneReferenceRuntimeError(
                    f"{request.sample_id}: robot mask dimensions differ from frame 0",
                    diagnostics={**base_diagnostics, "stage": "segmentation"},
                )
            union |= prediction.mask
        final_mask = self._postprocess_mask(union)
        final_summary = _mask_summary(final_mask)
        area = float(final_summary["area_fraction"])
        base_diagnostics["final_mask"] = final_summary
        if not (
            self.config.min_mask_area_fraction
            <= area
            <= self.config.max_mask_area_fraction
        ):
            raise NoValidSceneReferenceError(
                f"{request.sample_id}: final robot-removal mask area {area:.6f} is "
                "outside configured bounds",
                diagnostics=base_diagnostics,
            )
        return _PreparedReference(
            request=request,
            frame=frame,
            masks=validated_masks,
            diagnostics=diagnostics,
            final_mask=final_mask,
            final_mask_sha256=str(final_summary["sha256"]),
            final_mask_area_fraction=area,
        )

    @staticmethod
    def _valid_generated(source: np.ndarray, generated: np.ndarray) -> np.ndarray:
        if (
            not isinstance(generated, np.ndarray)
            or generated.dtype != np.uint8
            or generated.shape != source.shape
        ):
            raise SceneReferenceRuntimeError(
                "inpainter output must be uint8 BGR with the source frame shape",
                diagnostics={"stage": "inpainting"},
            )
        return generated

    @staticmethod
    def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
        success, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            raise SceneReferenceRuntimeError(
                "OpenCV failed to encode the scene Reference JPEG",
                diagnostics={"stage": "reference_encoding"},
            )
        return encoded.tobytes()

    @staticmethod
    def _encode_mask_png(mask: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
        if not success:
            raise SceneReferenceRuntimeError(
                "OpenCV failed to encode the removal mask PNG",
                diagnostics={"stage": "mask_encoding"},
            )
        return encoded.tobytes()

    def _residual_fraction(
        self,
        scene: np.ndarray,
        predictions: Sequence[RobotMaskPrediction],
    ) -> float:
        if not predictions:
            return 0.0
        union = np.zeros(scene.shape[:2], dtype=np.uint8)
        for prediction in predictions:
            if prediction.mask.shape != scene.shape[:2]:
                raise SceneReferenceRuntimeError(
                    "residual robot mask dimensions differ from scene",
                    diagnostics={"stage": "residual_qc"},
                )
            union |= prediction.mask
        return float(np.count_nonzero(union)) / float(union.size)

    def _build_result(
        self,
        prepared: _PreparedReference,
        generated: np.ndarray,
        residual_predictions: Sequence[RobotMaskPrediction],
    ) -> SceneReferenceResult:
        source = prepared.frame
        generated = self._valid_generated(source, generated)
        foreground = prepared.final_mask.astype(bool)
        scene = source.copy()
        scene[foreground] = generated[foreground]
        if not np.array_equal(scene[~foreground], source[~foreground]):
            raise SceneReferenceRuntimeError(
                f"{prepared.request.sample_id}: pixels outside removal mask changed",
                diagnostics={"stage": "composition"},
            )
        difference = np.max(
            np.abs(scene.astype(np.int16) - source.astype(np.int16)), axis=2
        )
        change_fraction = float(
            np.count_nonzero(difference[foreground] > _CHANGE_THRESHOLD)
        ) / float(np.count_nonzero(foreground))
        residual_fraction = self._residual_fraction(scene, residual_predictions)
        residual_threshold = self.config.max_residual_area_fraction
        residual_pass = residual_fraction <= residual_threshold
        residual_qa = {
            "enabled": self.config.residual_check,
            "detector": (
                _identity(self.residual_detector)
                if self.residual_detector is not None
                else None
            ),
            "queries": list(self.config.robot_queries),
            "area_fraction": residual_fraction,
            "threshold": residual_threshold,
            "pass": residual_pass,
        }
        quality_control = {
            "outside_mask_unchanged": True,
            "inpaint_change_fraction": change_fraction,
            "inpaint_change_pixel_threshold": _CHANGE_THRESHOLD,
            "min_inpaint_change_fraction": self.config.min_inpaint_change_fraction,
            "residual_check_enabled": self.config.residual_check,
            "residual_mask_area_fraction": residual_fraction,
            "max_residual_area_fraction": residual_threshold,
        }
        failure_diagnostics = {
            "sample_id": prepared.request.sample_id,
            "raw_masks": [
                item.model_dump(mode="json") for item in prepared.diagnostics
            ],
            "final_mask": _mask_summary(prepared.final_mask),
            "quality_control": quality_control,
            "residual_qa": residual_qa,
        }
        if change_fraction < self.config.min_inpaint_change_fraction:
            raise NoValidSceneReferenceError(
                f"{prepared.request.sample_id}: inpainting changed too little of "
                "the robot-removal region",
                diagnostics=failure_diagnostics,
            )
        if not residual_pass:
            raise NoValidSceneReferenceError(
                f"{prepared.request.sample_id}: residual robot mask area "
                f"{residual_fraction:.6f} exceeds the QC limit",
                diagnostics=failure_diagnostics,
            )
        jpeg = self._encode_jpeg(scene, self.config.jpeg_quality)
        jpeg_sha256 = hashlib.sha256(jpeg).hexdigest()
        mask_processing = {
            "sha256": prepared.final_mask_sha256,
            "area_fraction": prepared.final_mask_area_fraction,
            "close_kernel": self.config.mask_close_kernel,
            "dilation_pixels": self.config.mask_dilation_pixels,
            "hash_encoding": "full-resolution-uint8-binary-c-order",
        }
        provenance = {
            "operation": "robot_removal_inpainting",
            "source_frame_hash_encoding": "bgr8-c-order",
            "segmenter": _identity(self.segmenter),
            "robot_masks": [
                item.model_dump(mode="json") for item in prepared.diagnostics
            ],
            "mask_processing": mask_processing,
            "final_mask": mask_processing,
            "inpainter": _identity(self.inpainter),
            "residual_qa": residual_qa,
            "quality_control": quality_control,
        }
        artifact = SceneReferenceArtifact(
            sample_id=prepared.request.sample_id,
            reference_id=f"sha256:{jpeg_sha256}",
            relative_path=(
                Path(REFERENCE_DIRECTORY)
                / f"episode_{prepared.request.episode_index:06d}"
                / "reference_00.jpg"
            ),
            jpeg=jpeg,
            source_view=prepared.request.source_view,
            source_frame_index=0,
            scope="environment",
            reference_kind="robot_removed_scene",
            width=scene.shape[1],
            height=scene.shape[0],
            source_frame_sha256=hashlib.sha256(source.tobytes(order="C")).hexdigest(),
            mask_sha256=prepared.final_mask_sha256,
            mask_area_fraction=prepared.final_mask_area_fraction,
            sha256=jpeg_sha256,
            provenance=provenance,
        )
        return SceneReferenceResult(
            sample_id=prepared.request.sample_id,
            artifact=artifact,
            robot_masks=prepared.diagnostics,
            removal_mask_png=self._encode_mask_png(prepared.final_mask),
            input_fingerprint=self._fingerprint(prepared.request, source),
        )

    def process_batch(
        self, requests: Sequence[ReferenceBranchInput]
    ) -> list[SceneReferenceResult]:
        """Process one batch; any unsafe item fails the batch closed."""

        requests = list(requests)
        if not requests:
            return []
        frames: list[np.ndarray] = []
        for request in requests:
            if not request.sample_id.strip() or not request.source_view.strip():
                raise ValueError("sample_id and source_view must be non-empty")
            if request.episode_index < 0:
                raise ValueError("episode_index must be non-negative")
            frames.append(_decode_frame(request.frame0))
        try:
            predicted = list(self.segmenter.predict(frames, self.config.robot_queries))
        except Exception as error:
            raise SceneReferenceRuntimeError(
                f"robot segmentation failed: {error}",
                diagnostics={"stage": "segmentation"},
            ) from error
        if len(predicted) != len(requests):
            raise SceneReferenceRuntimeError(
                "robot segmenter returned an incomplete batch",
                diagnostics={"stage": "segmentation"},
            )
        prepared = [
            self._prepare(request, frame, masks)
            for request, frame, masks in zip(requests, frames, predicted, strict=True)
        ]
        try:
            generated = list(
                self.inpainter.inpaint(
                    [item.frame for item in prepared],
                    [item.final_mask for item in prepared],
                )
            )
        except Exception as error:
            raise SceneReferenceRuntimeError(
                f"scene inpainting failed: {error}",
                diagnostics={"stage": "inpainting"},
            ) from error
        if len(generated) != len(prepared):
            raise SceneReferenceRuntimeError(
                "inpainter returned an incomplete batch",
                diagnostics={"stage": "inpainting"},
            )
        scenes: list[np.ndarray] = []
        for item, output in zip(prepared, generated, strict=True):
            output = self._valid_generated(item.frame, output)
            scene = item.frame.copy()
            foreground = item.final_mask.astype(bool)
            scene[foreground] = output[foreground]
            scenes.append(scene)
        residuals: list[list[RobotMaskPrediction]]
        if self.residual_detector is None:
            residuals = [[] for _ in scenes]
        else:
            try:
                residuals = list(
                    self.residual_detector.predict(scenes, self.config.robot_queries)
                )
            except Exception as error:
                raise SceneReferenceRuntimeError(
                    f"residual robot QC failed: {error}",
                    diagnostics={"stage": "residual_qc"},
                ) from error
            if len(residuals) != len(scenes):
                raise SceneReferenceRuntimeError(
                    "residual detector returned an incomplete batch",
                    diagnostics={"stage": "residual_qc"},
                )
        return [
            self._build_result(item, output, residual)
            for item, output, residual in zip(
                prepared, generated, residuals, strict=True
            )
        ]

    def process(
        self,
        *,
        sample_id: str,
        episode_index: int,
        frame0: np.ndarray | bytes,
        source_view: str = "camera_head",
    ) -> SceneReferenceResult:
        """Process exactly one complete Real frame zero."""

        return self.process_batch(
            [
                ReferenceBranchInput(
                    sample_id=sample_id,
                    episode_index=episode_index,
                    frame0=frame0,
                    source_view=source_view,
                )
            ]
        )[0]
