"""Lazy adapter for the official showlab/RobotSeg image/video predictor."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io_utils import sha256_file
from .models import RobotMaskPrediction
from .robot_mask import mask_bbox, validate_bgr_frame


class RobotSegError(RuntimeError):
    """Base error raised by the RobotSeg adapter."""


class RobotSegUnavailableError(RobotSegError):
    """The optional runtime/checkpoint cannot be used on this host."""


PredictorFactory = Callable[[str, str, str], Any]


def _official_factory(config_name: str, checkpoint: str, device: str) -> Any:
    try:
        from robotseg.build_robotseg import build_robotseg_video_predictor
    except (ImportError, ModuleNotFoundError) as error:
        raise RobotSegUnavailableError(
            "RobotSeg runtime is unavailable. Install the official showlab/RobotSeg "
            "package and its compiled dependencies before preprocessing."
        ) from error
    return build_robotseg_video_predictor(
        config_name,
        checkpoint,
        device=device,
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class RobotSegSegmenter:
    """Automatic full-robot segmentation with no manual point or box prompt.

    The upstream predictor currently accepts an MP4 or a numbered JPEG directory.
    For a single in-memory frame, this adapter writes one quality-100 temporary JPEG,
    calls ``add_new_robot`` on frame zero, and immediately discards the state.
    """

    def __init__(
        self,
        model_path: str | Path = "robotseg.pt",
        *,
        model_sha256: str | None = None,
        config_name: str = "configs/robotseg-infer",
        device: str = "cuda:0",
        category: str = "robot",
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        if category not in {"robot", "arm", "gripper"}:
            raise ValueError("RobotSeg category must be robot, arm, or gripper")
        if not config_name.strip():
            raise ValueError("RobotSeg config_name must be non-empty")
        self.model_path = str(model_path)
        self.model_sha256 = model_sha256
        self.config_name = config_name
        self.device = device
        self.category = category
        self._custom_factory = predictor_factory is not None
        self._factory = predictor_factory or _official_factory
        self._predictor: Any | None = None
        self._weight_identity: (
            tuple[tuple[str, int, int, int, int, int], dict[str, object]] | None
        ) = None

    def _validate_device(self) -> None:
        if not self.device.casefold().startswith("cuda"):
            return
        try:
            import torch
        except (ImportError, ModuleNotFoundError) as error:
            raise RobotSegUnavailableError(
                "CUDA RobotSeg requires a working PyTorch installation"
            ) from error
        if not torch.cuda.is_available():
            raise RobotSegUnavailableError(
                f"RobotSeg device {self.device!r} requested but CUDA is unavailable"
            )

    def _checkpoint_identity(self, path: Path) -> dict[str, object]:
        """Hash the local checkpoint once per concrete filesystem identity."""

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
        if self._weight_identity is None or self._weight_identity[0] != key:
            self._weight_identity = (
                key,
                {
                    "path": str(resolved),
                    "size": stat.st_size,
                    "sha256": sha256_file(resolved),
                },
            )
        return self._weight_identity[1]

    def _load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        path = Path(self.model_path).expanduser()
        if not self._custom_factory or self.model_sha256 is not None:
            if not path.is_file():
                raise RobotSegUnavailableError(
                    f"RobotSeg checkpoint is not an exact local file: {path}"
                )
        if self.model_sha256 is not None and path.is_file():
            actual = self._checkpoint_identity(path)["sha256"]
            if actual != self.model_sha256:
                raise RobotSegUnavailableError(
                    "RobotSeg checkpoint SHA-256 mismatch: "
                    f"expected {self.model_sha256}, got {actual}"
                )
        self._validate_device()
        try:
            predictor = self._factory(self.config_name, str(path), self.device)
        except RobotSegUnavailableError:
            raise
        except Exception as error:  # noqa: BLE001 - optional runtime boundary
            raise RobotSegUnavailableError(
                f"Failed to initialize RobotSeg from {path}: {error}"
            ) from error
        required = ("init_state", "add_new_robot")
        if any(not callable(getattr(predictor, name, None)) for name in required):
            raise RobotSegUnavailableError(
                "RobotSeg predictor does not expose init_state/add_new_robot"
            )
        self._predictor = predictor
        return predictor

    def ensure_ready(self) -> None:
        self._load_predictor()

    def cache_identity(self) -> dict[str, object]:
        path = Path(self.model_path).expanduser()
        model: dict[str, object] = {"locator": self.model_path}
        if path.is_file():
            model["file"] = self._checkpoint_identity(path)
        try:
            runtime_version: str | None = version("robotseg")
        except PackageNotFoundError:
            runtime_version = None
        return {
            "backend": "robotseg",
            "runtime_version": runtime_version,
            "model": model,
            "configured_model_sha256": self.model_sha256,
            "config_name": self.config_name,
            "device": self.device,
            "category": self.category,
            "adapter": "official-video-predictor-single-frame-v1",
        }

    def _predict_one(self, frame: np.ndarray) -> list[RobotMaskPrediction]:
        frame = validate_bgr_frame(frame)
        predictor = self._load_predictor()
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not success:
            raise RobotSegError("OpenCV failed to encode the temporary RobotSeg frame")
        state: Any | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="sim2real-robotseg-") as directory:
                (Path(directory) / "00000.jpg").write_bytes(encoded.tobytes())
                state = predictor.init_state(
                    video_path=directory,
                    async_loading_frames=False,
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False,
                )
                output = predictor.add_new_robot(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=0,
                    robot=self.category,
                )
        except Exception as error:  # noqa: BLE001 - optional runtime boundary
            raise RobotSegError(
                f"RobotSeg frame-zero inference failed: {error}"
            ) from error
        finally:
            reset_state = getattr(predictor, "reset_state", None)
            if state is not None and callable(reset_state):
                try:
                    reset_state(state)
                except Exception:  # noqa: BLE001 - best-effort upstream cleanup
                    pass
        if not isinstance(output, tuple) or len(output) != 3:
            raise RobotSegError("RobotSeg add_new_robot returned an invalid result")
        _, object_ids, raw_logits = output
        logits = _as_numpy(raw_logits)
        if logits.ndim < 2 or len(object_ids) < 1:
            return []
        while logits.ndim > 2:
            logits = logits[0]
        if logits.shape != frame.shape[:2]:
            logits = cv2.resize(
                logits.astype(np.float32),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        mask = np.ascontiguousarray(logits > 0.0, dtype=np.uint8)
        bbox = mask_bbox(mask)
        if bbox is None:
            return []
        foreground_logits = np.clip(logits[mask.astype(bool)], -20.0, 20.0)
        confidence = float(np.mean(1.0 / (1.0 + np.exp(-foreground_logits))))
        return [
            RobotMaskPrediction(
                backend="robotseg",
                query=self.category,
                confidence=confidence,
                bbox_xyxy=bbox,
                image_width=frame.shape[1],
                image_height=frame.shape[0],
                mask=mask,
            )
        ]

    def predict(
        self,
        frames: Sequence[np.ndarray],
        queries: Sequence[str] = ("robot",),
    ) -> list[list[RobotMaskPrediction]]:
        del queries  # RobotSeg uses its learned automatic robot category prompt.
        return [self._predict_one(frame) for frame in frames]
