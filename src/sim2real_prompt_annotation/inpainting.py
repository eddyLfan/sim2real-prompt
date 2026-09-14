"""Lazy, local-only Big-LaMa inpainting adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .io_utils import sha256_file
from .robot_mask import validate_bgr_frame


class InpaintingError(RuntimeError):
    """Base error raised by the inpainting adapter."""


class InpaintingUnavailableError(InpaintingError):
    """The optional runtime/checkpoint cannot be used on this host."""


class Inpainter(Protocol):
    def ensure_ready(self) -> None:
        """Load and validate the configured runtime and checkpoint."""

    def cache_identity(self) -> dict[str, object]:
        """Return stable inputs that can affect generated pixels."""

    def inpaint(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Replace nonzero mask pixels in each BGR image."""


ModelFactory = Callable[[str, str], Any]


def _torch() -> Any:
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as error:
        raise InpaintingUnavailableError(
            "Big-LaMa requires PyTorch; install the Reference runtime first"
        ) from error
    return torch


def _torchscript_factory(model_path: str, device: str) -> Any:
    torch = _torch()
    try:
        model = torch.jit.load(model_path, map_location=device)
    except Exception as error:  # noqa: BLE001 - TorchScript loader boundary
        raise InpaintingUnavailableError(
            f"Failed to load Big-LaMa TorchScript checkpoint {model_path}: {error}"
        ) from error
    model.eval()
    return model.to(device)


class _OfficialLamaModel:
    """Adapt the official batch-dict model to the image/mask call surface."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def __call__(self, image: Any, mask: Any) -> Any:
        return self.model({"image": image, "mask": mask})


def _official_lama_factory(model_path: str, device: str) -> Any:
    directory = Path(model_path)
    config_path = directory / "config.yaml"
    checkpoint_path = directory / "models" / "best.ckpt"
    try:
        from omegaconf import OmegaConf
        from saicinpainting.training.trainers import load_checkpoint
    except (ImportError, ModuleNotFoundError) as error:
        raise InpaintingUnavailableError(
            "Official Big-LaMa directories require OmegaConf and the official "
            "advimman/lama saicinpainting package"
        ) from error
    try:
        train_config = OmegaConf.load(config_path)
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"
        model = load_checkpoint(
            train_config,
            str(checkpoint_path),
            strict=False,
            map_location=device,
        )
        model.freeze()
        model.eval()
        model.to(device)
    except Exception as error:  # noqa: BLE001 - official runtime boundary
        raise InpaintingUnavailableError(
            f"Failed to load official Big-LaMa model directory {directory}: {error}"
        ) from error
    return _OfficialLamaModel(model)


def _default_factory(model_path: str, device: str) -> Any:
    if Path(model_path).is_dir():
        return _official_lama_factory(model_path, device)
    return _torchscript_factory(model_path, device)


class BigLamaInpainter:
    """Adapter for official Big-LaMa directories or TorchScript exports.

    The official layout is ``<model>/config.yaml`` plus
    ``<model>/models/best.ckpt``. A TorchScript file must accept image and mask
    tensors directly. Both consume RGB images and binary masks in [0, 1] and
    return an ``inpainted`` RGB tensor. Nothing is downloaded implicitly.
    """

    def __init__(
        self,
        model_path: str | Path = "weights/big-lama",
        *,
        model_sha256: str | None = None,
        device: str = "cuda:0",
        modulo: int = 8,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if modulo < 1:
            raise ValueError("Big-LaMa modulo must be positive")
        self.model_path = str(model_path)
        self.model_sha256 = model_sha256
        self.device = device
        self.modulo = modulo
        self._custom_factory = model_factory is not None
        self._factory = model_factory or _default_factory
        self._model: Any | None = None
        self._checkpoint_identity: (
            tuple[tuple[str, int, int, int, int, int], dict[str, object]] | None
        ) = None
        self._config_identity: (
            tuple[tuple[str, int, int, int, int, int], dict[str, object]] | None
        ) = None

    def _validate_device(self) -> None:
        if not self.device.casefold().startswith("cuda"):
            return
        torch = _torch()
        if not torch.cuda.is_available():
            raise InpaintingUnavailableError(
                f"Big-LaMa device {self.device!r} requested but CUDA is unavailable"
            )

    def _checkpoint_file_identity(self, path: Path) -> dict[str, object]:
        """Hash a checkpoint once per concrete filesystem identity."""

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
        if self._checkpoint_identity is None or self._checkpoint_identity[0] != key:
            self._checkpoint_identity = (
                key,
                {
                    "path": str(resolved),
                    "size": stat.st_size,
                    "sha256": sha256_file(resolved),
                },
            )
        return self._checkpoint_identity[1]

    @staticmethod
    def _checkpoint_path(path: Path) -> Path:
        return path / "models" / "best.ckpt" if path.is_dir() else path

    @staticmethod
    def _validate_layout(path: Path) -> Path:
        if path.is_dir():
            config_path = path / "config.yaml"
            checkpoint_path = path / "models" / "best.ckpt"
            if not config_path.is_file() or not checkpoint_path.is_file():
                raise InpaintingUnavailableError(
                    "Official Big-LaMa model directory must contain config.yaml "
                    "and models/best.ckpt: "
                    f"{path}"
                )
            return checkpoint_path
        if path.is_file():
            return path
        raise InpaintingUnavailableError(
            "Big-LaMa model must be an official local model directory or an "
            f"exact TorchScript file: {path}"
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        path = Path(self.model_path).expanduser()
        checkpoint_path: Path | None = None
        if not self._custom_factory or self.model_sha256 is not None:
            checkpoint_path = self._validate_layout(path)
        if self.model_sha256 is not None:
            assert checkpoint_path is not None
            actual = self._checkpoint_file_identity(checkpoint_path)["sha256"]
            if actual != self.model_sha256:
                raise InpaintingUnavailableError(
                    "Big-LaMa checkpoint SHA-256 mismatch: "
                    f"expected {self.model_sha256}, got {actual}"
                )
        self._validate_device()
        self._model = self._factory(str(path), self.device)
        if not callable(self._model):
            raise InpaintingUnavailableError("Big-LaMa model must be callable")
        return self._model

    def ensure_ready(self) -> None:
        self._load_model()

    def cache_identity(self) -> dict[str, object]:
        path = Path(self.model_path).expanduser()
        checkpoint_path = self._checkpoint_path(path)
        checkpoint: dict[str, object] = {"locator": str(checkpoint_path)}
        if checkpoint_path.is_file():
            checkpoint.update(self._checkpoint_file_identity(checkpoint_path))
        config: dict[str, object] | None = None
        if path.is_dir():
            config_path = path / "config.yaml"
            config = {"locator": str(config_path)}
            if config_path.is_file():
                resolved = config_path.resolve()
                stat = resolved.stat()
                key = (
                    str(resolved),
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                    stat.st_dev,
                    stat.st_ino,
                )
                if self._config_identity is None or self._config_identity[0] != key:
                    self._config_identity = (
                        key,
                        {
                            "path": str(resolved),
                            "size": stat.st_size,
                            "sha256": sha256_file(resolved),
                        },
                    )
                config["file"] = self._config_identity[1]
        torch = _torch()
        return {
            "backend": "big_lama",
            "layout": "official-directory" if path.is_dir() else "torchscript",
            "model_path": str(path.resolve()),
            "checkpoint": checkpoint,
            "config": config,
            "configured_model_sha256": self.model_sha256,
            "torch_version": torch.__version__,
            "device": self.device,
            "modulo": self.modulo,
            "adapter": "official-or-torchscript-v2",
        }

    @staticmethod
    def _validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != shape:
            raise ValueError("inpainting mask must match the image height and width")
        if mask.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
            raise ValueError("inpainting mask must have bool or uint8 dtype")
        binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
        if not np.any(binary):
            raise ValueError("inpainting mask must contain foreground pixels")
        return binary

    def _prepare(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[Any, Any, int, int]:
        torch = _torch()
        import torch.nn.functional as functional

        image = validate_bgr_frame(image)
        mask = self._validate_mask(mask, image.shape[:2])
        height, width = image.shape[:2]
        pad_height = (-height) % self.modulo
        pad_width = (-width) % self.modulo
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        if pad_height or pad_width:
            image_tensor = functional.pad(
                image_tensor,
                (0, pad_width, 0, pad_height),
                mode="reflect" if min(height, width) > 1 else "replicate",
            )
            mask_tensor = functional.pad(
                mask_tensor,
                (0, pad_width, 0, pad_height),
                mode="constant",
                value=0,
            )
        return (
            image_tensor.to(self.device),
            mask_tensor.to(self.device),
            height,
            width,
        )

    @staticmethod
    def _extract_tensor(output: Any) -> Any:
        if isinstance(output, dict):
            for key in ("inpainted", "predicted_image", "output"):
                if key in output:
                    return output[key]
            raise InpaintingError("Big-LaMa result mapping has no image tensor")
        if isinstance(output, (tuple, list)):
            if not output:
                raise InpaintingError("Big-LaMa returned an empty sequence")
            return output[0]
        return output

    def inpaint(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        images = list(images)
        masks = list(masks)
        if len(images) != len(masks):
            raise ValueError("inpainting image and mask batch lengths must match")
        if not images:
            return []
        model = self._load_model()
        torch = _torch()

        prepared = [
            self._prepare(image, mask)
            for image, mask in zip(images, masks, strict=True)
        ]
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, (image_tensor, _, _, _) in enumerate(prepared):
            groups.setdefault(tuple(image_tensor.shape), []).append(index)
        results: list[np.ndarray | None] = [None] * len(images)
        with torch.inference_mode():
            for indices in groups.values():
                image_batch = torch.cat(
                    [prepared[index][0] for index in indices], dim=0
                )
                mask_batch = torch.cat([prepared[index][1] for index in indices], dim=0)
                try:
                    raw = self._extract_tensor(model(image_batch, mask_batch))
                except Exception as error:  # noqa: BLE001 - model boundary
                    raise InpaintingError(
                        f"Big-LaMa inference failed: {error}"
                    ) from error
                if (
                    not hasattr(raw, "shape")
                    or raw.ndim != 4
                    or raw.shape[0] != len(indices)
                    or raw.shape[1] != 3
                ):
                    raise InpaintingError("Big-LaMa output must have shape [B,3,H,W]")
                try:
                    finite = bool(torch.isfinite(raw).all().item())
                except (AttributeError, RuntimeError, TypeError) as error:
                    raise InpaintingError(
                        "Big-LaMa output must be a finite torch tensor"
                    ) from error
                if not finite:
                    raise InpaintingError(
                        "Big-LaMa output contains NaN or infinite values"
                    )
                for offset, index in enumerate(indices):
                    _, _, height, width = prepared[index]
                    tensor = raw[offset, :, :height, :width]
                    array = (
                        tensor.detach()
                        .float()
                        .clamp(0.0, 1.0)
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    rgb = np.rint(array * 255.0).astype(np.uint8)
                    results[index] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if any(result is None for result in results):
            raise InpaintingError("Big-LaMa returned an incomplete batch")
        return [result for result in results if result is not None]
