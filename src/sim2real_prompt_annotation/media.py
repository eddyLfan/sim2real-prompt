"""Provider-neutral temporal sampling and media preparation."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .config import MediaConfig
from .lerobot import SampleRecord


@dataclass(frozen=True)
class PreparedFrame:
    evidence_id: str
    frame_index: int
    timestamp_seconds: float
    jpeg: bytes


@dataclass(frozen=True)
class MediaGroup:
    source: Literal["sim", "real"]
    view: str
    frames: tuple[PreparedFrame, ...] = ()
    native_path: Path | None = None
    sampling_fps: float = 2.0


@dataclass(frozen=True)
class ReferenceImage:
    view: str
    frame_index: int
    evidence_id: str
    jpeg: bytes


@dataclass(frozen=True)
class PreparedMedia:
    groups: tuple[MediaGroup, ...]
    reference: ReferenceImage | None


class ReferenceInputError(ValueError):
    """The canonical per-episode Reference input is absent or unusable."""


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float


def _probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0:
        raise ValueError(f"Video has no decodable frames: {path}")
    return VideoInfo(frame_count=frame_count, fps=fps if fps > 0 else 1.0)


def _uniform_indices(frame_count: int, count: int) -> list[int]:
    count = min(frame_count, count)
    if count == 1:
        return [0]
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def _read_raw_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    result: dict[int, np.ndarray] = {}
    for index in sorted(set(indices)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"Failed to decode frame {index} from {path}")
        result[index] = frame
    capture.release()
    return result


def _keyframe_indices(path: Path, frame_count: int, count: int) -> list[int]:
    count = min(frame_count, count)
    if count <= 2:
        return _uniform_indices(frame_count, count)
    candidates = _uniform_indices(frame_count, min(frame_count, max(count * 5, 20)))
    frames = _read_raw_frames(path, candidates)
    previous: np.ndarray | None = None
    scored: list[tuple[float, int]] = []
    for index in candidates:
        gray = cv2.resize(cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY), (64, 64))
        if previous is not None:
            score = float(np.mean(cv2.absdiff(gray, previous)))
            scored.append((score, index))
        previous = gray
    selected = {0, frame_count - 1}
    for _, index in sorted(scored, reverse=True):
        selected.add(index)
        if len(selected) >= count:
            break
    if len(selected) < count:
        selected.update(_uniform_indices(frame_count, count))
    return sorted(selected)[:count]


def _resize_and_encode(frame: np.ndarray, config: MediaConfig) -> bytes:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > config.resize_long_edge:
        scale = config.resize_long_edge / longest
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode a sampled frame as JPEG")
    return encoded.tobytes()


def _encode_full_resolution(frame: np.ndarray, jpeg_quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("OpenCV failed to encode a Reference frame as JPEG")
    return encoded.tobytes()


def _explicit_mapping(record: SampleRecord, view: str) -> list[tuple[int, int]]:
    raw = record.metadata.get("frame_mapping")
    if isinstance(raw, dict):
        raw = raw.get(view) or raw.get("default")
    if not isinstance(raw, list):
        return []
    result: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, dict) and "sim_frame" in item and "real_frame" in item:
            result.append((int(item["sim_frame"]), int(item["real_frame"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            result.append((int(item[0]), int(item[1])))
    return sorted(result)


def _aligned_real_indices(
    sim_indices: list[int],
    sim_count: int,
    real_count: int,
    mapping: list[tuple[int, int]],
) -> list[int]:
    if mapping:
        return [
            min(mapping, key=lambda pair: abs(pair[0] - sim_index))[1]
            for sim_index in sim_indices
        ]
    if sim_count <= 1:
        return [0] * len(sim_indices)
    return [
        round(sim_index * (real_count - 1) / (sim_count - 1))
        for sim_index in sim_indices
    ]


class MediaPreparer:
    def __init__(self, config: MediaConfig):
        self.config = config
        self._manifest_cache: dict[Path, dict[int, dict[str, Any]]] = {}
        self._manifest_lock = threading.Lock()

    def _reference_manifest(self, dataset_root: Path) -> dict[int, dict[str, Any]]:
        with self._manifest_lock:
            cached = self._manifest_cache.get(dataset_root)
            if cached is not None:
                return cached
            rows: dict[int, dict[str, Any]] = {}
            path = dataset_root / "meta/reference_images.jsonl"
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if isinstance(row, dict) and isinstance(
                            row.get("episode_index"), int
                        ):
                            rows[row["episode_index"]] = row
            except (OSError, ValueError, json.JSONDecodeError):
                rows = {}
            self._manifest_cache[dataset_root] = rows
            return rows

    def _saved_reference(
        self,
        record: SampleRecord,
        views: list[str],
    ) -> ReferenceImage | None:
        row = self._reference_manifest(record.dataset_root).get(record.episode_index)
        if row is None:
            return None
        view = row.get("reference_view")
        frame_index = row.get("reference_frame_index")
        relative_path = row.get("reference_path")
        if (
            not isinstance(view, str)
            or view not in views
            or not isinstance(frame_index, int)
            or frame_index < 0
            or not isinstance(relative_path, str)
        ):
            return None
        dataset_root = record.dataset_root.resolve()
        candidates = [
            dataset_root / "Reference" / f"episode_{record.episode_index:06d}.{suffix}"
            for suffix in ("jpg", "jpeg", "png")
        ]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            return None
        path = existing[0].resolve()
        try:
            path.relative_to(dataset_root)
            payload = path.read_bytes()
        except (OSError, ValueError):
            return None
        manifest_path = (dataset_root / relative_path).resolve()
        if manifest_path != path:
            return None
        expected_digest = row.get("sha256")
        if isinstance(expected_digest, str) and (
            hashlib.sha256(payload).hexdigest() != expected_digest
        ):
            return None
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return None
        return ReferenceImage(
            view=view,
            frame_index=frame_index,
            evidence_id=f"reference:{view}:frame_{frame_index:06d}",
            jpeg=_resize_and_encode(frame, self.config),
        )

    def _views(self, record: SampleRecord) -> list[str]:
        available = list(record.paired_views)
        if not self.config.views:
            return available
        missing = set(self.config.views) - set(available)
        if missing:
            raise ValueError(
                f"{record.sample_id}: configured views are unavailable: "
                f"{sorted(missing)}"
            )
        return [view for view in self.config.views if view in available]

    def reference(
        self,
        record: SampleRecord,
        views: list[str] | None = None,
        *,
        full_resolution: bool = False,
        jpeg_quality: int = 95,
    ) -> ReferenceImage:
        """Select the deterministic same-episode Reference frame.

        Prompt preparation uses the configured resized representation. Dataset
        export can request a full-resolution JPEG while retaining the same view
        and frame identity.
        """

        views = self._views(record) if views is None else views
        if not views:
            raise ValueError(f"{record.sample_id}: no paired views selected")
        if not full_resolution:
            saved = self._saved_reference(record, views)
            if saved is not None:
                return saved
        view = self.config.reference_view
        if view not in record.real_videos:
            view = views[0]
        path = record.real_videos[view]
        frame_count = _probe_video(path).frame_count
        digest = hashlib.blake2b(
            f"{self.config.reference_seed}:{record.sample_id}".encode(),
            digest_size=8,
        ).digest()
        frame_index = int.from_bytes(digest, "little") % frame_count
        frame = _read_raw_frames(path, [frame_index])[frame_index]
        return ReferenceImage(
            view=view,
            frame_index=frame_index,
            evidence_id=f"reference:{view}:frame_{frame_index:06d}",
            jpeg=(
                _encode_full_resolution(frame, jpeg_quality)
                if full_resolution
                else _resize_and_encode(frame, self.config)
            ),
        )

    def prepare(self, record: SampleRecord) -> PreparedMedia:
        views = self._views(record)
        if not views:
            raise ValueError(f"{record.sample_id}: no paired views selected")
        reference = self._saved_reference(record, views)
        if reference is None:
            expected = (
                record.dataset_root
                / "Reference"
                / f"episode_{record.episode_index:06d}.jpg"
            )
            raise ReferenceInputError(
                f"{record.sample_id}: missing or invalid Reference input: {expected}; "
                "export References before prompt annotation"
            )
        groups: list[MediaGroup] = []
        for view in views:
            sim_path = record.sim_videos[view]
            real_path = record.real_videos[view]
            if self.config.mode == "native_video":
                for path in (sim_path, real_path):
                    size_mb = path.stat().st_size / (1024 * 1024)
                    if size_mb > self.config.max_native_video_mb:
                        raise ValueError(
                            f"Native video {path} is {size_mb:.1f} MiB, above "
                            f"media.max_native_video_mb={self.config.max_native_video_mb}"
                        )
                groups.extend(
                    [
                        MediaGroup(
                            source="sim",
                            view=view,
                            native_path=sim_path,
                            sampling_fps=self.config.native_video_fps,
                        ),
                        MediaGroup(
                            source="real",
                            view=view,
                            native_path=real_path,
                            sampling_fps=self.config.native_video_fps,
                        ),
                    ]
                )
                continue

            sim_info = _probe_video(sim_path)
            real_info = _probe_video(real_path)
            if self.config.strategy == "keyframe":
                sim_indices = _keyframe_indices(
                    sim_path, sim_info.frame_count, self.config.max_frames
                )
            else:
                sim_indices = _uniform_indices(
                    sim_info.frame_count, self.config.max_frames
                )
            real_indices = _aligned_real_indices(
                sim_indices,
                sim_info.frame_count,
                real_info.frame_count,
                _explicit_mapping(record, view),
            )
            sim_raw = _read_raw_frames(sim_path, sim_indices)
            real_raw = _read_raw_frames(real_path, real_indices)
            sim_frames = tuple(
                PreparedFrame(
                    evidence_id=f"sim:{view}:frame_{index:06d}",
                    frame_index=index,
                    timestamp_seconds=index / sim_info.fps,
                    jpeg=_resize_and_encode(sim_raw[index], self.config),
                )
                for index in sim_indices
            )
            real_frames = tuple(
                PreparedFrame(
                    evidence_id=f"real:{view}:frame_{index:06d}",
                    frame_index=index,
                    timestamp_seconds=index / real_info.fps,
                    jpeg=_resize_and_encode(real_raw[index], self.config),
                )
                for index in real_indices
            )
            duration = max((sim_info.frame_count - 1) / sim_info.fps, 0.1)
            sampling_fps = min(10.0, max(0.1, (len(sim_indices) - 1) / duration))
            groups.extend(
                [
                    MediaGroup(
                        source="sim",
                        view=view,
                        frames=sim_frames,
                        sampling_fps=sampling_fps,
                    ),
                    MediaGroup(
                        source="real",
                        view=view,
                        frames=real_frames,
                        sampling_fps=sampling_fps,
                    ),
                ]
            )
        return PreparedMedia(groups=tuple(groups), reference=reference)
