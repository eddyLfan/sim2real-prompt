"""Single-open, direct-seek decoding of the configured Real camera video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import PROMPT_FRAME_COUNT, PromptConfig
from .models import EpisodeRecord, RealFrame, RealFrameBundle


def uniform_frame_indices(
    frame_count: int,
    count: int = PROMPT_FRAME_COUNT,
) -> tuple[int, ...]:
    """Return unique inclusive-endpoint indices, always beginning at frame zero."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 1:
        raise ValueError("count must be positive")
    if frame_count < count:
        raise ValueError(
            f"video has {frame_count} frames, fewer than the required {count}"
        )
    if count == 1:
        return (0,)
    return tuple(
        round(position * (frame_count - 1) / (count - 1)) for position in range(count)
    )


def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode a Real frame as JPEG")
    return encoded.tobytes()


def resize_and_encode_jpeg(
    frame: np.ndarray,
    *,
    resize_long_edge: int,
    jpeg_quality: int,
) -> bytes:
    """Downscale without upsampling and encode one VLM input frame."""

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("expected one non-empty BGR image")
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > resize_long_edge:
        scale = resize_long_edge / longest
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return _encode_jpeg(frame, jpeg_quality)


def decode_jpeg(payload: bytes) -> np.ndarray:
    """Decode an in-memory JPEG and fail instead of returning an empty image."""

    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("invalid JPEG payload")
    return frame


def decode_real_first_frame(
    record: EpisodeRecord,
) -> np.ndarray:
    """Decode only Real frame zero for a cached-Prompt Reference rerun."""

    capture = cv2.VideoCapture(str(record.real_video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open Real video: {record.real_video}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(f"Cannot decode Real frame zero: {record.real_video}")
    return frame


def decode_real_video(
    record: EpisodeRecord,
    config: PromptConfig,
) -> RealFrameBundle:
    """Decode frame zero and eight uniform Prompt frames with one capture.

    The ``VideoCapture`` is opened exactly once and seeks directly to the eight
    selected indices. This avoids decoding thousands of unused intermediate frames.
    The full-resolution first frame stays as its original BGR array for YOLOE; only
    the smaller Prompt copy is JPEG-encoded.
    """

    if record.real_view == "" or record.real_video == Path():
        raise ValueError(f"{record.sample_id}: Real camera input is not configured")
    capture = cv2.VideoCapture(str(record.real_video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open Real video: {record.real_video}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            raise ValueError(f"Real video has no frames: {record.real_video}")
        if frame_count != record.episode_length:
            raise ValueError(
                f"Real video has {frame_count} frames but episode metadata declares "
                f"{record.episode_length}: {record.real_video}"
            )
        if fps <= 0 or abs(fps - record.fps) > 0.01:
            raise ValueError(
                f"Real video fps {fps} differs from metadata fps {record.fps}: "
                f"{record.real_video}"
            )
        indices = uniform_frame_indices(frame_count, config.frame_count)
        encoded_frames: dict[int, RealFrame] = {}
        first_frame_bgr: np.ndarray | None = None
        width = 0
        height = 0
        for position, frame_index in enumerate(indices):
            if position > 0 and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise RuntimeError(
                    f"Failed to seek to frame {frame_index} in {record.real_video}"
                )
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(
                    f"Failed to decode frame {frame_index} from {record.real_video}"
                )
            if frame_index == 0:
                height, width = frame.shape[:2]
                first_frame_bgr = frame.copy()
            encoded_frames[frame_index] = RealFrame(
                frame_index=frame_index,
                timestamp_seconds=frame_index / fps,
                jpeg=resize_and_encode_jpeg(
                    frame,
                    resize_long_edge=config.resize_long_edge,
                    jpeg_quality=config.jpeg_quality,
                ),
            )
    finally:
        capture.release()

    if first_frame_bgr is None or len(encoded_frames) != len(indices):
        raise RuntimeError(f"Incomplete Real frame bundle for {record.sample_id}")
    return RealFrameBundle(
        sample_id=record.sample_id,
        view=record.real_view,
        video_path=record.real_video,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        first_frame_bgr=first_frame_bgr,
        prompt_frames=tuple(encoded_frames[index] for index in indices),
    )
