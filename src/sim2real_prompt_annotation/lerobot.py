"""Bounded discovery for paired LeRobot Sim/Real episodes."""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _fallback_paths(path: Path) -> Iterable[Path]:
    """Support the two common mount prefixes used by the original dataset."""

    text = str(path)
    yield path
    for before, after in (
        ("/media/datasets/", "/media/unify/"),
        ("/media/unify/", "/media/datasets/"),
    ):
        if before in text:
            yield Path(text.replace(before, after, 1))


def resolve_existing(path: str | Path, *, directory: bool = False) -> Path:
    requested = Path(path)
    for candidate in _fallback_paths(requested):
        if candidate.is_dir() if directory else candidate.is_file():
            return candidate
    kind = "directory" if directory else "file"
    raise FileNotFoundError(f"Missing {kind} at both media fallbacks: {requested}")


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    if not value:
        raise ValueError("Cannot make a non-empty sample id")
    return value


def _view_name(video_key: str) -> str:
    return video_key.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset_name: str
    dataset_root: Path
    episode_index: int
    episode_length: int
    fps: float
    robot_type: str | None
    task: str | None
    subtasks: tuple[dict[str, Any], ...]
    sim_videos: dict[str, Path]
    real_videos: dict[str, Path]
    metadata: dict[str, Any]

    @property
    def paired_views(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.sim_videos) & set(self.real_videos)))

    def annotation_metadata(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset_name,
            "episode_id": self.episode_index,
            "episode_length_frames": self.episode_length,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "task": self.task,
            "camera_views": list(self.paired_views),
            "subtasks": list(self.subtasks),
            **self.metadata,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            result.append(value)
    return result


def load_metadata_manifest(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    path = resolve_existing(path)
    records = (
        _read_jsonl(path)
        if path.suffix == ".jsonl"
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if isinstance(records, dict):
        records = records.get("samples", [records])
    if not isinstance(records, list):
        raise ValueError(f"Metadata manifest must contain a list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("sample_id"):
            raise ValueError(f"Every metadata manifest record needs sample_id: {path}")
        sample_id = str(record["sample_id"])
        result[sample_id] = {
            key: value for key, value in record.items() if key != "sample_id"
        }
    return result


def _dataset_roots(root: Path, dataset_glob: str) -> list[Path]:
    root = resolve_existing(root, directory=True)
    if (root / "meta/info.json").is_file():
        return [root]
    patterns = [value.strip() for value in dataset_glob.split(",") if value.strip()]
    if not patterns:
        raise ValueError("dataset_glob must contain at least one pattern")
    # Only inspect one bounded directory level; never recursively scan video trees.
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
        and (path / "meta/info.json").is_file()
    ]


def _subtasks_by_episode(dataset_root: Path) -> dict[int, tuple[dict[str, Any], ...]]:
    path = dataset_root / "labels/labels.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", []) if isinstance(payload, dict) else []
    return {
        int(item["episode_index"]): tuple(item.get("subtasks") or [])
        for item in labels
        if isinstance(item, dict) and "episode_index" in item
    }


def _video_path(
    dataset_root: Path,
    template: str,
    chunks_size: int,
    video_key: str,
    episode_index: int,
) -> Path:
    path = dataset_root / template.format(
        episode_chunk=episode_index // chunks_size,
        video_key=video_key,
        episode_index=episode_index,
    )
    return resolve_existing(path)


def discover_samples(
    root: str | Path,
    *,
    dataset_glob: str = "*",
    episodes: set[int] | None = None,
    limit: int | None = None,
    metadata_manifest: str | Path | None = None,
    min_episode_frames: int = 1,
    required_views: tuple[str, ...] = (),
    require_later_window: bool = False,
    window_stride: int = 1,
) -> list[SampleRecord]:
    """Discover episodes from metadata tables without recursively listing media."""

    manifest = load_metadata_manifest(metadata_manifest)
    result: list[SampleRecord] = []
    for dataset_root in _dataset_roots(Path(root), dataset_glob):
        info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
        feature_keys = {
            key
            for key, value in info.get("features", {}).items()
            if value.get("dtype") == "video"
        }
        pairs = {
            _view_name(key[:-4]): (key, key[:-4])
            for key in feature_keys
            if key.endswith("_sim") and key[:-4] in feature_keys
        }
        if not pairs or set(required_views) - set(pairs):
            continue
        chunks_size = int(info.get("chunks_size", 1000))
        template = str(
            info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
        )
        subtasks = _subtasks_by_episode(dataset_root)
        episode_path = dataset_root / "meta/episodes.jsonl"
        episode_rows = [
            episode
            for episode in _read_jsonl(episode_path)
            if int(episode.get("length", 0)) >= int(min_episode_frames)
        ]
        if require_later_window and not any(
            int(episode["length"]) >= int(min_episode_frames) + int(window_stride)
            for episode in episode_rows
        ):
            continue
        for episode in episode_rows:
            episode_index = int(episode["episode_index"])
            if episodes is not None and episode_index not in episodes:
                continue
            sample_id = f"{_safe_id(dataset_root.name)}__episode_{episode_index:06d}"
            sim_videos = {
                view: _video_path(
                    dataset_root, template, chunks_size, sim_key, episode_index
                )
                for view, (sim_key, _) in pairs.items()
            }
            real_videos = {
                view: _video_path(
                    dataset_root, template, chunks_size, real_key, episode_index
                )
                for view, (_, real_key) in pairs.items()
            }
            tasks = episode.get("tasks") or []
            record_metadata = dict(manifest.get(sample_id, {}))
            record_metadata.setdefault(
                "metadata_evidence",
                {
                    "robot": "meta/info.json:robot_type",
                    "camera": "meta/info.json:features",
                    "task": "meta/episodes.jsonl:tasks",
                    "actions": "labels/labels.json:subtasks"
                    if subtasks.get(episode_index)
                    else None,
                },
            )
            result.append(
                SampleRecord(
                    sample_id=sample_id,
                    dataset_name=dataset_root.name,
                    dataset_root=dataset_root,
                    episode_index=episode_index,
                    episode_length=int(episode.get("length", 0)),
                    fps=float(info.get("fps", 0.0)),
                    robot_type=str(info["robot_type"])
                    if info.get("robot_type")
                    else None,
                    task=str(tasks[0]) if tasks else None,
                    subtasks=subtasks.get(episode_index, ()),
                    sim_videos=sim_videos,
                    real_videos=real_videos,
                    metadata=record_metadata,
                )
            )
            if limit is not None and len(result) >= limit:
                return result
    return result
