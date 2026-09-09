"""Strict, read-only discovery for Real videos in paired LeRobot datasets."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .io_utils import read_json, read_jsonl, resolve_existing
from .models import EpisodeRecord, Split

_VALID_SPLITS = {"train", "validation"}


def canonical_episode_id(source_id: str, episode_index: int) -> str:
    """Injectively encode a source-local episode key without lossy cleanup."""

    if not source_id or source_id != source_id.strip():
        raise ValueError("source_id must be a trimmed non-empty string")
    if isinstance(episode_index, bool) or episode_index < 0:
        raise ValueError("episode_index must be a non-negative integer")
    return f"{len(source_id)}:{source_id}:{episode_index}"


def sample_artifact_stem(sample_id: str) -> str:
    """Map a canonical sample identity to one fixed-size path-safe basename."""

    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def _required_trimmed(info_path: Path, payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{info_path}: {field} must be an explicit trimmed non-empty string"
        )
    return value


def _dataset_roots(root: Path, dataset_glob: str) -> list[Path]:
    root = resolve_existing(root, directory=True)
    if (root / "meta/info.json").is_file():
        return [root]
    patterns = [part.strip() for part in dataset_glob.split(",") if part.strip()]
    if not patterns:
        raise ValueError("dataset.dataset_glob must contain at least one pattern")
    return [
        child
        for child in sorted(root.iterdir())
        if child.is_dir()
        and any(fnmatch.fnmatch(child.name, pattern) for pattern in patterns)
        and (child / "meta/info.json").is_file()
    ]


def _view_name(video_key: str) -> str:
    return video_key.rsplit(".", 1)[-1]


def _real_video_key(info_path: Path, info: dict[str, Any], view: str) -> str:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{info_path}: features must be an object")
    matches: list[str] = []
    for key, spec in features.items():
        if not isinstance(key, str) or not isinstance(spec, dict):
            continue
        if spec.get("dtype") != "video" or key.endswith("_sim"):
            continue
        if key == view or _view_name(key) == view:
            matches.append(key)
    if not matches:
        raise ValueError(f"{info_path}: required Real view {view!r} is unavailable")
    if len(matches) > 1:
        raise ValueError(
            f"{info_path}: Real view {view!r} is ambiguous across {sorted(matches)}"
        )
    real_key = matches[0]
    sim_key = f"{real_key}_sim"
    sim_spec = features.get(sim_key)
    if not isinstance(sim_spec, dict) or sim_spec.get("dtype") != "video":
        raise ValueError(
            f"{info_path}: paired Sim view {sim_key!r} is unavailable; "
            "the pipeline reads only Real pixels but requires paired metadata"
        )
    return real_key


def _video_path(
    dataset_root: Path,
    template: str,
    chunks_size: int,
    video_key: str,
    episode_index: int,
) -> Path:
    try:
        relative = template.format(
            episode_chunk=episode_index // chunks_size,
            video_key=video_key,
            episode_index=episode_index,
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid video_path template: {template!r}") from error
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"video_path must stay inside the dataset: {relative!r}")
    return resolve_existing(dataset_root / path)


def load_metadata_manifest(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load optional metadata overrides keyed by canonical ``sample_id``."""

    if path is None:
        return {}
    resolved = resolve_existing(path)
    if resolved.suffix == ".jsonl":
        values: Any = read_jsonl(resolved)
    else:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        values = (
            payload.get("samples", [payload]) if isinstance(payload, dict) else payload
        )
    if not isinstance(values, list):
        raise ValueError(f"Metadata manifest must contain a list: {resolved}")
    result: dict[str, dict[str, Any]] = {}
    for line_number, value in enumerate(values, 1):
        if not isinstance(value, dict):
            raise ValueError(f"{resolved}:{line_number}: expected a metadata object")
        sample_id = value.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id != sample_id.strip()
        ):
            raise ValueError(
                f"{resolved}:{line_number}: sample_id must be trimmed and non-empty"
            )
        if sample_id in result:
            raise ValueError(
                f"{resolved}:{line_number}: duplicate sample_id {sample_id!r}"
            )
        if "split" in value:
            raise ValueError(
                f"{resolved}:{line_number}: metadata cannot override split; use "
                "dataset.split_manifest"
            )
        result[sample_id] = {
            key: item for key, item in value.items() if key != "sample_id"
        }
    return result


@dataclass(frozen=True)
class SplitAssignments:
    domains: dict[str, Split]

    def resolve(self, domain: str) -> Split | None:
        return self.domains.get(domain)


def load_split_manifest(path: str | Path | None) -> SplitAssignments:
    """Load exact ``{domain, split}`` assignments from JSON or JSONL."""

    if path is None:
        return SplitAssignments(domains={})
    resolved = resolve_existing(path)
    if resolved.suffix == ".jsonl":
        rows: Any = read_jsonl(resolved)
    else:
        payload = read_json(resolved)
        if set(payload) != {"assignments"}:
            raise ValueError(
                f"{resolved}: expected exactly one top-level assignments field"
            )
        rows = payload["assignments"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{resolved}: assignments must be a non-empty list")
    domains: dict[str, Split] = {}
    for line_number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"domain", "split"}:
            raise ValueError(
                f"{resolved}:{line_number}: expected exactly domain and split"
            )
        domain, split = row["domain"], row["split"]
        if not isinstance(domain, str) or not domain or domain != domain.strip():
            raise ValueError(f"{resolved}:{line_number}: invalid domain")
        if split not in _VALID_SPLITS:
            raise ValueError(f"{resolved}:{line_number}: invalid split {split!r}")
        if domain in domains:
            raise ValueError(f"{resolved}:{line_number}: duplicate domain {domain!r}")
        domains[domain] = split
    return SplitAssignments(domains=domains)


def _split_selection_includes(
    name: str,
    selection: object,
    episode_index: int,
) -> bool:
    if isinstance(selection, str):
        components = [item.strip() for item in selection.split(",")]
        if not components or any(not item for item in components):
            raise ValueError(
                f"invalid info.splits selector for {name!r}: {selection!r}"
            )
        included = False
        for item in components:
            match = re.fullmatch(r"(\d+):(\d+)", item)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                if end <= start:
                    raise ValueError(
                        f"invalid info.splits interval for {name!r}: {item!r}"
                    )
                included = included or start <= episode_index < end
            elif item.isdigit():
                included = included or int(item) == episode_index
            else:
                raise ValueError(f"invalid info.splits selector for {name!r}: {item!r}")
        return included
    if isinstance(selection, list):
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in selection
        ):
            raise ValueError(
                f"invalid info.splits episode list for {name!r}: {selection!r}"
            )
        return episode_index in selection
    raise ValueError(
        f"info.splits selector for {name!r} must be a range string or integer list"
    )


def _source_split_for_episode(info: dict[str, Any], episode_index: int) -> Split | None:
    """Resolve common LeRobot ``info.splits`` ranges without guessing aliases."""

    splits = info.get("splits")
    if splits is None:
        return None
    if not isinstance(splits, dict):
        raise ValueError("meta/info.json:splits must be an object")
    matched: list[Split] = []
    for name, selection in splits.items():
        if name not in _VALID_SPLITS:
            continue
        if _split_selection_includes(name, selection, episode_index):
            matched.append(name)  # type: ignore[arg-type]
    if len(matched) > 1:
        raise ValueError(f"episode {episode_index} belongs to multiple source splits")
    return matched[0] if matched else None


def _episode_split(
    path: Path,
    line_number: int,
    row: dict[str, Any],
    info: dict[str, Any],
    domain: str,
    assignments: SplitAssignments,
) -> Split:
    row_split = row.get("split")
    if row_split is not None and row_split not in _VALID_SPLITS:
        raise ValueError(f"{path}:{line_number}: invalid split {row_split!r}")
    info_split = _source_split_for_episode(info, int(row["episode_index"]))
    external = assignments.resolve(domain)
    candidates = [
        value for value in (row_split, info_split, external) if value is not None
    ]
    if len(set(candidates)) > 1:
        raise ValueError(
            f"{path}:{line_number}: row/info/external splits conflict: {candidates}"
        )
    if not candidates:
        raise ValueError(
            f"{path}:{line_number}: no source split or external domain assignment"
        )
    return candidates[0]  # type: ignore[return-value]


def _subtasks_by_episode(dataset_root: Path) -> dict[int, tuple[dict[str, Any], ...]]:
    path = dataset_root / "labels/labels.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    values = payload.get("labels", [])
    if not isinstance(values, list):
        raise ValueError(f"{path}: labels must be a list")
    result: dict[int, tuple[dict[str, Any], ...]] = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(
            value.get("episode_index"), int
        ):
            continue
        subtasks = value.get("subtasks") or []
        if isinstance(subtasks, list) and all(
            isinstance(item, dict) for item in subtasks
        ):
            result[value["episode_index"]] = tuple(subtasks)
    return result


@dataclass(frozen=True)
class _DatasetSource:
    root: Path
    info: dict[str, Any]
    source_id: str
    domain: str
    real_video_key: str
    episodes: list[dict[str, Any]]


def _resolve_sources(
    config: DatasetConfig,
    assignments: SplitAssignments,
) -> list[_DatasetSource]:
    sources: list[_DatasetSource] = []
    source_roots: dict[str, Path] = {}
    domain_splits: dict[str, Split] = {}
    for root in _dataset_roots(config.root, config.dataset_glob):
        info_path = root / "meta/info.json"
        info = read_json(info_path)
        source_id = _required_trimmed(info_path, info, "source_id")
        domain = _required_trimmed(info_path, info, "domain")
        previous_root = source_roots.get(source_id)
        if previous_root is not None:
            raise ValueError(
                f"source_id {source_id!r} is shared by {previous_root} and {root}"
            )
        source_roots[source_id] = root
        real_video_key = _real_video_key(info_path, info, config.real_view)
        episode_path = root / "meta/episodes.jsonl"
        episodes = read_jsonl(episode_path)
        if not episodes:
            raise ValueError(f"{episode_path}: expected at least one episode")
        episode_ids: set[int] = set()
        local_split: Split | None = None
        for line_number, row in enumerate(episodes, 1):
            index = row.get("episode_index")
            length = row.get("length")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index in episode_ids
            ):
                raise ValueError(
                    f"{episode_path}:{line_number}: episode_index must be unique "
                    "and non-negative"
                )
            episode_ids.add(index)
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or length < config.min_episode_frames
            ):
                raise ValueError(
                    f"{episode_path}:{line_number}: episode {index} must contain at "
                    f"least {config.min_episode_frames} frames"
                )
            split = _episode_split(
                episode_path, line_number, row, info, domain, assignments
            )
            row["_effective_split"] = split
            if local_split is not None and split != local_split:
                raise ValueError(
                    f"{episode_path}:{line_number}: domain {domain!r} mixes "
                    f"{local_split!r} and {split!r} splits"
                )
            local_split = split
        assert local_split is not None
        previous_split = domain_splits.get(domain)
        if previous_split is not None and previous_split != local_split:
            raise ValueError(
                f"domain {domain!r} mixes {previous_split!r} and "
                f"{local_split!r} splits across datasets"
            )
        domain_splits[domain] = local_split
        sources.append(
            _DatasetSource(
                root=root,
                info=info,
                source_id=source_id,
                domain=domain,
                real_video_key=real_video_key,
                episodes=episodes,
            )
        )
    unmatched = sorted(set(assignments.domains) - set(domain_splits))
    if unmatched:
        raise ValueError(f"split manifest contains unmatched domains: {unmatched}")
    return sources


def discover_episodes(
    config: DatasetConfig,
    *,
    episodes: set[int] | None = None,
    limit: int | None = None,
) -> list[EpisodeRecord]:
    """Discover validated episodes without opening Parquet or decoding video."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    manifest = load_metadata_manifest(config.metadata_manifest)
    assignments = load_split_manifest(config.split_manifest)
    result: list[EpisodeRecord] = []
    sample_ids: set[str] = set()
    matched_manifest_ids: set[str] = set()
    for source in _resolve_sources(config, assignments):
        info = source.info
        chunks_size = int(info.get("chunks_size", 1000))
        if chunks_size < 1:
            raise ValueError(f"{source.root / 'meta/info.json'}: invalid chunks_size")
        video_template = str(
            info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4",
            )
        )
        fps = float(info.get("fps", 0.0))
        if fps <= 0:
            raise ValueError(f"{source.root / 'meta/info.json'}: fps must be positive")
        subtasks = _subtasks_by_episode(source.root)
        for row in source.episodes:
            episode_index = int(row["episode_index"])
            if episodes is not None and episode_index not in episodes:
                continue
            sample_id = canonical_episode_id(source.source_id, episode_index)
            if sample_id in sample_ids:
                raise ValueError(f"duplicate canonical sample_id {sample_id!r}")
            sample_ids.add(sample_id)
            tasks = row.get("tasks") or []
            if (
                not isinstance(tasks, list)
                or not tasks
                or not isinstance(tasks[0], str)
            ):
                raise ValueError(
                    f"{source.root / 'meta/episodes.jsonl'}: episode "
                    f"{episode_index} has no task description"
                )
            task = tasks[0].strip()
            if not task:
                raise ValueError(f"{sample_id}: task description is empty")
            metadata = dict(manifest.get(sample_id, {}))
            if sample_id in manifest:
                matched_manifest_ids.add(sample_id)
            metadata.setdefault(
                "metadata_evidence",
                {
                    "robot": "meta/info.json:robot_type",
                    "task": "meta/episodes.jsonl:tasks",
                    "actions": "labels/labels.json:subtasks"
                    if episode_index in subtasks
                    else None,
                },
            )
            result.append(
                EpisodeRecord(
                    sample_id=sample_id,
                    source_id=source.source_id,
                    dataset_name=source.root.name,
                    domain=source.domain,
                    dataset_root=source.root.resolve(),
                    episode_index=episode_index,
                    episode_length=int(row["length"]),
                    split=row["_effective_split"],
                    fps=fps,
                    robot_type=str(info["robot_type"])
                    if info.get("robot_type")
                    else None,
                    task=task,
                    subtasks=subtasks.get(episode_index, ()),
                    real_view=config.real_view,
                    real_video=_video_path(
                        source.root,
                        video_template,
                        chunks_size,
                        source.real_video_key,
                        episode_index,
                    ),
                    metadata=metadata,
                )
            )
            if limit is not None and len(result) >= limit:
                return result
    unmatched_metadata = sorted(set(manifest) - matched_manifest_ids)
    if unmatched_metadata and episodes is None and limit is None:
        raise ValueError(
            f"metadata manifest contains unmatched sample_ids: {unmatched_metadata}"
        )
    return result


def preflight_dataset(config: DatasetConfig) -> dict[str, Any]:
    """Return a cheap discovery report; intentionally performs no video decoding."""

    records = discover_episodes(config)
    by_split = {
        name: sum(record.split == name for record in records) for name in _VALID_SPLITS
    }
    return {
        "dataset_root": str(config.root.resolve()),
        "real_view": config.real_view,
        "episode_count": len(records),
        "source_count": len({record.source_id for record in records}),
        "domain_count": len({record.domain for record in records}),
        "split_counts": dict(sorted(by_split.items())),
    }


# Transitional spelling for callers moving from the old module.
discover_samples = discover_episodes
