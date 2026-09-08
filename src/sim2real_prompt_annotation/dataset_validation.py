"""Read-only validation for paired Sim/Real LeRobot v2.1 datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

Severity = Literal["error", "warning", "repairable"]


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _format_path(
    dataset: Path,
    template: str,
    chunks_size: int,
    episode_index: int,
    *,
    video_key: str | None = None,
) -> Path:
    return dataset / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def _column_numpy(table: pa.Table, name: str) -> np.ndarray:
    array = table[name].combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        return array.values.to_numpy(zero_copy_only=False).reshape(
            len(array), array.type.list_size
        )
    return np.asarray(array.to_numpy(zero_copy_only=False)).reshape(len(array), 1)


def _prompt_status(meta: Path, episode_ids: set[int]) -> dict[str, Any]:
    path = meta / "episodes_prompt.jsonl"
    if not path.is_file():
        return {"status": "missing", "path": str(path), "rows": 0}
    try:
        rows = _read_jsonl(path)
    except (OSError, ValueError) as error:
        return {"status": "invalid", "path": str(path), "rows": 0, "reason": str(error)}

    found: set[int] = set()
    for line_number, row in enumerate(rows, 1):
        episode_index = row.get("episode_index")
        prompt = row.get("prompt")
        reference_ids = row.get("reference_ids")
        if (
            not isinstance(episode_index, int)
            or episode_index in found
            or not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(reference_ids, list)
            or not reference_ids
            or any(not isinstance(value, str) or not value for value in reference_ids)
        ):
            return {
                "status": "invalid",
                "path": str(path),
                "rows": len(rows),
                "reason": (
                    f"line {line_number} does not match the current prompt schema "
                    "{episode_index,prompt,reference_ids}"
                ),
            }
        found.add(episode_index)
    missing = sorted(episode_ids - found)
    extra = sorted(found - episode_ids)
    return {
        "status": "complete" if not missing and not extra else "incomplete",
        "path": str(path),
        "rows": len(rows),
        "missing_episodes": missing,
        "extra_episodes": extra,
    }


def _reference_status(dataset: Path, episode_ids: set[int]) -> dict[str, Any]:
    directory = dataset / "Reference"
    manifest = dataset / "meta" / "reference_images.jsonl"
    if not directory.is_dir():
        return {
            "status": "missing",
            "directory": str(directory),
            "images": 0,
            "manifest": str(manifest),
        }

    found: set[int] = set()
    invalid: list[str] = []
    manifest_status = "missing"
    manifest_rows = 0
    if manifest.is_file():
        try:
            rows = _read_jsonl(manifest)
            manifest_id_list = [int(row["episode_index"]) for row in rows]
            manifest_ids = set(manifest_id_list)
            manifest_rows = len(rows)
            manifest_status = (
                "complete"
                if manifest_ids == episode_ids and len(manifest_ids) == len(rows)
                else "incomplete"
            )
            for row in rows:
                episode_index = int(row["episode_index"])
                references = row.get("references")
                if not isinstance(references, list) or not 1 <= len(references) <= 12:
                    invalid.append(
                        f"episode {episode_index}: expected 1--12 Reference images"
                    )
                    continue
                valid = True
                reference_ids: list[str] = []
                for reference in references:
                    if not isinstance(reference, dict):
                        invalid.append(
                            f"episode {episode_index}: Reference entry is not an object"
                        )
                        valid = False
                        continue
                    reference_id = reference.get("reference_id")
                    if not isinstance(reference_id, str) or not reference_id:
                        invalid.append(
                            f"episode {episode_index}: missing Reference identity"
                        )
                        valid = False
                        continue
                    reference_ids.append(reference_id)
                    if reference.get("source_frame_index") != 0:
                        invalid.append(
                            f"episode {episode_index}: Reference is not from frame zero"
                        )
                        valid = False
                    path = (
                        dataset / str(reference.get("reference_path", ""))
                    ).resolve()
                    try:
                        path.relative_to(dataset.resolve())
                    except ValueError:
                        invalid.append(
                            f"episode {episode_index}: Reference path escapes dataset"
                        )
                        valid = False
                        continue
                    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if image is None or image.size == 0:
                        invalid.append(
                            f"episode {episode_index}: invalid Reference image {path}"
                        )
                        valid = False
                        continue
                    expected_digest = reference.get("sha256")
                    if isinstance(expected_digest, str) and (
                        hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest
                    ):
                        invalid.append(
                            f"episode {episode_index}: Reference digest mismatch {path}"
                        )
                        valid = False
                if len(reference_ids) != len(set(reference_ids)):
                    invalid.append(
                        f"episode {episode_index}: duplicate Reference identities"
                    )
                    valid = False
                if valid:
                    found.add(episode_index)
        except (KeyError, TypeError, ValueError, OSError):
            manifest_status = "invalid"
    missing = sorted(episode_ids - found)
    return {
        "status": ("complete" if not missing and not invalid else "incomplete"),
        "directory": str(directory),
        "images": len(found),
        "missing_episodes": missing,
        "invalid": invalid,
        "extra_files": [],
        "manifest": str(manifest),
        "manifest_status": manifest_status,
        "manifest_rows": manifest_rows,
    }


def inspect_dataset(
    dataset: str | Path, *, probe_videos: bool = True
) -> dict[str, Any]:
    """Inspect a dataset without modifying it and return a JSON-serializable report."""

    root = Path(dataset).expanduser().resolve()
    issues: list[Issue] = []
    meta = root / "meta"
    required_meta = (
        "info.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "stats.json",
        "tasks.jsonl",
    )
    if not root.is_dir():
        issues.append(
            Issue("error", "dataset.missing", f"Dataset directory is missing: {root}")
        )
    for filename in required_meta:
        if not (meta / filename).is_file():
            issues.append(
                Issue(
                    "error",
                    "metadata.missing",
                    f"Required metadata is missing: {meta / filename}",
                )
            )
    if any(issue.severity == "error" for issue in issues):
        return {
            "dataset": str(root),
            "core_valid": False,
            "pipeline_complete": False,
            "issues": [asdict(issue) for issue in issues],
        }

    try:
        info = _read_json(meta / "info.json")
        episodes = _read_jsonl(meta / "episodes.jsonl")
        episodes_stats = _read_jsonl(meta / "episodes_stats.jsonl")
        global_stats = _read_json(meta / "stats.json")
        tasks = _read_jsonl(meta / "tasks.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        issues.append(Issue("error", "metadata.invalid", str(error)))
        return {
            "dataset": str(root),
            "core_valid": False,
            "pipeline_complete": False,
            "issues": [asdict(issue) for issue in issues],
        }

    episode_ids_list = [row.get("episode_index") for row in episodes]
    if any(not isinstance(value, int) for value in episode_ids_list):
        issues.append(
            Issue(
                "error",
                "episodes.invalid_index",
                "Every episode needs an integer index",
            )
        )
        episode_ids: set[int] = set()
    else:
        episode_ids = {int(value) for value in episode_ids_list}
        if len(episode_ids) != len(episode_ids_list):
            issues.append(
                Issue("error", "episodes.duplicate", "Duplicate episode indices found")
            )
        if sorted(episode_ids) != list(range(len(episodes))):
            issues.append(
                Issue(
                    "error",
                    "episodes.non_contiguous",
                    "Episode indices must be contiguous from zero",
                )
            )

    total_episodes = info.get("total_episodes")
    if total_episodes != len(episodes):
        issues.append(
            Issue(
                "error",
                "info.total_episodes",
                f"info.json says {total_episodes}, episodes.jsonl contains "
                f"{len(episodes)}",
            )
        )
    if info.get("total_tasks") != len(tasks):
        issues.append(
            Issue(
                "error",
                "info.total_tasks",
                f"info.json says {info.get('total_tasks')}, tasks.jsonl contains "
                f"{len(tasks)}",
            )
        )
    task_ids = [row.get("task_index") for row in tasks]
    if task_ids != list(range(len(tasks))):
        issues.append(
            Issue(
                "error",
                "tasks.indices",
                "Task indices must be unique and contiguous from zero",
            )
        )
    task_names = {row.get("task") for row in tasks if isinstance(row.get("task"), str)}
    for episode in episodes:
        episode_tasks = episode.get("tasks")
        if not isinstance(episode_tasks, list) or not episode_tasks:
            issues.append(
                Issue(
                    "error",
                    "episodes.tasks",
                    f"Episode {episode.get('episode_index')} has no task label",
                )
            )
        elif any(task not in task_names for task in episode_tasks):
            issues.append(
                Issue(
                    "error",
                    "episodes.tasks",
                    f"Episode {episode.get('episode_index')} references "
                    "an unknown task",
                )
            )

    episode_stats_ids = [row.get("episode_index") for row in episodes_stats]
    if (
        any(not isinstance(value, int) for value in episode_stats_ids)
        or len(set(episode_stats_ids)) != len(episode_stats_ids)
        or set(episode_stats_ids) != episode_ids
    ):
        issues.append(
            Issue(
                "error",
                "episodes_stats.coverage",
                "episodes_stats.jsonl must contain exactly one row per episode",
            )
        )
    if not isinstance(global_stats, dict) or not global_stats:
        issues.append(
            Issue("error", "stats.invalid", "stats.json must be a non-empty object")
        )
    if info.get("codebase_version") != "v2.1":
        issues.append(
            Issue(
                "warning",
                "info.codebase_version",
                f"Expected LeRobot v2.1, found {info.get('codebase_version')!r}",
            )
        )
    fps = float(info.get("fps", 0.0))
    if fps <= 0:
        issues.append(Issue("error", "info.fps", f"Invalid dataset FPS: {fps}"))

    features = info.get("features")
    if not isinstance(features, dict):
        features = {}
        issues.append(
            Issue("error", "info.features", "info.json features must be an object")
        )
    video_keys = sorted(
        name for name, feature in features.items() if feature.get("dtype") == "video"
    )
    sim_keys = [key for key in video_keys if key.endswith("_sim")]
    paired_video_keys = [key for key in sim_keys if key[:-4] in video_keys]
    unpaired = sorted(set(sim_keys) - set(paired_video_keys))
    real_without_sim = sorted(
        key
        for key in video_keys
        if not key.endswith("_sim") and f"{key}_sim" not in video_keys
    )
    if not paired_video_keys:
        issues.append(
            Issue("error", "videos.no_pairs", "No paired Real/Sim video features found")
        )
    if unpaired or real_without_sim:
        issues.append(
            Issue(
                "error",
                "videos.unpaired",
                f"Unpaired sim={unpaired}, real={real_without_sim}",
            )
        )

    chunks_size = int(info.get("chunks_size", 1000))
    data_template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    video_template = str(
        info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
    )
    expected_task_ids = {int(row["task_index"]) for row in tasks if "task_index" in row}
    expected_global_index = 0
    total_rows = 0
    parquet_files = 0
    video_files = 0
    decoded_videos = 0
    for episode in sorted(episodes, key=lambda row: int(row.get("episode_index", -1))):
        episode_index = episode.get("episode_index")
        length = episode.get("length")
        if (
            not isinstance(episode_index, int)
            or not isinstance(length, int)
            or length <= 0
        ):
            issues.append(
                Issue(
                    "error",
                    "episodes.invalid_row",
                    f"Invalid episode metadata row: {episode}",
                )
            )
            continue
        parquet_path = _format_path(
            root, data_template, chunks_size, episode_index, video_key=None
        )
        if not parquet_path.is_file() or parquet_path.stat().st_size == 0:
            issues.append(
                Issue(
                    "error",
                    "parquet.missing",
                    f"Missing or empty parquet: {parquet_path}",
                )
            )
            continue
        parquet_files += 1
        try:
            parquet = pq.ParquetFile(parquet_path)
            if parquet.metadata.num_rows != length:
                issues.append(
                    Issue(
                        "error",
                        "parquet.length",
                        (
                            f"{parquet_path}: rows={parquet.metadata.num_rows}, "
                            f"metadata length={length}"
                        ),
                    )
                )
            available = set(parquet.schema_arrow.names)
            required_columns = {"episode_index", "frame_index", "index", "task_index"}
            required_columns.update(
                name
                for name, feature in features.items()
                if feature.get("dtype") != "video" and "pose" not in name
            )
            missing_required = required_columns - available
            if missing_required:
                issues.append(
                    Issue(
                        "error",
                        "parquet.columns",
                        f"{parquet_path}: missing {sorted(missing_required)}",
                    )
                )
                continue
            table = pq.read_table(parquet_path, columns=sorted(required_columns))
        except (OSError, ValueError, pa.ArrowException) as error:
            issues.append(Issue("error", "parquet.invalid", f"{parquet_path}: {error}"))
            continue

        episode_values = _column_numpy(table, "episode_index").reshape(-1)
        frame_values = _column_numpy(table, "frame_index").reshape(-1)
        index_values = _column_numpy(table, "index").reshape(-1)
        task_values = _column_numpy(table, "task_index").reshape(-1)
        if not np.all(episode_values == episode_index):
            issues.append(
                Issue(
                    "error",
                    "parquet.episode_index",
                    f"{parquet_path}: inconsistent episode_index",
                )
            )
        if not np.array_equal(frame_values, np.arange(len(table))):
            issues.append(
                Issue(
                    "error",
                    "parquet.frame_index",
                    f"{parquet_path}: frame_index is not contiguous",
                )
            )
        if not np.array_equal(
            index_values,
            np.arange(expected_global_index, expected_global_index + len(table)),
        ):
            issues.append(
                Issue(
                    "error",
                    "parquet.global_index",
                    f"{parquet_path}: global index is not contiguous",
                )
            )
        if not set(int(value) for value in np.unique(task_values)) <= expected_task_ids:
            issues.append(
                Issue(
                    "error",
                    "parquet.task_index",
                    f"{parquet_path}: unknown task_index value",
                )
            )
        expected_global_index += len(table)
        total_rows += len(table)

        for video_key in video_keys:
            path = _format_path(
                root,
                video_template,
                chunks_size,
                episode_index,
                video_key=video_key,
            )
            if not path.is_file() or path.stat().st_size == 0:
                issues.append(
                    Issue("error", "video.missing", f"Missing or empty video: {path}")
                )
                continue
            video_files += 1
            if not probe_videos:
                continue
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                issues.append(
                    Issue("error", "video.decode", f"Cannot open video: {path}")
                )
                continue
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            decoded_videos += 1
            if frame_count != length:
                issues.append(
                    Issue(
                        "error",
                        "video.length",
                        f"{path}: frames={frame_count}, episode length={length}",
                    )
                )
            if video_fps <= 0 or not np.isclose(video_fps, fps, atol=0.01):
                issues.append(
                    Issue(
                        "error",
                        "video.fps",
                        f"{path}: fps={video_fps}, dataset fps={fps}",
                    )
                )

    if total_rows != info.get("total_frames"):
        issues.append(
            Issue(
                "error",
                "info.total_frames",
                f"info.json says {info.get('total_frames')}, parquet rows total "
                f"{total_rows}",
            )
        )

    prompt = _prompt_status(meta, episode_ids)
    if prompt["status"] != "complete":
        issues.append(
            Issue(
                "repairable",
                f"prompt.{prompt['status']}",
                (
                    "Prompt metadata requires generation or repair: "
                    f"{prompt.get('reason', prompt['path'])}"
                ),
            )
        )
    reference = _reference_status(root, episode_ids)
    if reference["status"] != "complete":
        issues.append(
            Issue(
                "repairable",
                f"reference.{reference['status']}",
                f"Reference images require generation: {reference['directory']}",
            )
        )
    elif reference.get("manifest_status") != "complete":
        issues.append(
            Issue(
                "warning",
                "reference.manifest",
                "Reference images exist but reference_images.jsonl is missing "
                "or incomplete",
            )
        )

    if (
        prompt["status"] == "complete"
        and reference.get("manifest_status") == "complete"
    ):
        prompt_rows = {
            int(row["episode_index"]): row
            for row in _read_jsonl(meta / "episodes_prompt.jsonl")
        }
        reference_rows = {
            int(row["episode_index"]): row
            for row in _read_jsonl(meta / "reference_images.jsonl")
        }
        mismatches = sorted(
            episode_index
            for episode_index in episode_ids
            if prompt_rows[episode_index]["reference_ids"]
            != [
                reference.get("reference_id")
                for reference in reference_rows[episode_index].get("references", [])
            ]
        )
        if mismatches:
            prompt["status"] = "inconsistent"
            prompt["mismatched_reference_episodes"] = mismatches
            issues.append(
                Issue(
                    "repairable",
                    "prompt.reference_mismatch",
                    f"Prompt and Reference identity differs for episodes {mismatches}",
                )
            )

    core_valid = not any(issue.severity == "error" for issue in issues)
    pipeline_complete = (
        core_valid
        and prompt["status"] == "complete"
        and reference["status"] == "complete"
    )
    return {
        "dataset": str(root),
        "codebase_version": info.get("codebase_version"),
        "counts": {
            "tasks": len(tasks),
            "episodes": len(episodes),
            "frames": total_rows,
            "parquet_files": parquet_files,
            "video_features": len(video_keys),
            "video_files": video_files,
            "decoded_videos": decoded_videos,
        },
        "paired_real_sim_views": len(paired_video_keys),
        "core_valid": core_valid,
        "pipeline_complete": pipeline_complete,
        "prompt": prompt,
        "reference": reference,
        "issues": [asdict(issue) for issue in issues],
    }
