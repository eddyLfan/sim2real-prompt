"""Independent branch checkpoints and fail-closed schema-v3 publication."""

from __future__ import annotations

import fcntl
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from .config import OutputConfig
from .io_utils import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    resolve_inside,
    sha256_bytes,
)
from .models import (
    EpisodePromptRow,
    EpisodeRecord,
    EpisodeReferenceRow,
    PromptResult,
    RobotMaskDiagnostic,
    SceneReferenceArtifact,
    SceneReferenceResult,
)

# Prompt checkpoints intentionally retain their previous envelope version. The
# payload loader extracts only current PromptResult fields, so old checkpoints
# containing now-removed ``reference_queries`` remain reusable when their key is
# otherwise still valid. Reference artifacts have incompatible semantics and must
# never be loaded from the schema-v2 crop cache.
_PROMPT_CHECKPOINT_SCHEMA_VERSION = 2
_REFERENCE_CHECKPOINT_SCHEMA_VERSION = 3
_REFERENCE_ROW_FIELDS = {
    "reference_id",
    "reference_path",
    "source_view",
    "source_frame_index",
    "scope",
    "reference_kind",
    "width",
    "height",
    "source_frame_sha256",
    "mask_sha256",
    "mask_area_fraction",
    "sha256",
    "provenance",
}


def _artifact_stem(sample_id: str) -> str:
    return sha256_bytes(sample_id.encode("utf-8"))


def _staged_reference_path(root: Path, sample_id: str, cache_key: str) -> Path:
    stem = _artifact_stem(sample_id)
    return root / "reference_images" / stem[:2] / stem / cache_key / "reference_00.jpg"


def _staged_mask_path(root: Path, sample_id: str, cache_key: str) -> Path:
    """Private pilot/debug sidecar; it is never copied into the dataset."""

    return _staged_reference_path(root, sample_id, cache_key).with_name(
        "removal_mask.png"
    )


@contextmanager
def _reference_checkpoint_lock(
    root: Path,
    sample_id: str,
    *,
    exclusive: bool,
) -> Iterator[None]:
    """Lock one sample's mutually exclusive Reference checkpoint outcome."""

    stem = _artifact_stem(sample_id)
    path = root / ".locks" / "reference" / stem[:2] / f"{stem}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _dataset_lock(dataset_root: Path) -> Iterator[None]:
    """Serialize manifest publication by independent preprocessing processes."""

    path = dataset_root / "meta" / ".sim2real-prompt.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ArtifactStore:
    """Persist compact, independently keyed per-branch checkpoints."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def prompt_path(self, sample_id: str) -> Path:
        stem = _artifact_stem(sample_id)
        return self.root / "prompt" / stem[:2] / f"{stem}.json"

    def reference_path(self, sample_id: str) -> Path:
        stem = _artifact_stem(sample_id)
        return self.root / "reference" / stem[:2] / f"{stem}.json"

    def reference_failure_path(self, sample_id: str) -> Path:
        stem = _artifact_stem(sample_id)
        return self.root / "reference_failures" / stem[:2] / f"{stem}.json"

    def save_reference_failure(
        self,
        sample_id: str,
        *,
        cache_key: str,
        failure: dict[str, Any],
    ) -> None:
        """Checkpoint deterministic Reference failures without touching Prompt."""

        with _reference_checkpoint_lock(self.root, sample_id, exclusive=True):
            # A forced rerun can use the same semantic cache key. Never let an older
            # success shadow the newer fail-closed outcome on the next resume.
            self.reference_path(sample_id).unlink(missing_ok=True)
            _staged_reference_path(self.root, sample_id, cache_key).unlink(
                missing_ok=True
            )
            _staged_mask_path(self.root, sample_id, cache_key).unlink(missing_ok=True)
            atomic_write_json(
                self.reference_failure_path(sample_id),
                {
                    "schema_version": _REFERENCE_CHECKPOINT_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "sample_id": sample_id,
                    "failure": failure,
                },
            )

    def load_reference_failure(
        self, sample_id: str, *, cache_key: str
    ) -> dict[str, Any] | None:
        try:
            with _reference_checkpoint_lock(self.root, sample_id, exclusive=False):
                path = self.reference_failure_path(sample_id)
                if not path.is_file():
                    return None
                payload = read_json(path)
                if (
                    payload.get("schema_version")
                    != _REFERENCE_CHECKPOINT_SCHEMA_VERSION
                    or payload.get("cache_key") != cache_key
                    or payload.get("sample_id") != sample_id
                    or not isinstance(payload.get("failure"), dict)
                ):
                    return None
                return payload["failure"]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def save_prompt(self, result: PromptResult, *, cache_key: str) -> None:
        atomic_write_json(
            self.prompt_path(result.sample_id),
            {
                "schema_version": _PROMPT_CHECKPOINT_SCHEMA_VERSION,
                "cache_key": cache_key,
                "result": result.model_dump(mode="json"),
            },
        )

    def load_prompt(self, sample_id: str, *, cache_key: str) -> PromptResult | None:
        """Load current or schema-compatible legacy Prompt result fields."""

        path = self.prompt_path(sample_id)
        if not path.is_file():
            return None
        try:
            payload = read_json(path)
            if (
                payload.get("schema_version") != _PROMPT_CHECKPOINT_SCHEMA_VERSION
                or payload.get("cache_key") != cache_key
            ):
                return None
            raw = payload.get("result")
            if not isinstance(raw, dict):
                return None
            # Strict models rightly reject stale extra fields at public boundaries;
            # this checkpoint migration boundary deliberately extracts only the
            # current Prompt result before validating it.
            current = {key: raw[key] for key in PromptResult.model_fields if key in raw}
            result = PromptResult.model_validate(current)
            return result if result.sample_id == sample_id else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def save_reference(
        self,
        record: EpisodeRecord,
        result: SceneReferenceResult,
        *,
        cache_key: str,
    ) -> None:
        del record  # Dataset publication is a separate, locked transaction.
        with _reference_checkpoint_lock(self.root, result.sample_id, exclusive=True):
            artifact = result.artifact
            atomic_write_bytes(
                _staged_reference_path(self.root, result.sample_id, cache_key),
                artifact.jpeg,
            )
            removal_mask_png = result.removal_mask_png
            atomic_write_bytes(
                _staged_mask_path(self.root, result.sample_id, cache_key),
                removal_mask_png,
            )
            atomic_write_json(
                self.reference_path(result.sample_id),
                {
                    "schema_version": _REFERENCE_CHECKPOINT_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "sample_id": result.sample_id,
                    "input_fingerprint": result.input_fingerprint,
                    "artifact": artifact.row,
                    "robot_masks": [
                        diagnostic.model_dump(mode="json")
                        for diagnostic in result.robot_masks
                    ],
                    "removal_mask_sha256": sha256_bytes(removal_mask_png),
                    "rejected_reasons": list(result.rejected_reasons),
                },
            )
            self.reference_failure_path(result.sample_id).unlink(missing_ok=True)

    def load_reference(
        self,
        record: EpisodeRecord,
        *,
        cache_key: str,
    ) -> SceneReferenceResult | None:
        try:
            with _reference_checkpoint_lock(
                self.root, record.sample_id, exclusive=False
            ):
                path = self.reference_path(record.sample_id)
                if not path.is_file():
                    return None
                payload = read_json(path)
                if (
                    payload.get("schema_version")
                    != _REFERENCE_CHECKPOINT_SCHEMA_VERSION
                    or payload.get("cache_key") != cache_key
                    or payload.get("sample_id") != record.sample_id
                ):
                    return None
                row = payload.get("artifact")
                if not isinstance(row, dict):
                    return None
                jpeg = _staged_reference_path(
                    self.root, record.sample_id, cache_key
                ).read_bytes()
                digest = sha256_bytes(jpeg)
                if row.get("sha256") != digest:
                    return None
                artifact = SceneReferenceArtifact(
                    sample_id=record.sample_id,
                    reference_id=row["reference_id"],
                    relative_path=Path(row["reference_path"]),
                    jpeg=jpeg,
                    source_view=row["source_view"],
                    source_frame_index=row["source_frame_index"],
                    scope=row["scope"],
                    reference_kind=row["reference_kind"],
                    width=row["width"],
                    height=row["height"],
                    source_frame_sha256=row["source_frame_sha256"],
                    mask_sha256=row["mask_sha256"],
                    mask_area_fraction=row["mask_area_fraction"],
                    sha256=digest,
                    provenance=row["provenance"],
                )
                masks = tuple(
                    RobotMaskDiagnostic.model_validate(item)
                    for item in payload.get("robot_masks", [])
                )
                mask_png = _staged_mask_path(
                    self.root, record.sample_id, cache_key
                ).read_bytes()
                if payload.get("removal_mask_sha256") != sha256_bytes(mask_png):
                    return None
                return SceneReferenceResult(
                    sample_id=record.sample_id,
                    artifact=artifact,
                    robot_masks=masks,
                    removal_mask_png=mask_png,
                    rejected_reasons=tuple(payload.get("rejected_reasons", [])),
                    input_fingerprint=payload["input_fingerprint"],
                )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None


def _read_rows_by_episode(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        episode_index = row.get("episode_index")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index in rows
        ):
            raise ValueError(f"{path}: invalid or duplicate episode_index")
        rows[episode_index] = row
    return rows


def _clean_reference_directory(
    record: EpisodeRecord, output: OutputConfig, *, keep_reference_00: bool
) -> None:
    episode_directory = resolve_inside(
        record.dataset_root,
        Path(output.reference_directory) / f"episode_{record.episode_index:06d}",
    )
    if not episode_directory.is_dir():
        return
    for path in episode_directory.iterdir():
        if (
            not path.is_file()
            or re.fullmatch(r"reference_\d{2}\.jpg", path.name) is None
        ):
            continue
        if keep_reference_00 and path.name == "reference_00.jpg":
            continue
        path.unlink()


class PublishableProduct(Protocol):
    """Structural type used to avoid coupling export to the orchestrator."""

    record: EpisodeRecord
    prompt_row: EpisodePromptRow
    reference_row: EpisodeReferenceRow
    reference_cache_key: str


class DatasetPublisher:
    """Publish the exact-one join, with Prompt manifest as the commit marker."""

    def __init__(self, config: OutputConfig):
        self.config = config

    def publish_rows(
        self,
        products: Iterable[PublishableProduct],
        *,
        records: Iterable[EpisodeRecord] | None = None,
    ) -> tuple[int, int]:
        """Atomically publish successes and remove stale rows in processed scope."""

        grouped: defaultdict[Path, list[PublishableProduct]] = defaultdict(list)
        for product in products:
            grouped[product.record.dataset_root].append(product)
        scoped: defaultdict[Path, list[EpisodeRecord]] = defaultdict(list)
        if records is None:
            for selected in grouped.values():
                for product in selected:
                    scoped[product.record.dataset_root].append(product.record)
        else:
            for record in records:
                scoped[record.dataset_root].append(record)

        dataset_count = 0
        episode_count = 0
        for dataset_root, selected_records in sorted(
            scoped.items(), key=lambda item: str(item[0])
        ):
            selected = grouped[dataset_root]
            prompt_path = dataset_root / "meta" / self.config.prompt_filename
            reference_path = dataset_root / "meta" / self.config.reference_filename
            with _dataset_lock(dataset_root):
                transaction_path = (
                    dataset_root / "meta" / ".sim2real-prompt.transaction.json"
                )
                selected_sample_ids = {item.sample_id for item in selected_records}
                if transaction_path.exists():
                    interrupted = read_json(transaction_path)
                    previous_ids = interrupted.get("sample_ids")
                    if (
                        interrupted.get("schema_version") != 1
                        or not isinstance(previous_ids, list)
                        or not all(isinstance(value, str) for value in previous_ids)
                        or len(previous_ids) != len(set(previous_ids))
                    ):
                        raise ValueError(
                            "Invalid interrupted-publication marker: "
                            f"{transaction_path}"
                        )
                    uncovered = sorted(set(previous_ids) - selected_sample_ids)
                    if uncovered:
                        raise ValueError(
                            "The previous interrupted publication must be fully "
                            f"reprocessed before another subset: {uncovered}"
                        )

                prompt_rows = _read_rows_by_episode(prompt_path)
                reference_rows = _read_rows_by_episode(reference_path)
                atomic_write_json(
                    transaction_path,
                    {
                        "schema_version": 1,
                        "sample_ids": [item.sample_id for item in selected_records],
                        "reference_cache_keys": [
                            item.reference_cache_key for item in selected
                        ],
                    },
                )

                published_indices = {
                    product.record.episode_index for product in selected
                }
                for record in selected_records:
                    if record.episode_index in published_indices:
                        continue
                    prompt_rows.pop(record.episode_index, None)
                    reference_rows.pop(record.episode_index, None)
                    _clean_reference_directory(
                        record, self.config, keep_reference_00=False
                    )

                seen_indices: set[int] = set()
                for product in selected:
                    record = product.record
                    if record.episode_index in seen_indices:
                        raise ValueError(
                            f"{record.sample_id}: duplicate publish episode_index"
                        )
                    seen_indices.add(record.episode_index)
                    references = product.reference_row.references
                    if (
                        product.reference_row.schema_version != 3
                        or len(references) != 1
                    ):
                        raise ValueError(
                            f"{record.sample_id}: schema-v3 requires exactly one "
                            "Reference"
                        )
                    reference = references[0]
                    if set(reference) != _REFERENCE_ROW_FIELDS:
                        raise ValueError(
                            f"{record.sample_id}: Reference fields do not match "
                            "schema-v3"
                        )
                    if (
                        reference.get("source_frame_index") != 0
                        or reference.get("source_view") != record.real_view
                        or reference.get("scope") != "environment"
                        or reference.get("reference_kind") != "robot_removed_scene"
                    ):
                        raise ValueError(
                            f"{record.sample_id}: invalid frame-zero scene semantics"
                        )
                    reference_id = reference.get("reference_id")
                    if reference_id != f"sha256:{reference.get('sha256')}":
                        raise ValueError(
                            f"{record.sample_id}: invalid Reference identity"
                        )
                    if product.prompt_row.reference_ids != [reference_id]:
                        raise ValueError(
                            f"{record.sample_id}: prompt/reference join differs"
                        )
                    if product.prompt_row.episode_index != record.episode_index:
                        raise ValueError(
                            f"{record.sample_id}: prompt episode_index differs"
                        )
                    if product.reference_row.episode_index != record.episode_index:
                        raise ValueError(
                            f"{record.sample_id}: reference episode_index differs"
                        )
                    expected_path = (
                        Path(self.config.reference_directory)
                        / f"episode_{record.episode_index:06d}"
                        / "reference_00.jpg"
                    )
                    if Path(reference.get("reference_path", "")) != expected_path:
                        raise ValueError(
                            f"{record.sample_id}: invalid schema-v3 Reference path"
                        )
                    payload = _staged_reference_path(
                        self.config.root.resolve(),
                        record.sample_id,
                        product.reference_cache_key,
                    ).read_bytes()
                    if sha256_bytes(payload) != reference.get("sha256"):
                        raise ValueError(
                            f"{record.sample_id}: staged Reference digest differs"
                        )
                    atomic_write_bytes(
                        resolve_inside(record.dataset_root, expected_path), payload
                    )
                    # The previous multi-Reference contract may have left 01/02;
                    # schema-v3 publication retains only the newly written 00.
                    _clean_reference_directory(
                        record, self.config, keep_reference_00=True
                    )
                    reference_rows[record.episode_index] = (
                        product.reference_row.model_dump(mode="json")
                    )
                    prompt_rows[record.episode_index] = product.prompt_row.model_dump(
                        mode="json"
                    )

                atomic_write_jsonl(
                    reference_path,
                    (reference_rows[index] for index in sorted(reference_rows)),
                )
                # Prompt is the final commit marker: it points only at an already
                # written and hashed Reference image and schema-v3 row.
                atomic_write_jsonl(
                    prompt_path,
                    (prompt_rows[index] for index in sorted(prompt_rows)),
                )
                transaction_path.unlink()
            dataset_count += 1
            episode_count += len(selected)
        return dataset_count, episode_count
