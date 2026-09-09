"""Branch checkpoints and the sole publisher for Transfer training artifacts."""

from __future__ import annotations

import fcntl
import json
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
    Detection,
    EpisodePromptRow,
    EpisodeRecord,
    EpisodeReferenceRow,
    PromptResult,
    ReferenceArtifact,
    ReferenceBranchResult,
)

_CHECKPOINT_SCHEMA_VERSION = 2


def _artifact_stem(sample_id: str) -> str:
    return sha256_bytes(sample_id.encode("utf-8"))


def _staged_reference_path(
    root: Path,
    sample_id: str,
    cache_key: str,
    position: int,
) -> Path:
    stem = _artifact_stem(sample_id)
    return (
        root
        / "reference_images"
        / stem[:2]
        / stem
        / cache_key
        / f"reference_{position:02d}.jpg"
    )


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
    """Persist compact, fingerprinted per-branch checkpoints."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def prompt_path(self, sample_id: str) -> Path:
        stem = _artifact_stem(sample_id)
        return self.root / "prompt" / stem[:2] / f"{stem}.json"

    def reference_path(self, sample_id: str) -> Path:
        stem = _artifact_stem(sample_id)
        return self.root / "reference" / stem[:2] / f"{stem}.json"

    def save_prompt(self, result: PromptResult, *, cache_key: str) -> None:
        atomic_write_json(
            self.prompt_path(result.sample_id),
            {
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "cache_key": cache_key,
                "result": result.model_dump(mode="json"),
            },
        )

    def load_prompt(self, sample_id: str, *, cache_key: str) -> PromptResult | None:
        path = self.prompt_path(sample_id)
        if not path.is_file():
            return None
        try:
            payload = read_json(path)
            if (
                payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
                or payload.get("cache_key") != cache_key
            ):
                return None
            result = PromptResult.model_validate(payload.get("result"))
            return result if result.sample_id == sample_id else None
        except (OSError, ValueError, TypeError):
            return None

    def save_reference(
        self,
        record: EpisodeRecord,
        result: ReferenceBranchResult,
        *,
        cache_key: str,
    ) -> None:
        del record  # Dataset publication is a separate, locked transaction.
        for position, artifact in enumerate(result.selected_artifacts):
            atomic_write_bytes(
                _staged_reference_path(
                    self.root, result.sample_id, cache_key, position
                ),
                artifact.jpeg,
            )
        atomic_write_json(
            self.reference_path(result.sample_id),
            {
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "cache_key": cache_key,
                "sample_id": result.sample_id,
                "input_fingerprint": result.input_fingerprint,
                "candidate_pool": [
                    detection.model_dump(mode="json")
                    for detection in result.candidate_pool
                ],
                "references": [artifact.row for artifact in result.selected_artifacts],
                "rejected_reasons": list(result.rejected_reasons),
            },
        )

    def load_reference(
        self,
        record: EpisodeRecord,
        *,
        cache_key: str,
    ) -> ReferenceBranchResult | None:
        path = self.reference_path(record.sample_id)
        if not path.is_file():
            return None
        try:
            payload = read_json(path)
            if (
                payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
                or payload.get("cache_key") != cache_key
                or payload.get("sample_id") != record.sample_id
            ):
                return None
            detections = tuple(
                Detection.model_validate(item)
                for item in payload.get("candidate_pool", [])
            )
            artifacts: list[ReferenceArtifact] = []
            for position, row in enumerate(payload.get("references", [])):
                relative = Path(row["reference_path"])
                jpeg = _staged_reference_path(
                    self.root, record.sample_id, cache_key, position
                ).read_bytes()
                digest = sha256_bytes(jpeg)
                if row.get("sha256") != digest:
                    return None
                role = row.get("role")
                if role is None:
                    scope = row.get("scope", "objects")
                    role = scope if scope in {"robot", "workspace"} else "primary"
                artifacts.append(
                    ReferenceArtifact(
                        sample_id=record.sample_id,
                        reference_id=row["reference_id"],
                        relative_path=relative,
                        jpeg=jpeg,
                        source_view=row["source_view"],
                        source_frame_index=row["source_frame_index"],
                        query=row.get("query") or row.get("label"),
                        role=role,
                        confidence=row["confidence"],
                        bbox_xyxy=tuple(row["bbox_xyxy"]),
                        crop_xyxy=tuple(row["crop_xyxy"]),
                        sha256=digest,
                        description=row.get("description"),
                        provenance=row.get("provenance", {}),
                    )
                )
            return ReferenceBranchResult(
                sample_id=record.sample_id,
                candidate_pool=detections,
                selected_artifacts=tuple(artifacts),
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


class PublishableProduct(Protocol):
    """Structural type used to avoid coupling export to the orchestrator."""

    record: EpisodeRecord
    prompt_row: EpisodePromptRow
    reference_row: EpisodeReferenceRow
    reference_cache_key: str


class DatasetPublisher:
    """Publish the stable two-table contract exactly once per dataset and run."""

    def __init__(self, config: OutputConfig):
        self.config = config

    def publish_rows(
        self,
        products: Iterable[PublishableProduct],
    ) -> tuple[int, int]:
        """Atomically publish lightweight joined rows after JPEG checkpoints exist."""

        grouped: defaultdict[Path, list[PublishableProduct]] = defaultdict(list)
        for product in products:
            grouped[product.record.dataset_root].append(product)

        dataset_count = 0
        episode_count = 0
        for dataset_root, selected in sorted(
            grouped.items(), key=lambda item: str(item[0])
        ):
            prompt_path = dataset_root / "meta" / self.config.prompt_filename
            reference_path = dataset_root / "meta" / self.config.reference_filename
            with _dataset_lock(dataset_root):
                transaction_path = (
                    dataset_root / "meta" / ".sim2real-prompt.transaction.json"
                )
                selected_sample_ids = {item.record.sample_id for item in selected}
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
                            f"Invalid interrupted-publication marker: "
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
                        "sample_ids": [item.record.sample_id for item in selected],
                        "reference_cache_keys": [
                            item.reference_cache_key for item in selected
                        ],
                    },
                )
                for product in selected:
                    record = product.record
                    reference_ids = [
                        value["reference_id"]
                        for value in product.reference_row.references
                    ]
                    if product.prompt_row.reference_ids != reference_ids:
                        raise ValueError(
                            f"{record.sample_id}: prompt/reference join order differs"
                        )
                    if product.prompt_row.episode_index != record.episode_index:
                        raise ValueError(
                            f"{record.sample_id}: prompt episode_index differs"
                        )
                    if product.reference_row.episode_index != record.episode_index:
                        raise ValueError(
                            f"{record.sample_id}: reference episode_index differs"
                        )
                    for position, reference in enumerate(
                        product.reference_row.references
                    ):
                        expected_path = (
                            Path(self.config.reference_directory)
                            / f"episode_{record.episode_index:06d}"
                            / f"reference_{position:02d}.jpg"
                        )
                        if Path(reference["reference_path"]) != expected_path:
                            raise ValueError(
                                f"{record.sample_id}: invalid Reference path at "
                                f"position {position}"
                            )
                        payload = _staged_reference_path(
                            self.config.root.resolve(),
                            record.sample_id,
                            product.reference_cache_key,
                            position,
                        ).read_bytes()
                        if sha256_bytes(payload) != reference["sha256"]:
                            raise ValueError(
                                f"{record.sample_id}: staged Reference digest differs"
                            )
                        atomic_write_bytes(
                            resolve_inside(record.dataset_root, expected_path), payload
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
                # Prompt is the final commit marker: it points only at already-written
                # and hashed Reference images/rows.
                atomic_write_jsonl(
                    prompt_path,
                    (prompt_rows[index] for index in sorted(prompt_rows)),
                )
                transaction_path.unlink()
            dataset_count += 1
            episode_count += len(selected)
        return dataset_count, episode_count
