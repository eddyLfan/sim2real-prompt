"""Fast structural audit of the schema-v3 Prompt/scene-Reference join."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import OutputConfig
from .io_utils import resolve_inside
from .models import EpisodePromptRow, EpisodeRecord, EpisodeReferenceRow
from .validation import validate_prompt_text

_REFERENCE_FIELDS = {
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
_REFERENCE_FILE_PATTERN = re.compile(r"reference_\d{2}\.jpg")


@contextmanager
def _dataset_read_lock(dataset_root: Path) -> Iterator[None]:
    """Hold the publisher's dataset lock in shared mode for a complete audit."""

    path = dataset_root / "meta" / ".sim2real-prompt.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _index_rows(
    path: Path,
    errors: list[str],
    *,
    selected_episode_ids: set[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Stream a manifest, retaining only rows in a selection when provided."""

    if not path.is_file():
        errors.append(f"missing manifest: {path}")
        return {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot open {path}: {error}")
        return {}

    rows: dict[int, dict[str, Any]] = {}
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                if selected_episode_ids is None:
                    errors.append(f"{path}:{line_number}: invalid JSON: {error}")
                continue
            if not isinstance(value, dict):
                if selected_episode_ids is None:
                    errors.append(f"{path}:{line_number}: expected a JSON object")
                continue
            episode_index = value.get("episode_index")
            if isinstance(episode_index, bool) or not isinstance(episode_index, int):
                if selected_episode_ids is None:
                    errors.append(f"{path}:{line_number}: invalid episode_index")
                continue
            if (
                selected_episode_ids is not None
                and episode_index not in selected_episode_ids
            ):
                continue
            if episode_index in rows:
                errors.append(
                    f"{path}:{line_number}: duplicate episode_index {episode_index}"
                )
                continue
            rows[episode_index] = value
    return rows


def _is_hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _audit_reference(
    *,
    dataset_root: Path,
    episode_index: int,
    value: dict[str, Any],
    expected_view: str,
    expected_width: int,
    expected_height: int,
    expected_directory: str,
    errors: list[str],
) -> str | None:
    prefix = f"{dataset_root}: episode {episode_index} reference 0"
    missing = sorted(_REFERENCE_FIELDS - set(value))
    extra = sorted(set(value) - _REFERENCE_FIELDS)
    if missing:
        errors.append(f"{prefix}: missing schema-v3 fields {missing}")
    if extra:
        errors.append(f"{prefix}: unexpected schema-v3 fields {extra}")
    if missing:
        return None

    reference_id = value.get("reference_id")
    digest = value.get("sha256")
    if not _is_hex_digest(digest):
        errors.append(f"{prefix}: sha256 must be 64 lowercase hex characters")
    if reference_id != f"sha256:{digest}":
        errors.append(f"{prefix}: reference_id does not match sha256")
    if value.get("source_frame_index") != 0:
        errors.append(f"{prefix}: source_frame_index must be exactly 0")
    if value.get("source_view") != expected_view:
        errors.append(
            f"{prefix}: source_view must be configured Real view {expected_view!r}"
        )
    if value.get("scope") != "environment":
        errors.append(f"{prefix}: scope must be 'environment'")
    if value.get("reference_kind") != "robot_removed_scene":
        errors.append(f"{prefix}: reference_kind must be 'robot_removed_scene'")
    for key in ("source_frame_sha256", "mask_sha256"):
        if not _is_hex_digest(value.get(key)):
            errors.append(f"{prefix}: {key} must be 64 lowercase hex characters")
    width = value.get("width")
    height = value.get("height")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        errors.append(f"{prefix}: width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height < 1:
        errors.append(f"{prefix}: height must be a positive integer")
    if width != expected_width or height != expected_height:
        errors.append(
            f"{prefix}: dimensions must match configured Real view "
            f"{expected_width}x{expected_height}"
        )
    mask_fraction = value.get("mask_area_fraction")
    if (
        isinstance(mask_fraction, bool)
        or not isinstance(mask_fraction, (int, float))
        or not 0.0 < mask_fraction < 1.0
    ):
        errors.append(f"{prefix}: mask_area_fraction must be in (0, 1)")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        errors.append(f"{prefix}: provenance must be a non-empty object")
    else:
        if provenance.get("operation") != "robot_removal_inpainting":
            errors.append(f"{prefix}: provenance operation is invalid")
        segmenter = provenance.get("segmenter")
        if not isinstance(segmenter, dict) or not segmenter:
            errors.append(f"{prefix}: provenance lacks segmenter identity")
        inpainter = provenance.get("inpainter")
        if not isinstance(inpainter, dict) or not inpainter:
            errors.append(f"{prefix}: provenance lacks inpainter identity")
        final_mask = provenance.get("final_mask")
        if not isinstance(final_mask, dict):
            errors.append(f"{prefix}: provenance lacks final_mask")
        else:
            final_area = final_mask.get("area_fraction")
            if final_mask.get("sha256") != value.get("mask_sha256"):
                errors.append(f"{prefix}: final_mask sha256 differs from row")
            if (
                isinstance(final_area, bool)
                or not isinstance(final_area, (int, float))
                or not isinstance(mask_fraction, (int, float))
                or not math.isclose(final_area, mask_fraction)
            ):
                errors.append(f"{prefix}: final_mask area differs from row")
        quality_control = provenance.get("quality_control")
        if (
            not isinstance(quality_control, dict)
            or quality_control.get("outside_mask_unchanged") is not True
        ):
            errors.append(
                f"{prefix}: provenance does not prove outside-mask preservation"
            )
        residual_qa = provenance.get("residual_qa")
        expected_residual_fields = {
            "enabled",
            "detector",
            "queries",
            "area_fraction",
            "threshold",
            "pass",
        }
        if not isinstance(residual_qa, dict):
            errors.append(f"{prefix}: provenance lacks residual_qa")
        elif set(residual_qa) != expected_residual_fields:
            errors.append(f"{prefix}: residual_qa fields are incomplete or unexpected")
        else:
            enabled = residual_qa["enabled"]
            detector = residual_qa["detector"]
            queries = residual_qa["queries"]
            residual_area = residual_qa["area_fraction"]
            residual_threshold = residual_qa["threshold"]
            residual_pass = residual_qa["pass"]
            if not isinstance(enabled, bool):
                errors.append(f"{prefix}: residual_qa enabled is invalid")
            elif (enabled and (not isinstance(detector, dict) or not detector)) or (
                not enabled and detector is not None
            ):
                errors.append(f"{prefix}: residual_qa detector is inconsistent")
            if (
                not isinstance(queries, list)
                or not queries
                or any(
                    not isinstance(query, str) or not query.strip() for query in queries
                )
            ):
                errors.append(f"{prefix}: residual_qa queries are invalid")
            metrics_valid = not (
                isinstance(residual_area, bool)
                or not isinstance(residual_area, (int, float))
                or not math.isfinite(residual_area)
                or not 0.0 <= residual_area <= 1.0
                or isinstance(residual_threshold, bool)
                or not isinstance(residual_threshold, (int, float))
                or not math.isfinite(residual_threshold)
                or not 0.0 <= residual_threshold < 1.0
                or not isinstance(residual_pass, bool)
            )
            if not metrics_valid:
                errors.append(f"{prefix}: residual_qa metrics are invalid")
            elif residual_pass is not True or residual_pass != (
                residual_area <= residual_threshold
            ):
                errors.append(f"{prefix}: residual_qa did not pass")
            if isinstance(quality_control, dict) and metrics_valid:
                quality_enabled = quality_control.get("residual_check_enabled")
                quality_area = quality_control.get("residual_mask_area_fraction")
                quality_threshold = quality_control.get("max_residual_area_fraction")
                if (
                    quality_enabled is not enabled
                    or isinstance(quality_area, bool)
                    or not isinstance(quality_area, (int, float))
                    or not math.isfinite(quality_area)
                    or not 0.0 <= quality_area <= 1.0
                    or not math.isclose(quality_area, residual_area)
                    or isinstance(quality_threshold, bool)
                    or not isinstance(quality_threshold, (int, float))
                    or not math.isfinite(quality_threshold)
                    or not 0.0 <= quality_threshold < 1.0
                    or not math.isclose(quality_threshold, residual_threshold)
                ):
                    errors.append(f"{prefix}: residual_qa differs from quality_control")

    expected_path = (
        Path(expected_directory) / f"episode_{episode_index:06d}" / "reference_00.jpg"
    )
    try:
        relative_path = Path(value["reference_path"])
        if relative_path != expected_path:
            errors.append(
                f"{prefix}: expected path {expected_path.as_posix()!r}, got "
                f"{relative_path.as_posix()!r}"
            )
        image_path = resolve_inside(dataset_root, relative_path)
        payload = image_path.read_bytes()
    except (KeyError, OSError, TypeError, ValueError) as error:
        errors.append(f"{prefix}: cannot read reference image: {error}")
        return reference_id if isinstance(reference_id, str) else None
    actual_digest = hashlib.sha256(payload).hexdigest()
    if digest != actual_digest:
        errors.append(f"{prefix}: sha256 does not match image bytes")
    if not payload.startswith(b"\xff\xd8"):
        errors.append(f"{prefix}: reference image is not a JPEG")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        errors.append(f"{prefix}: reference JPEG cannot be decoded")
    elif isinstance(width, int) and isinstance(height, int):
        if decoded.shape[:2] != (height, width):
            errors.append(f"{prefix}: JPEG dimensions do not match full-scene metadata")
    return reference_id if isinstance(reference_id, str) else None


def _audit_reference_directory(
    record: EpisodeRecord, output: OutputConfig, errors: list[str]
) -> None:
    directory = resolve_inside(
        record.dataset_root,
        Path(output.reference_directory) / f"episode_{record.episode_index:06d}",
    )
    if not directory.is_dir():
        return
    stale = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file()
        and _REFERENCE_FILE_PATTERN.fullmatch(path.name)
        and path.name != "reference_00.jpg"
    )
    if stale:
        errors.append(
            f"{record.dataset_root}: episode {record.episode_index}: stale "
            f"multi-Reference files {stale}"
        )


def _audit_products_locked(
    records: list[EpisodeRecord],
    output: OutputConfig,
    *,
    show: int = 20,
    require_exact_episode_set: bool = True,
) -> dict[str, Any]:
    """Audit selected episodes without opening source videos or Parquet files."""

    grouped: dict[Path, list[EpisodeRecord]] = {}
    for record in records:
        grouped.setdefault(record.dataset_root, []).append(record)

    errors: list[str] = []
    complete = 0
    for dataset_root, dataset_records in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        transaction_path = dataset_root / "meta" / ".sim2real-prompt.transaction.json"
        if transaction_path.exists():
            errors.append(f"{transaction_path}: interrupted publication must be rerun")
        prompt_path = dataset_root / "meta" / output.prompt_filename
        reference_path = dataset_root / "meta" / output.reference_filename
        expected = {record.episode_index for record in dataset_records}
        selected_ids = None if require_exact_episode_set else expected
        prompt_rows = _index_rows(
            prompt_path, errors, selected_episode_ids=selected_ids
        )
        reference_rows = _index_rows(
            reference_path, errors, selected_episode_ids=selected_ids
        )
        if require_exact_episode_set:
            extra_prompts = sorted(set(prompt_rows) - expected)
            extra_references = sorted(set(reference_rows) - expected)
            if extra_prompts:
                errors.append(
                    f"{prompt_path}: rows for unknown episodes {extra_prompts}"
                )
            if extra_references:
                errors.append(
                    f"{reference_path}: rows for unknown episodes {extra_references}"
                )
        prompt_only = sorted(set(prompt_rows) - set(reference_rows))
        reference_only = sorted(set(reference_rows) - set(prompt_rows))
        if prompt_only:
            errors.append(
                f"{dataset_root}: prompt rows without reference rows {prompt_only}"
            )
        if reference_only:
            errors.append(
                f"{dataset_root}: reference rows without prompt rows {reference_only}"
            )

        for record in dataset_records:
            before = len(errors)
            prompt_value = prompt_rows.get(record.episode_index)
            reference_value = reference_rows.get(record.episode_index)
            if prompt_value is None:
                errors.append(
                    f"{dataset_root}: episode {record.episode_index} has no prompt row"
                )
            if reference_value is None:
                errors.append(
                    f"{dataset_root}: episode {record.episode_index} has no "
                    "reference row"
                )
            _audit_reference_directory(record, output, errors)
            if prompt_value is None or reference_value is None:
                continue

            try:
                prompt_row = EpisodePromptRow.model_validate(prompt_value)
                normalized = validate_prompt_text(prompt_row.prompt)
                if normalized != prompt_row.prompt:
                    raise ValueError("prompt is not in canonical normalized form")
            except ValueError as error:
                errors.append(f"{prompt_path}: episode {record.episode_index}: {error}")
                continue
            try:
                reference_row = EpisodeReferenceRow.model_validate(reference_value)
            except ValueError as error:
                errors.append(
                    f"{reference_path}: episode {record.episode_index}: {error}"
                )
                continue

            # The Pydantic contract already enforces exactly one; retain the
            # explicit check so a future relaxed DTO cannot silently weaken audit.
            if reference_row.schema_version != 3 or len(reference_row.references) != 1:
                errors.append(
                    f"{reference_path}: episode {record.episode_index}: expected "
                    "schema-v3 exact-one Reference"
                )
                continue
            reference_id = _audit_reference(
                dataset_root=dataset_root,
                episode_index=record.episode_index,
                value=reference_row.references[0],
                expected_view=record.real_view,
                expected_width=record.real_frame_width,
                expected_height=record.real_frame_height,
                expected_directory=output.reference_directory,
                errors=errors,
            )
            if prompt_row.reference_ids != [reference_id]:
                errors.append(
                    f"{dataset_root}: episode {record.episode_index}: prompt "
                    "reference_ids do not exactly match the scene Reference"
                )
            if len(errors) == before:
                complete += 1

    return {
        "status": "complete" if not errors else "incomplete",
        "schema_version": 3,
        "selected_episodes": len(records),
        "complete_episodes": complete,
        "incomplete_episodes": len(records) - complete,
        "error_count": len(errors),
        "errors": errors[: max(0, show)],
        "errors_truncated": max(0, len(errors) - max(0, show)),
        "exact_episode_set_required": require_exact_episode_set,
    }


def audit_products(
    records: list[EpisodeRecord],
    output: OutputConfig,
    *,
    show: int = 20,
    require_exact_episode_set: bool = True,
) -> dict[str, Any]:
    """Audit both tables and all References under publisher-compatible locks."""

    dataset_roots = sorted({record.dataset_root for record in records}, key=str)
    with ExitStack() as locks:
        for dataset_root in dataset_roots:
            locks.enter_context(_dataset_read_lock(dataset_root))
        return _audit_products_locked(
            records,
            output,
            show=show,
            require_exact_episode_set=require_exact_episode_set,
        )
