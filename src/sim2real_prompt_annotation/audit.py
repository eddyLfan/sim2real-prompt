"""Fast structural audit of the two manifests consumed by Transfer training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import MAX_REFERENCE_IMAGES, OutputConfig
from .io_utils import resolve_inside
from .models import EpisodePromptRow, EpisodeRecord, EpisodeReferenceRow
from .validation import validate_prompt_text


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


def _audit_reference(
    *,
    dataset_root: Path,
    episode_index: int,
    position: int,
    value: dict[str, Any],
    expected_view: str,
    expected_directory: str,
    errors: list[str],
) -> str | None:
    prefix = f"{dataset_root}: episode {episode_index} reference {position}"
    required = {
        "reference_id",
        "reference_path",
        "source_view",
        "source_frame_index",
        "query",
        "label",
        "role",
        "scope",
        "confidence",
        "bbox_xyxy",
        "crop_xyxy",
        "sha256",
    }
    missing = sorted(required - set(value))
    if missing:
        errors.append(f"{prefix}: missing fields {missing}")
        return None

    reference_id = value.get("reference_id")
    digest = value.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        errors.append(f"{prefix}: sha256 must be 64 lowercase hex characters")
    if reference_id != f"sha256:{digest}":
        errors.append(f"{prefix}: reference_id does not match sha256")
    if value.get("source_frame_index") != 0:
        errors.append(f"{prefix}: source_frame_index must be 0")
    if value.get("source_view") != expected_view:
        errors.append(
            f"{prefix}: source_view must be configured Real view {expected_view!r}"
        )
    query = value.get("query")
    if not isinstance(query, str) or not query.strip():
        errors.append(f"{prefix}: query must be a non-empty string")
    if value.get("label") != query:
        errors.append(f"{prefix}: label must equal detector query")
    role = value.get("role")
    allowed_roles = {
        "primary",
        "destination",
        "secondary",
        "robot",
        "workspace",
        "environment",
        "background",
    }
    if role not in allowed_roles:
        errors.append(f"{prefix}: invalid role {role!r}")
    expected_scope = (
        role
        if role in {"robot", "workspace", "environment", "background"}
        else "objects"
    )
    if value.get("scope") != expected_scope:
        errors.append(f"{prefix}: scope is inconsistent with role {role!r}")
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= confidence <= 1.0
    ):
        errors.append(f"{prefix}: confidence must be in [0, 1]")
    bbox = value.get("bbox_xyxy")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in bbox
        )
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
    ):
        errors.append(f"{prefix}: bbox_xyxy must be a positive numeric box")
    crop = value.get("crop_xyxy")
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in crop)
        or min(crop, default=-1) < 0
        or crop[2] <= crop[0]
        or crop[3] <= crop[1]
    ):
        errors.append(f"{prefix}: crop_xyxy must be a positive integer box")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("backend") != "yoloe":
        errors.append(f"{prefix}: provenance.backend must be 'yoloe'")

    expected_path = (
        Path(expected_directory)
        / f"episode_{episode_index:06d}"
        / f"reference_{position:02d}.jpg"
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
    elif (
        isinstance(crop, list)
        and len(crop) == 4
        and all(isinstance(item, int) and not isinstance(item, bool) for item in crop)
        and decoded.shape[:2] != (crop[3] - crop[1], crop[2] - crop[0])
    ):
        errors.append(f"{prefix}: JPEG dimensions do not match crop_xyxy")
    return reference_id if isinstance(reference_id, str) else None


def audit_products(
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
            prompt_path,
            errors,
            selected_episode_ids=selected_ids,
        )
        reference_rows = _index_rows(
            reference_path,
            errors,
            selected_episode_ids=selected_ids,
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

            if not 1 <= len(reference_row.references) <= MAX_REFERENCE_IMAGES:
                errors.append(
                    f"{reference_path}: episode {record.episode_index}: expected "
                    f"1-{MAX_REFERENCE_IMAGES} references"
                )
            ids: list[str] = []
            roles: list[str] = []
            for position, reference in enumerate(reference_row.references):
                reference_id = _audit_reference(
                    dataset_root=dataset_root,
                    episode_index=record.episode_index,
                    position=position,
                    value=reference,
                    expected_view=record.real_view,
                    expected_directory=output.reference_directory,
                    errors=errors,
                )
                if reference_id is not None:
                    ids.append(reference_id)
                role = reference.get("role")
                if isinstance(role, str):
                    roles.append(role)
            if prompt_row.reference_ids != ids:
                errors.append(
                    f"{dataset_root}: episode {record.episode_index}: prompt "
                    "reference_ids do not exactly match reference row order"
                )
            if len(ids) != len(set(ids)):
                errors.append(
                    f"{dataset_root}: episode {record.episode_index}: duplicate "
                    "reference_id"
                )
            if "primary" not in roles:
                errors.append(
                    f"{dataset_root}: episode {record.episode_index}: no primary "
                    "task-object reference"
                )
            if len(errors) == before:
                complete += 1

    return {
        "status": "complete" if not errors else "incomplete",
        "selected_episodes": len(records),
        "complete_episodes": complete,
        "incomplete_episodes": len(records) - complete,
        "error_count": len(errors),
        "errors": errors[: max(0, show)],
        "errors_truncated": max(0, len(errors) - max(0, show)),
        "exact_episode_set_required": require_exact_episode_set,
    }
