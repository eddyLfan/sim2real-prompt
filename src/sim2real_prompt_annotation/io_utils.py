"""Deterministic JSON, hashing, path-safety, and atomic-write helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def resolve_existing(path: str | Path, *, directory: bool = False) -> Path:
    """Return an existing absolute path or raise a precise error."""

    requested = Path(path).expanduser()
    if requested.is_dir() if directory else requested.is_file():
        return requested.resolve()
    kind = "directory" if directory else "file"
    raise FileNotFoundError(f"Missing {kind}: {requested}")


def resolve_inside(root: str | Path, relative: str | Path) -> Path:
    """Resolve a relative artifact path and reject dataset-root escapes."""

    root_path = Path(root).expanduser().resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"Expected a relative path, got: {relative_path}")
    result = (root_path / relative_path).resolve()
    try:
        result.relative_to(root_path)
    except ValueError as error:
        raise ValueError(f"Path escapes root {root_path}: {relative_path}") from error
    return result


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_existing(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {resolved}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_existing(path)
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{resolved}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize stable UTF-8 JSON for cache keys and byte-identical manifests."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with resolve_existing(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(parts: Mapping[str, Any]) -> str:
    """Return a versionable SHA-256 cache key from named, JSON-safe inputs."""

    return sha256_bytes(canonical_json_bytes(dict(parts)))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    _atomic_write(Path(path), payload)


def atomic_write_text(path: str | Path, value: str) -> None:
    _atomic_write(Path(path), value.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.tmp.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(dict(row)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            temporary.chmod(destination.stat().st_mode & 0o777)
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
