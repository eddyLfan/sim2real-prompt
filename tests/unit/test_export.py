from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import sim2real_prompt_annotation.export as export_module
from sim2real_prompt_annotation.export import ArtifactStore


def test_failure_commit_blocks_both_same_sample_readers_only(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ArtifactStore(tmp_path / "checkpoints")
    sample_id = "source:episode:0"
    other_sample_id = "source:episode:1"
    cache_key = "reference-cache-key"
    failure = {"stage": "reference", "error": "no valid scene"}
    success_path = store.reference_path(sample_id)
    success_path.parent.mkdir(parents=True, exist_ok=True)
    success_path.write_text("{}", encoding="utf-8")

    commit_entered = Event()
    allow_commit = Event()
    original_write_json = export_module.atomic_write_json

    def paused_write_json(path: str | Path, value: Any) -> None:
        if Path(path) == store.reference_failure_path(sample_id):
            commit_entered.set()
            if not allow_commit.wait(timeout=5):
                raise AssertionError("timed out waiting to finish failure commit")
        original_write_json(path, value)

    monkeypatch.setattr(export_module, "atomic_write_json", paused_write_json)

    failure_reader_started = Event()
    failure_reader_finished = Event()
    success_reader_started = Event()
    success_reader_finished = Event()

    def read_failure() -> dict[str, Any] | None:
        failure_reader_started.set()
        try:
            return store.load_reference_failure(sample_id, cache_key=cache_key)
        finally:
            failure_reader_finished.set()

    def read_success() -> Any:
        success_reader_started.set()
        try:
            record = SimpleNamespace(sample_id=sample_id)
            return store.load_reference(record, cache_key=cache_key)
        finally:
            success_reader_finished.set()

    with ThreadPoolExecutor(max_workers=4) as executor:
        writer = executor.submit(
            store.save_reference_failure,
            sample_id,
            cache_key=cache_key,
            failure=failure,
        )
        assert commit_entered.wait(timeout=5)
        failure_reader = executor.submit(read_failure)
        success_reader = executor.submit(read_success)
        assert failure_reader_started.wait(timeout=5)
        assert success_reader_started.wait(timeout=5)

        # A distinct sample has a distinct lock and remains immediately readable.
        other_reader = executor.submit(
            store.load_reference_failure,
            other_sample_id,
            cache_key=cache_key,
        )
        assert other_reader.result(timeout=5) is None
        assert not failure_reader_finished.wait(timeout=0.1)
        assert not success_reader_finished.wait(timeout=0.1)

        allow_commit.set()
        writer.result(timeout=5)
        assert success_reader.result(timeout=5) is None
        assert failure_reader.result(timeout=5) == failure


def test_success_commit_blocks_same_sample_failure_reader(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ArtifactStore(tmp_path / "checkpoints")
    sample_id = "source:episode:0"
    cache_key = "reference-cache-key"
    store.save_reference_failure(
        sample_id,
        cache_key=cache_key,
        failure={"error": "old deterministic failure"},
    )

    write_entered = Event()
    allow_write = Event()
    original_write_bytes = export_module.atomic_write_bytes

    def paused_write_bytes(path: str | Path, payload: bytes) -> None:
        original_write_bytes(path, payload)
        if Path(path).name == "reference_00.jpg":
            write_entered.set()
            if not allow_write.wait(timeout=5):
                raise AssertionError("timed out waiting to finish success commit")

    monkeypatch.setattr(export_module, "atomic_write_bytes", paused_write_bytes)
    artifact = SimpleNamespace(jpeg=b"new-jpeg", row={"sha256": "unused"})
    result = SimpleNamespace(
        sample_id=sample_id,
        artifact=artifact,
        removal_mask_png=b"new-mask",
        robot_masks=(),
        rejected_reasons=(),
        input_fingerprint="input-fingerprint",
    )
    reader_started = Event()
    reader_finished = Event()

    def read_failure() -> dict[str, Any] | None:
        reader_started.set()
        try:
            return store.load_reference_failure(sample_id, cache_key=cache_key)
        finally:
            reader_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(
            store.save_reference,
            SimpleNamespace(sample_id=sample_id),
            result,
            cache_key=cache_key,
        )
        assert write_entered.wait(timeout=5)
        reader = executor.submit(read_failure)
        assert reader_started.wait(timeout=5)
        assert not reader_finished.wait(timeout=0.1)

        allow_write.set()
        writer.result(timeout=5)
        assert reader.result(timeout=5) is None


def test_same_sample_checkpoint_readers_use_shared_locks(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ArtifactStore(tmp_path / "checkpoints")
    sample_id = "source:episode:0"
    cache_key = "reference-cache-key"
    failure = {"error": "deterministic failure"}
    store.save_reference_failure(
        sample_id,
        cache_key=cache_key,
        failure=failure,
    )

    release_reads = Event()
    both_readers_entered = Event()
    counter_lock = Lock()
    entered = 0
    original_read_json = export_module.read_json

    def paused_read_json(path: str | Path) -> dict[str, Any]:
        nonlocal entered
        if Path(path) == store.reference_failure_path(sample_id):
            with counter_lock:
                entered += 1
                if entered == 2:
                    both_readers_entered.set()
            if not release_reads.wait(timeout=5):
                raise AssertionError("timed out waiting to release checkpoint reads")
        return original_read_json(path)

    monkeypatch.setattr(export_module, "read_json", paused_read_json)

    with ThreadPoolExecutor(max_workers=2) as executor:
        readers = [
            executor.submit(
                store.load_reference_failure,
                sample_id,
                cache_key=cache_key,
            )
            for _ in range(2)
        ]
        shared = both_readers_entered.wait(timeout=2)
        release_reads.set()
        assert shared
        assert [reader.result(timeout=5) for reader in readers] == [failure, failure]
