"""Resumable orchestration for independent Prompt and scene-Reference branches."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from openai import APIConnectionError, APITimeoutError

from .audit import audit_products
from .config import PipelineConfig
from .dataset import discover_episodes
from .export import ArtifactStore, DatasetPublisher
from .io_utils import atomic_write_json, fingerprint, resolve_inside, sha256_bytes
from .models import (
    EpisodePromptRow,
    EpisodeRecord,
    EpisodeReferenceRow,
    PromptResult,
    RealFrameBundle,
    SceneReferenceResult,
)
from .prompt_branch import PromptBranch
from .qwen import QwenOpenAIClient, ResponseParseError, VLMClient
from .reference_branch import (
    NoValidReferenceError,
    ReferenceBranch,
    ReferenceBranchInput,
)
from .validation import (
    ProductValidationError,
    validate_prompt_result,
    validate_reference_result,
)
from .video import decode_real_first_frame, decode_real_video

# The live branches have separate schema tokens, so changing either implementation
# cannot invalidate the other branch's checkpoint.
_PROMPT_CACHE_SCHEMA = "real8-task-prompt-only-v3"
_REFERENCE_CACHE_SCHEMA = "frame0-robot-removed-scene-v3"
# Schema-v2 Prompt checkpoints hashed the previous prompt-and-query instruction.
# Keeping the digest here is a narrow, reviewable one-release migration path; no
# legacy Reference behavior is retained.
_LEGACY_PROMPT_CACHE_SCHEMA = "real8-yoloe-multiframe-multiref-v2"
_LEGACY_PROMPT_SYSTEM_SHA256 = (
    "10ec161df8aabf955a37a2f23af132fe7945d56c7ac26caa39903f0ce5601203"
)
_LEGACY_PROMPT_MAX_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class PublishedProduct:
    """Joined publication DTO; the two branch checkpoints remain independent."""

    record: EpisodeRecord
    prompt_row: EpisodePromptRow
    reference_row: EpisodeReferenceRow
    reference_cache_key: str


def _chunks(values: list[EpisodeRecord], size: int) -> Iterable[list[EpisodeRecord]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _prompt_metadata(record: EpisodeRecord) -> dict[str, Any]:
    """Keep authoritative semantics small; visual context stays in the frames."""

    payload: dict[str, Any] = {
        "robot_type": record.robot_type,
        "subtasks": list(record.subtasks),
    }
    supplemental = {
        key: value
        for key, value in record.metadata.items()
        if key != "metadata_evidence"
    }
    if supplemental:
        payload["supplemental_metadata"] = supplemental
    return payload


def prompt_cache_key(
    record: EpisodeRecord,
    config: PipelineConfig,
    *,
    branch_identity: dict[str, object] | None = None,
    _schema: str = _PROMPT_CACHE_SCHEMA,
) -> str:
    """Fingerprint only Prompt inputs; Reference settings have no effect."""

    if branch_identity is None:
        system_prompt = config.prompt.system_prompt.expanduser().resolve().read_bytes()
        provider = config.prompt.provider
        endpoint = provider.base_url or os.getenv(provider.base_url_env)
        branch_identity = {
            "system_prompt_sha256": sha256_bytes(system_prompt),
            "provider": {
                "name": provider.name,
                "model": provider.model,
                "endpoint": endpoint,
                "response_format": provider.response_format,
                "enable_thinking": provider.enable_thinking,
            },
            "temperature": config.prompt.temperature,
            "max_tokens": config.prompt.max_tokens,
        }
    return fingerprint(
        {
            "pipeline": _schema,
            "branch": "prompt",
            "sample_id": record.sample_id,
            "task": record.task,
            "robot_metadata": _prompt_metadata(record),
            "episode_length": record.episode_length,
            "fps": record.fps,
            "real_video": _file_identity(record.real_video),
            "real_view": record.real_view,
            "frame_count": config.prompt.frame_count,
            "sampling": config.prompt.sampling,
            "resize_long_edge": config.prompt.resize_long_edge,
            "jpeg_quality": config.prompt.jpeg_quality,
            "generator": branch_identity,
        }
    )


def reference_cache_key(
    record: EpisodeRecord,
    config: PipelineConfig,
    *,
    branch_identity: dict[str, object] | None = None,
) -> str:
    """Fingerprint frame-zero robot removal without Prompt or task semantics."""

    if branch_identity is None:
        branch_identity = {
            "config": config.reference.model_dump(mode="json", exclude={"batch_size"})
        }
    return fingerprint(
        {
            "pipeline": _REFERENCE_CACHE_SCHEMA,
            "branch": "reference",
            "sample_id": record.sample_id,
            "real_video": _file_identity(record.real_video),
            "real_view": record.real_view,
            "real_frame_shape": [
                record.real_frame_height,
                record.real_frame_width,
                3,
            ],
            "source_frame_index": 0,
            "extractor": branch_identity,
        }
    )


def _error_row(
    record: EpisodeRecord, stage: str, error: BaseException
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": record.sample_id,
        "dataset": record.dataset_name,
        "episode_index": record.episode_index,
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    diagnostics = getattr(error, "diagnostics", None)
    if isinstance(diagnostics, dict) and diagnostics:
        row["diagnostics"] = diagnostics
    return row


class PreprocessingPipeline:
    """Run, checkpoint, and join the Prompt and clean-scene branches."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        vlm_client: VLMClient | None = None,
        prompt_branch: PromptBranch | None = None,
        reference_branch: ReferenceBranch | None = None,
    ) -> None:
        self.config = config.model_copy(deep=True)
        self._vlm_client = vlm_client
        self._prompt_branch = prompt_branch
        self._custom_prompt_branch = prompt_branch is not None
        self._reference_branch = reference_branch
        self.store = ArtifactStore(self.config.output.root)
        self.publisher = DatasetPublisher(self.config.output)

    def _get_prompt_branch(self) -> PromptBranch:
        if self._prompt_branch is None:
            client = self._vlm_client or QwenOpenAIClient(self.config.prompt.provider)
            self._prompt_branch = PromptBranch(
                client,
                system_prompt=self.config.prompt.system_prompt.read_text(
                    encoding="utf-8"
                ),
                temperature=self.config.prompt.temperature,
                max_tokens=self.config.prompt.max_tokens,
                provider_identity=self.config.prompt.provider.model,
            )
        return self._prompt_branch

    def _get_reference_branch(self) -> ReferenceBranch:
        if self._reference_branch is None:
            self._reference_branch = ReferenceBranch.from_config(self.config.reference)
        return self._reference_branch

    def _prompt_key(self, record: EpisodeRecord) -> str:
        identity = (
            self._prompt_branch.cache_identity()
            if self._custom_prompt_branch and self._prompt_branch is not None
            else None
        )
        return prompt_cache_key(record, self.config, branch_identity=identity)

    def _legacy_prompt_keys(self, record: EpisodeRecord) -> tuple[str, ...]:
        """Reconstruct compatible schema-v2 keys for one-release migration."""

        if self._custom_prompt_branch:
            return ()
        provider = self.config.prompt.provider
        endpoint = provider.base_url or os.getenv(provider.base_url_env)
        values: list[str] = []
        for max_tokens in {
            self.config.prompt.max_tokens,
            _LEGACY_PROMPT_MAX_TOKENS,
        }:
            identity: dict[str, object] = {
                "system_prompt_sha256": _LEGACY_PROMPT_SYSTEM_SHA256,
                "provider": {
                    "name": provider.name,
                    "model": provider.model,
                    "endpoint": endpoint,
                    "response_format": provider.response_format,
                    "enable_thinking": provider.enable_thinking,
                },
                "temperature": self.config.prompt.temperature,
                "max_tokens": max_tokens,
            }
            # The legacy max-token value participates both in branch identity and
            # in no other outer field, so a shallow Prompt config copy is enough.
            legacy_config = self.config.model_copy(deep=True)
            legacy_config.prompt.max_tokens = max_tokens
            values.append(
                prompt_cache_key(
                    record,
                    legacy_config,
                    branch_identity=identity,
                    _schema=_LEGACY_PROMPT_CACHE_SCHEMA,
                )
            )
        return tuple(dict.fromkeys(values))

    def _reference_key(self, record: EpisodeRecord) -> str:
        return reference_cache_key(
            record,
            self.config,
            branch_identity=self._get_reference_branch().cache_identity(),
        )

    def _decode_prompt_batch(
        self, records: list[EpisodeRecord]
    ) -> tuple[dict[str, RealFrameBundle], list[dict[str, Any]]]:
        bundles: dict[str, RealFrameBundle] = {}
        failures: list[dict[str, Any]] = []
        if not records:
            return bundles, failures
        with ThreadPoolExecutor(
            max_workers=min(self.config.runtime.decode_workers, len(records))
        ) as executor:
            futures: dict[Future[RealFrameBundle], EpisodeRecord] = {
                executor.submit(decode_real_video, record, self.config.prompt): record
                for record in records
            }
            for future in as_completed(futures):
                record = futures[future]
                try:
                    bundles[record.sample_id] = future.result()
                except Exception as error:  # noqa: BLE001 - episode boundary
                    failures.append(_error_row(record, "decode-prompt", error))
                    if self.config.runtime.fail_fast:
                        raise
        return bundles, failures

    def _decode_reference_batch(
        self, records: list[EpisodeRecord]
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        """Decode only frame zero for jobs not sharing a Prompt decode."""

        frames: dict[str, np.ndarray] = {}
        failures: list[dict[str, Any]] = []
        if not records:
            return frames, failures
        with ThreadPoolExecutor(
            max_workers=min(self.config.runtime.decode_workers, len(records))
        ) as executor:
            futures: dict[Future[np.ndarray], EpisodeRecord] = {
                executor.submit(decode_real_first_frame, record): record
                for record in records
            }
            for future in as_completed(futures):
                record = futures[future]
                try:
                    frames[record.sample_id] = future.result()
                except Exception as error:  # noqa: BLE001 - episode boundary
                    failures.append(
                        _error_row(record, "decode-reference-frame0", error)
                    )
                    if self.config.runtime.fail_fast:
                        raise
        return frames, failures

    @staticmethod
    def _retryable_prompt_error(error: BaseException) -> bool:
        if isinstance(error, (ResponseParseError, ProductValidationError)):
            return True
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code in {408, 409, 429} or status_code >= 500
        return isinstance(
            error,
            (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError),
        )

    def _generate_prompt(
        self,
        branch: PromptBranch,
        record: EpisodeRecord,
        bundle: RealFrameBundle,
        attempt_counter: list[int],
        attempt_lock: Lock,
    ) -> PromptResult:
        retries = self.config.runtime.api_retry_count
        delay = self.config.runtime.backoff_initial_seconds
        request_max_tokens = self.config.prompt.max_tokens
        for attempt in range(retries + 1):
            try:
                with attempt_lock:
                    attempt_counter[0] += 1
                result = branch.run(
                    sample_id=record.sample_id,
                    task_description=record.task,
                    robot_metadata=_prompt_metadata(record),
                    images=bundle.prompt_frames,
                    max_tokens=request_max_tokens,
                )
                return validate_prompt_result(result)
            except Exception as error:  # noqa: BLE001 - provider boundary
                if attempt >= retries or not self._retryable_prompt_error(error):
                    raise
                if isinstance(error, ResponseParseError) and error.truncated:
                    request_max_tokens = min(4096, max(request_max_tokens * 2, 1024))
                if delay:
                    time.sleep(delay)
                delay = min(
                    self.config.runtime.backoff_max_seconds,
                    max(delay * 2, self.config.runtime.backoff_initial_seconds),
                )
        raise AssertionError("unreachable prompt retry state")

    def _prompt_batch(
        self,
        jobs: list[tuple[EpisodeRecord, RealFrameBundle]],
    ) -> tuple[dict[str, PromptResult], list[dict[str, Any]], int]:
        results: dict[str, PromptResult] = {}
        failures: list[dict[str, Any]] = []
        if not jobs:
            return results, failures, 0
        try:
            branch = self._get_prompt_branch()
        except Exception as error:  # noqa: BLE001 - provider preflight boundary
            if self.config.runtime.fail_fast:
                raise
            return (
                results,
                [_error_row(record, "prompt-preflight", error) for record, _ in jobs],
                0,
            )
        attempt_counter = [0]
        attempt_lock = Lock()
        with ThreadPoolExecutor(
            max_workers=min(self.config.runtime.api_concurrency, len(jobs))
        ) as executor:
            futures: dict[Future[PromptResult], EpisodeRecord] = {
                executor.submit(
                    self._generate_prompt,
                    branch,
                    record,
                    bundle,
                    attempt_counter,
                    attempt_lock,
                ): record
                for record, bundle in jobs
            }
            for future in as_completed(futures):
                record = futures[future]
                try:
                    results[record.sample_id] = future.result()
                except Exception as error:  # noqa: BLE001 - episode boundary
                    failures.append(_error_row(record, "prompt", error))
                    if self.config.runtime.fail_fast:
                        raise
        return results, failures, attempt_counter[0]

    def _reference_batch(
        self,
        jobs: list[tuple[EpisodeRecord, np.ndarray]],
    ) -> tuple[
        dict[str, SceneReferenceResult],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        int,
    ]:
        """Run frame-zero segmentation/inpainting while isolating bad episodes."""

        results: dict[str, SceneReferenceResult] = {}
        failures: list[dict[str, Any]] = []
        deterministic: dict[str, dict[str, Any]] = {}
        if not jobs:
            return results, failures, deterministic, 0
        try:
            branch = self._get_reference_branch()
            branch.ensure_ready()
        except Exception as error:  # noqa: BLE001 - model preflight boundary
            if self.config.runtime.fail_fast:
                raise
            return (
                results,
                [
                    _error_row(record, "reference-preflight", error)
                    for record, _ in jobs
                ],
                deterministic,
                0,
            )

        attempts = [0]

        def process_chunk(chunk: list[tuple[EpisodeRecord, np.ndarray]]) -> None:
            requests = [
                ReferenceBranchInput(
                    sample_id=record.sample_id,
                    episode_index=record.episode_index,
                    frame0=frame,
                    source_view=record.real_view,
                )
                for record, frame in chunk
            ]
            attempts[0] += len(requests)
            try:
                values = branch.process_batch(requests)
                if len(values) != len(requests):
                    raise RuntimeError(
                        "Reference branch returned an invalid batch length"
                    )
            except Exception as error:  # noqa: BLE001 - backend batch boundary
                if len(chunk) > 1:
                    midpoint = len(chunk) // 2
                    process_chunk(chunk[:midpoint])
                    process_chunk(chunk[midpoint:])
                    return
                record = chunk[0][0]
                row = _error_row(record, "reference", error)
                failures.append(row)
                if isinstance(
                    error,
                    (NoValidReferenceError, ProductValidationError, ValueError),
                ):
                    deterministic[record.sample_id] = row
                if self.config.runtime.fail_fast:
                    raise
                return

            for (record, frame), value in zip(chunk, values, strict=True):
                try:
                    result = (
                        value
                        if isinstance(value, SceneReferenceResult)
                        else SceneReferenceResult.model_validate(value)
                    )
                    if result.sample_id != record.sample_id:
                        raise ProductValidationError(
                            "Reference result sample_id differs from request"
                        )
                    artifact = result.artifact
                    if artifact.source_view != record.real_view:
                        raise ProductValidationError(
                            "Reference source_view differs from request"
                        )
                    if (artifact.height, artifact.width) != frame.shape[:2]:
                        raise ProductValidationError(
                            "Reference is not the full-resolution frame-zero scene"
                        )
                    if artifact.source_frame_sha256 != sha256_bytes(
                        frame.tobytes(order="C")
                    ):
                        raise ProductValidationError(
                            "Reference source-frame digest differs from frame zero"
                        )
                    expected_path = (
                        Path("Reference")
                        / f"episode_{record.episode_index:06d}"
                        / "reference_00.jpg"
                    )
                    if artifact.relative_path != expected_path:
                        raise ProductValidationError(
                            "Reference path is not the schema-v3 exact-one path"
                        )
                    results[record.sample_id] = validate_reference_result(result)
                except Exception as error:  # noqa: BLE001 - product boundary
                    row = _error_row(record, "reference", error)
                    failures.append(row)
                    if isinstance(
                        error,
                        (NoValidReferenceError, ProductValidationError, ValueError),
                    ):
                        deterministic[record.sample_id] = row
                    if self.config.runtime.fail_fast:
                        raise

        process_chunk(jobs)
        return results, failures, deterministic, attempts[0]

    @staticmethod
    def _product(
        record: EpisodeRecord,
        prompt: PromptResult,
        reference: SceneReferenceResult,
        reference_key: str,
    ) -> PublishedProduct:
        artifact = reference.artifact
        reference_row = EpisodeReferenceRow(
            episode_index=record.episode_index,
            references=[artifact.row],
        )
        prompt_row = EpisodePromptRow(
            episode_index=record.episode_index,
            prompt=prompt.prompt,
            reference_ids=[artifact.reference_id],
        )
        return PublishedProduct(record, prompt_row, reference_row, reference_key)

    def run(
        self,
        *,
        episodes: set[int] | None = None,
        limit: int | None = None,
        force: bool = False,
        audit: bool = True,
    ) -> dict[str, Any]:
        """Process selected episodes, join only successes, and return one report."""

        started = time.monotonic()
        records = discover_episodes(self.config.dataset, episodes=episodes, limit=limit)
        if not records:
            raise ValueError("No episodes matched the configured dataset selection")

        counters = {
            "prompt_cache_hits": 0,
            "prompt_cache_migrations": 0,
            "reference_cache_hits": 0,
            "reference_failure_cache_hits": 0,
            "prompt_decode_episodes": 0,
            "reference_only_decode_episodes": 0,
            "decoded_episodes": 0,
            "api_requests": 0,
            "reference_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        failures: list[dict[str, Any]] = []
        products: list[PublishedProduct] = []

        for batch in _chunks(records, self.config.reference.batch_size):
            prompts: dict[str, PromptResult] = {}
            references: dict[str, SceneReferenceResult] = {}
            prompt_keys: dict[str, str] = {}
            reference_keys: dict[str, str] = {}
            blocked_prompts: set[str] = set()
            blocked_references: set[str] = set()

            # Resolve and load both branch checkpoints independently. In particular,
            # Reference lookup never waits for or hashes a Prompt result.
            for record in batch:
                sample_id = record.sample_id
                try:
                    prompt_keys[sample_id] = self._prompt_key(record)
                except Exception as error:  # noqa: BLE001 - cache identity boundary
                    failures.append(_error_row(record, "prompt-cache-key", error))
                    blocked_prompts.add(sample_id)
                try:
                    reference_keys[sample_id] = self._reference_key(record)
                except Exception as error:  # noqa: BLE001 - cache identity boundary
                    failures.append(_error_row(record, "reference-cache-key", error))
                    blocked_references.add(sample_id)

            if self.config.runtime.resume and not force:
                for record in batch:
                    sample_id = record.sample_id
                    prompt_key = prompt_keys.get(sample_id)
                    if prompt_key is not None:
                        cached_prompt: PromptResult | None = None
                        matched_prompt_key: str | None = None
                        for candidate_key in (
                            prompt_key,
                            *self._legacy_prompt_keys(record),
                        ):
                            cached_prompt = self.store.load_prompt(
                                sample_id, cache_key=candidate_key
                            )
                            if cached_prompt is not None:
                                matched_prompt_key = candidate_key
                                break
                        if cached_prompt is not None and matched_prompt_key is not None:
                            try:
                                prompts[sample_id] = validate_prompt_result(
                                    cached_prompt
                                )
                                counters["prompt_cache_hits"] += 1
                                if matched_prompt_key != prompt_key:
                                    self.store.save_prompt(
                                        prompts[sample_id], cache_key=prompt_key
                                    )
                                    counters["prompt_cache_migrations"] += 1
                            except ProductValidationError:
                                pass

                    reference_key = reference_keys.get(sample_id)
                    if reference_key is None:
                        continue
                    cached_reference = self.store.load_reference(
                        record, cache_key=reference_key
                    )
                    if cached_reference is not None:
                        try:
                            references[sample_id] = validate_reference_result(
                                cached_reference
                            )
                            counters["reference_cache_hits"] += 1
                            continue
                        except ProductValidationError:
                            pass
                    cached_failure = self.store.load_reference_failure(
                        sample_id, cache_key=reference_key
                    )
                    if cached_failure is not None:
                        failure = dict(cached_failure)
                        failure["cached"] = True
                        failures.append(failure)
                        blocked_references.add(sample_id)
                        counters["reference_failure_cache_hits"] += 1

            prompt_decode_records = [
                record
                for record in batch
                if record.sample_id not in prompts
                and record.sample_id not in blocked_prompts
            ]
            bundles, prompt_decode_failures = self._decode_prompt_batch(
                prompt_decode_records
            )
            failures.extend(prompt_decode_failures)
            counters["prompt_decode_episodes"] += len(bundles)

            frame0: dict[str, np.ndarray] = {
                sample_id: bundle.first_frame_bgr
                for sample_id, bundle in bundles.items()
                if sample_id not in blocked_references and sample_id not in references
            }
            # If the eight-frame Prompt decode failed, frame zero gets an independent
            # second chance. Cached Prompt jobs also open only frame zero.
            reference_decode_records = [
                record
                for record in batch
                if record.sample_id not in references
                and record.sample_id not in blocked_references
                and record.sample_id not in frame0
            ]
            standalone_frames, reference_decode_failures = self._decode_reference_batch(
                reference_decode_records
            )
            frame0.update(standalone_frames)
            failures.extend(reference_decode_failures)
            counters["reference_only_decode_episodes"] += len(standalone_frames)
            counters["decoded_episodes"] += len(bundles) + len(standalone_frames)

            prompt_jobs = [
                (record, bundles[record.sample_id])
                for record in batch
                if record.sample_id not in prompts
                and record.sample_id in bundles
                and record.sample_id not in blocked_prompts
            ]
            reference_jobs = [
                (record, frame0[record.sample_id])
                for record in batch
                if record.sample_id not in references
                and record.sample_id in frame0
                and record.sample_id not in blocked_references
            ]

            # Network-bound Prompt work and model-bound robot removal overlap. Their
            # results and failures are collected and checkpointed independently.
            records_by_id = {record.sample_id: record for record in batch}
            with ThreadPoolExecutor(max_workers=2) as branch_executor:
                branch_futures: dict[Future[object], str] = {}
                if prompt_jobs:
                    branch_futures[
                        branch_executor.submit(self._prompt_batch, prompt_jobs)
                    ] = "prompt"
                if reference_jobs:
                    branch_futures[
                        branch_executor.submit(self._reference_batch, reference_jobs)
                    ] = "reference"
                for branch_future in as_completed(branch_futures):
                    if branch_futures[branch_future] == "prompt":
                        generated, branch_failures, attempts = branch_future.result()
                        counters["api_requests"] += attempts
                        failures.extend(branch_failures)
                        for sample_id, prompt in generated.items():
                            prompts[sample_id] = prompt
                            counters["input_tokens"] += prompt.input_tokens or 0
                            counters["output_tokens"] += prompt.output_tokens or 0
                            self.store.save_prompt(
                                prompt, cache_key=prompt_keys[sample_id]
                            )
                        continue

                    (
                        generated_references,
                        branch_failures,
                        deterministic_failures,
                        attempts,
                    ) = branch_future.result()
                    counters["reference_requests"] += attempts
                    failures.extend(branch_failures)
                    for sample_id, reference in generated_references.items():
                        references[sample_id] = reference
                        self.store.save_reference(
                            records_by_id[sample_id],
                            reference,
                            cache_key=reference_keys[sample_id],
                        )
                    for (
                        sample_id,
                        deterministic_failure,
                    ) in deterministic_failures.items():
                        self.store.save_reference_failure(
                            sample_id,
                            cache_key=reference_keys[sample_id],
                            failure=deterministic_failure,
                        )

            for record in batch:
                sample_id = record.sample_id

                prompt = prompts.get(sample_id)
                reference = references.get(sample_id)
                if prompt is not None and reference is not None:
                    products.append(
                        self._product(
                            record,
                            prompt,
                            reference,
                            reference_keys[sample_id],
                        )
                    )

        dataset_count, published = self.publisher.publish_rows(
            products, records=records
        )
        selection = episodes is not None or limit is not None
        audit_report = (
            audit_products(
                records,
                self.config.output,
                require_exact_episode_set=not selection,
            )
            if audit
            else None
        )
        if audit_report is not None:
            audit_report["scope"] = "selection" if selection else "dataset"
        audit_complete = audit_report is None or audit_report["status"] == "complete"
        report: dict[str, Any] = {
            "status": (
                "complete"
                if not failures and published == len(records) and audit_complete
                else "partial"
            ),
            "annotations_ready": (
                audit_report["status"] == "complete"
                if audit_report is not None and not selection
                else None
            ),
            "schema_version": 3,
            "selected_episodes": len(records),
            "published_episodes": published,
            "failed_episodes": len({row["sample_id"] for row in failures}),
            "dataset_count": dataset_count,
            **counters,
            "failures": failures,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "audit": audit_report,
        }
        if self.config.runtime.log_costs:
            input_rate = self.config.runtime.input_cost_per_million
            output_rate = self.config.runtime.output_cost_per_million
            if input_rate is not None and output_rate is not None:
                report["estimated_api_cost"] = round(
                    counters["input_tokens"] * input_rate / 1_000_000
                    + counters["output_tokens"] * output_rate / 1_000_000,
                    6,
                )
        report_filename = os.getenv(
            "SIM2REAL_PROMPT_RUN_REPORT_FILENAME", "run_report.json"
        )
        atomic_write_json(
            resolve_inside(self.config.output.root, report_filename), report
        )
        return report

    def audit(
        self,
        *,
        episodes: set[int] | None = None,
        limit: int | None = None,
        show: int = 20,
    ) -> dict[str, Any]:
        records = discover_episodes(self.config.dataset, episodes=episodes, limit=limit)
        if not records:
            raise ValueError("No episodes matched the configured dataset selection")
        selection = episodes is not None or limit is not None
        report = audit_products(
            records,
            self.config.output,
            show=show,
            require_exact_episode_set=not selection,
        )
        report["scope"] = "selection" if selection else "dataset"
        report["annotations_ready"] = (
            report["status"] == "complete" if not selection else None
        )
        return report
