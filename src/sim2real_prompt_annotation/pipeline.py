"""Chunked end-to-end orchestration for the Real-only two-branch pipeline."""

from __future__ import annotations

import os
import time
from collections import defaultdict
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
from .io_utils import atomic_write_json, fingerprint, sha256_bytes
from .models import (
    EpisodePromptRow,
    EpisodeRecord,
    EpisodeReferenceRow,
    PromptResult,
    RealFrameBundle,
    ReferenceBranchResult,
)
from .prompt_branch import PromptBranch
from .qwen import QwenOpenAIClient, ResponseParseError, VLMClient
from .reference_branch import ReferenceBranch
from .validation import (
    ProductValidationError,
    validate_prompt_result,
    validate_reference_result,
)
from .video import decode_real_first_frame, decode_real_video
from .yoloe import DetectionRequest, request_signature

_PIPELINE_SCHEMA = "real8-yoloe-multiref-v1"


@dataclass(frozen=True, slots=True)
class PublishedProduct:
    """Lightweight joined product; Reference JPEGs are already on disk."""

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


def _can_split_detector_batch(error: BaseException) -> bool:
    """Retry smaller batches only for memory-allocation failures."""

    messages: list[str] = []
    current: BaseException | None = error
    while current is not None and len(messages) < 4:
        messages.append(str(current).casefold())
        current = current.__cause__ or current.__context__
    message = " ".join(messages)
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cudnn_status_alloc_failed",
            "cublas_status_alloc_failed",
            "hip error out of memory",
        )
    )


def prompt_cache_key(
    record: EpisodeRecord,
    config: PipelineConfig,
    *,
    branch_identity: dict[str, object] | None = None,
) -> str:
    """Cache without decoding; source video stat changes invalidate the branch."""

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
            "pipeline": _PIPELINE_SCHEMA,
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
    prompt: PromptResult,
    config: PipelineConfig,
    *,
    branch_identity: dict[str, object] | None = None,
) -> str:
    """Bind a Reference cache entry to video, queries, weights, and selection."""

    if branch_identity is None:
        branch_identity = {
            "backend": config.reference.backend,
            "model": {"locator": str(config.reference.model_path)},
            "image_size": config.reference.image_size,
            "confidence": config.reference.confidence,
            "iou_threshold": config.reference.iou_threshold,
            "crop_padding": config.reference.crop_padding,
            "duplicate_iou": config.reference.duplicate_iou,
            "candidate_pool_size": config.reference.candidate_pool_size,
            "min_images": config.reference.min_images,
            "max_images": config.reference.max_images,
            "selection_seed": config.reference.selection_seed,
            "jpeg_quality": config.reference.jpeg_quality,
        }
    return fingerprint(
        {
            "pipeline": _PIPELINE_SCHEMA,
            "branch": "reference",
            "sample_id": record.sample_id,
            "real_video": _file_identity(record.real_video),
            "real_view": record.real_view,
            "queries": [
                query.model_dump(mode="json") for query in prompt.reference_queries
            ],
            "extractor": branch_identity,
        }
    )


def _error_row(
    record: EpisodeRecord, stage: str, error: BaseException
) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "dataset": record.dataset_name,
        "episode_index": record.episode_index,
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
    }


class PreprocessingPipeline:
    """Run one VLM request and one YOLOE first-frame pass per uncached episode."""

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
        return prompt_cache_key(
            record,
            self.config,
            branch_identity=identity,
        )

    def _reference_key(
        self,
        record: EpisodeRecord,
        prompt: PromptResult,
    ) -> str:
        return reference_cache_key(
            record,
            prompt,
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
                    failures.append(_error_row(record, "decode", error))
                    if self.config.runtime.fail_fast:
                        raise
        return bundles, failures

    def _decode_reference_batch(
        self, records: list[EpisodeRecord]
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        """Decode frame zero only when the Prompt branch is already cached."""

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
                    failures.append(_error_row(record, "decode-reference", error))
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
        for attempt in range(retries + 1):
            try:
                with attempt_lock:
                    attempt_counter[0] += 1
                result = branch.run(
                    sample_id=record.sample_id,
                    task_description=record.task,
                    robot_metadata=_prompt_metadata(record),
                    images=bundle.prompt_frames,
                )
                return validate_prompt_result(result)
            except Exception as error:  # noqa: BLE001 - provider boundary
                if attempt >= retries or not self._retryable_prompt_error(error):
                    raise
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
        branch = self._get_prompt_branch()
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

    def _detect_reference_batch(
        self,
        jobs: list[tuple[EpisodeRecord, np.ndarray, PromptResult]],
    ) -> tuple[dict[str, ReferenceBranchResult], list[dict[str, Any]]]:
        """Share GPU batches while isolating post-processing per episode."""

        results: dict[str, ReferenceBranchResult] = {}
        failures: list[dict[str, Any]] = []
        if not jobs:
            return results, failures
        branch = self._get_reference_branch()
        frames = [frame0 for _, frame0, _ in jobs]
        requests = [
            DetectionRequest(
                frame=frame,
                queries=prompt.reference_queries,
                request_id=record.sample_id,
            )
            for (record, _, prompt), frame in zip(jobs, frames, strict=True)
        ]
        detected: list[list[Any] | None] = [None] * len(requests)
        predict = getattr(branch.detector, "predict", None)
        if predict is None:
            try:
                batches = branch.detector.predict_requests(requests)
                if len(batches) != len(requests):
                    raise RuntimeError("Reference detector returned an invalid batch")
                detected = list(batches)
            except Exception as error:  # noqa: BLE001 - detector boundary
                for record, _, _ in jobs:
                    failures.append(_error_row(record, "reference-detect", error))
                if self.config.runtime.fail_fast:
                    raise
        else:
            grouped: defaultdict[tuple[tuple[str, str, bool], ...], list[int]] = (
                defaultdict(list)
            )
            for index, request in enumerate(requests):
                grouped[request_signature(request.queries)].append(index)

            def predict_group(indices: list[int]) -> None:
                representative = requests[indices[0]]
                try:
                    batches = predict(
                        [requests[index].frame for index in indices],
                        representative.queries,
                    )
                    if len(batches) != len(indices):
                        raise RuntimeError(
                            "Reference detector returned an invalid batch"
                        )
                    for index, values in zip(indices, batches, strict=True):
                        detected[index] = values
                except Exception as error:  # noqa: BLE001 - GPU batch boundary
                    if len(indices) > 1 and _can_split_detector_batch(error):
                        midpoint = len(indices) // 2
                        predict_group(indices[:midpoint])
                        predict_group(indices[midpoint:])
                        return
                    for index in indices:
                        record = jobs[index][0]
                        failures.append(_error_row(record, "reference-detect", error))
                    if self.config.runtime.fail_fast:
                        raise

            for indices in grouped.values():
                predict_group(indices)

        for (record, _, prompt), frame, detections in zip(
            jobs, frames, detected, strict=True
        ):
            if detections is None:
                continue
            try:
                result = branch.process_detections(
                    sample_id=record.sample_id,
                    episode_index=record.episode_index,
                    frame0=frame,
                    queries=prompt.reference_queries,
                    detections=detections,
                    source_view=record.real_view,
                )
                results[record.sample_id] = validate_reference_result(
                    result,
                    prompt.reference_queries,
                    min_images=self.config.reference.min_images,
                    max_images=self.config.reference.max_images,
                )
            except Exception as error:  # noqa: BLE001 - episode boundary
                failures.append(_error_row(record, "reference", error))
                if self.config.runtime.fail_fast:
                    raise
        return results, failures

    @staticmethod
    def _product(
        record: EpisodeRecord,
        prompt: PromptResult,
        reference: ReferenceBranchResult,
        reference_key: str,
    ) -> PublishedProduct:
        reference_row = EpisodeReferenceRow(
            episode_index=record.episode_index,
            references=[item.row for item in reference.selected_artifacts],
        )
        prompt_row = EpisodePromptRow(
            episode_index=record.episode_index,
            prompt=prompt.prompt,
            reference_ids=[item.reference_id for item in reference.selected_artifacts],
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
        """Process selected episodes, publish successes, and return one run report."""

        started = time.monotonic()
        records = discover_episodes(self.config.dataset, episodes=episodes, limit=limit)
        if not records:
            raise ValueError("No episodes matched the configured dataset selection")

        counters = {
            "prompt_cache_hits": 0,
            "reference_cache_hits": 0,
            "decoded_episodes": 0,
            "api_requests": 0,
            "detector_episodes": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        failures: list[dict[str, Any]] = []
        products: list[PublishedProduct] = []
        detector_ready = False
        detector_preflight_error: BaseException | None = None

        for batch in _chunks(records, self.config.reference.batch_size):
            prompts: dict[str, PromptResult] = {}
            references: dict[str, ReferenceBranchResult] = {}
            prompt_keys: dict[str, str] = {}
            reference_keys: dict[str, str] = {}

            for record in batch:
                key = self._prompt_key(record)
                prompt_keys[record.sample_id] = key
                if self.config.runtime.resume and not force:
                    cached = self.store.load_prompt(record.sample_id, cache_key=key)
                    if cached is not None:
                        try:
                            prompts[record.sample_id] = validate_prompt_result(cached)
                            counters["prompt_cache_hits"] += 1
                        except ProductValidationError:
                            pass

            for record in batch:
                prompt = prompts.get(record.sample_id)
                if prompt is None:
                    continue
                key = self._reference_key(record, prompt)
                reference_keys[record.sample_id] = key
                if self.config.runtime.resume and not force:
                    cached = self.store.load_reference(record, cache_key=key)
                    if cached is not None:
                        try:
                            references[record.sample_id] = validate_reference_result(
                                cached,
                                prompt.reference_queries,
                                min_images=self.config.reference.min_images,
                                max_images=self.config.reference.max_images,
                            )
                            counters["reference_cache_hits"] += 1
                        except ProductValidationError:
                            pass

            unresolved = [
                record
                for record in batch
                if record.sample_id not in prompts or record.sample_id not in references
            ]
            if unresolved and not detector_ready and detector_preflight_error is None:
                try:
                    self._get_reference_branch().ensure_ready()
                    detector_ready = True
                except Exception as error:  # noqa: BLE001 - runtime boundary
                    detector_preflight_error = error
                    if self.config.runtime.fail_fast:
                        raise
            if unresolved and detector_preflight_error is not None:
                failures.extend(
                    _error_row(record, "reference-preflight", detector_preflight_error)
                    for record in unresolved
                )
                for record in batch:
                    prompt = prompts.get(record.sample_id)
                    reference = references.get(record.sample_id)
                    if prompt is not None and reference is not None:
                        products.append(
                            self._product(
                                record,
                                prompt,
                                reference,
                                reference_keys[record.sample_id],
                            )
                        )
                continue

            prompt_decode = [
                record for record in batch if record.sample_id not in prompts
            ]
            reference_decode = [
                record
                for record in batch
                if record.sample_id in prompts and record.sample_id not in references
            ]
            bundles, decode_failures = self._decode_prompt_batch(prompt_decode)
            first_frames, reference_decode_failures = self._decode_reference_batch(
                reference_decode
            )
            counters["decoded_episodes"] += len(bundles) + len(first_frames)
            first_frames.update(
                {
                    sample_id: bundle.first_frame_bgr
                    for sample_id, bundle in bundles.items()
                }
            )
            failures.extend(decode_failures)
            failures.extend(reference_decode_failures)

            prompt_jobs = [
                (record, bundles[record.sample_id])
                for record in batch
                if record.sample_id not in prompts and record.sample_id in bundles
            ]
            generated, prompt_failures, api_attempts = self._prompt_batch(prompt_jobs)
            counters["api_requests"] += api_attempts
            failures.extend(prompt_failures)
            for record in batch:
                result = generated.get(record.sample_id)
                if result is not None:
                    prompts[record.sample_id] = result
                    counters["input_tokens"] += result.input_tokens or 0
                    counters["output_tokens"] += result.output_tokens or 0
                    self.store.save_prompt(
                        result, cache_key=prompt_keys[record.sample_id]
                    )

            reference_jobs: list[tuple[EpisodeRecord, np.ndarray, PromptResult]] = []
            for record in batch:
                sample_id = record.sample_id
                prompt = prompts.get(sample_id)
                first_frame = first_frames.get(sample_id)
                if prompt is None or sample_id in references or first_frame is None:
                    continue
                key = self._reference_key(record, prompt)
                reference_keys[sample_id] = key
                if self.config.runtime.resume and not force:
                    cached = self.store.load_reference(record, cache_key=key)
                    if cached is not None:
                        try:
                            references[sample_id] = validate_reference_result(
                                cached,
                                prompt.reference_queries,
                                min_images=self.config.reference.min_images,
                                max_images=self.config.reference.max_images,
                            )
                            counters["reference_cache_hits"] += 1
                            continue
                        except ProductValidationError:
                            pass
                reference_jobs.append((record, first_frame, prompt))

            detected, reference_failures = self._detect_reference_batch(reference_jobs)
            counters["detector_episodes"] += len(reference_jobs)
            failures.extend(reference_failures)
            for record in batch:
                result = detected.get(record.sample_id)
                if result is not None:
                    references[record.sample_id] = result
                    self.store.save_reference(
                        record,
                        result,
                        cache_key=reference_keys[record.sample_id],
                    )

            for record in batch:
                prompt = prompts.get(record.sample_id)
                reference = references.get(record.sample_id)
                if prompt is not None and reference is not None:
                    products.append(
                        self._product(
                            record,
                            prompt,
                            reference,
                            reference_keys[record.sample_id],
                        )
                    )

        dataset_count, published = self.publisher.publish_rows(products)
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
        atomic_write_json(self.config.output.root / "run_report.json", report)
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
