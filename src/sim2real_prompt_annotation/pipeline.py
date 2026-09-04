"""Concurrent, resumable project-specific annotation pipeline."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DatasetPromptExportConfig, PipelineConfig
from .lerobot import SampleRecord, discover_samples
from .media import MediaPreparer, PreparedMedia
from .models import StructuredAnnotation, ValidationResult
from .qwen import QwenOpenAIClient, ResponseParseError, VLMClient, VLMResponse
from .renderer import PromptRenderer
from .validation import (
    canonicalize_annotation,
    find_local_issues,
    merge_validation,
)


class AnnotationService:
    """Run annotation and critic calls with schema-specific retries."""

    def __init__(self, config: PipelineConfig, client: VLMClient):
        self.config = config
        self.client = client
        self.annotation_prompt = self._load_prompt(config.annotation.system_prompt)
        self.critic_prompt = self._load_prompt(config.critic.system_prompt)

    @staticmethod
    def _load_prompt(path: Path) -> str:
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"System prompt is empty: {path}")
        return value

    def _api_call(self, **kwargs: Any) -> VLMResponse:
        retries = self.config.batch.api_retry_count
        delay = self.config.batch.backoff_initial_seconds
        for attempt in range(retries + 1):
            try:
                return self.client.generate(**kwargs)
            except ResponseParseError:
                raise
            except Exception:
                if attempt >= retries:
                    raise
                if delay:
                    time.sleep(min(delay, self.config.batch.backoff_max_seconds))
                delay = min(
                    max(delay * 2, 0.001), self.config.batch.backoff_max_seconds
                )
        raise AssertionError("unreachable")

    @staticmethod
    def _prompt_metadata(record: SampleRecord, media: PreparedMedia) -> dict[str, Any]:
        metadata = record.annotation_metadata()
        metadata.pop("frame_mapping", None)
        metadata.pop("media_urls", None)
        if media.reference is not None:
            metadata["selected_reference"] = {
                "view": media.reference.view,
                "frame_index": media.reference.frame_index,
                "evidence_id": media.reference.evidence_id,
            }
        return metadata

    @staticmethod
    def _fix_identifiers(
        annotation: StructuredAnnotation,
        record: SampleRecord,
        media: PreparedMedia,
    ) -> StructuredAnnotation:
        updates: dict[str, Any] = {"sample_id": record.sample_id}
        if media.reference is not None:
            updates["reference"] = annotation.reference.model_copy(
                update={
                    "view": media.reference.view,
                    "frame_index": media.reference.frame_index,
                }
            )
        return annotation.model_copy(update=updates)

    def annotate(
        self,
        record: SampleRecord,
        media: PreparedMedia,
        *,
        critic_feedback: ValidationResult | None = None,
    ) -> VLMResponse:
        metadata = json.dumps(
            self._prompt_metadata(record, media),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        user_text = (
            "Create the detailed annotation and compact prompt plan for this paired "
            "episode.\n\n"
            f"AUTHORITATIVE METADATA JSON:\n{metadata}"
        )
        if critic_feedback is not None:
            user_text += (
                "\n\nThe previous candidate failed validation. Correct the listed "
                "problems while re-deriving claims from the supplied evidence:\n"
                + critic_feedback.model_dump_json(indent=2)
            )
        last_error: Exception | None = None
        for malformed_attempt in range(self.config.annotation.retry_count + 1):
            retry_text = user_text
            if malformed_attempt:
                retry_text += (
                    "\n\nThe previous response failed JSON/schema validation. "
                    "Return one complete JSON object matching the schema exactly."
                )
            try:
                response = self._api_call(
                    sample_id=record.sample_id,
                    stage="annotation",
                    system_prompt=self.annotation_prompt,
                    user_text=retry_text,
                    media=media,
                    response_model=StructuredAnnotation,
                    temperature=self.config.annotation.temperature,
                    max_tokens=self.config.annotation.max_tokens,
                )
                annotation = self._fix_identifiers(
                    StructuredAnnotation.model_validate(response.payload.model_dump()),
                    record,
                    media,
                )
                return VLMResponse(
                    payload=annotation,
                    raw_text=response.raw_text,
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    request_id=response.request_id,
                )
            except ResponseParseError as error:
                last_error = error
        assert last_error is not None
        raise last_error

    def critique(
        self,
        record: SampleRecord,
        media: PreparedMedia,
        annotation: StructuredAnnotation,
    ) -> VLMResponse:
        metadata = json.dumps(
            self._prompt_metadata(record, media),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        user_text = (
            "Validate the candidate against every supplied condition.\n\n"
            f"AUTHORITATIVE METADATA JSON:\n{metadata}\n\n"
            f"CANDIDATE ANNOTATION JSON:\n{annotation.model_dump_json(indent=2)}"
        )
        last_error: Exception | None = None
        for malformed_attempt in range(self.config.critic.retry_count + 1):
            retry_text = user_text
            if malformed_attempt:
                retry_text += (
                    "\n\nThe previous critic response failed JSON/schema validation. "
                    "Return one complete validation JSON object matching the schema."
                )
            try:
                return self._api_call(
                    sample_id=record.sample_id,
                    stage="critique",
                    system_prompt=self.critic_prompt,
                    user_text=retry_text,
                    media=media,
                    response_model=ValidationResult,
                    temperature=self.config.critic.temperature,
                    max_tokens=self.config.critic.max_tokens,
                )
            except ResponseParseError as error:
                last_error = error
        assert last_error is not None
        raise last_error


class DatasetPromptExporter:
    """Merge one prompt and its exact Reference identity into dataset metadata."""

    def __init__(self, config: DatasetPromptExportConfig, output_root: Path):
        self.config = config
        self.output_root = output_root

    def _prompt_path(self, record: SampleRecord) -> Path:
        return self.output_root / "prompts" / f"{record.sample_id}.txt"

    def _annotation_path(self, record: SampleRecord) -> Path:
        return self.output_root / "annotations" / f"{record.sample_id}.json"

    @staticmethod
    def _read_existing(path: Path) -> dict[int, dict[str, Any]]:
        if not path.is_file():
            return {}
        result: dict[int, dict[str, Any]] = {}
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
                if not isinstance(row, dict) or "episode_index" not in row:
                    raise ValueError(f"{path}:{line_number}: expected episode_index")
                episode_index = int(row["episode_index"])
                if episode_index in result:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate episode {episode_index}"
                    )
                prompt = row.get("prompt")
                view = row.get("reference_view")
                frame = row.get("reference_frame_index")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        f"{path}:{line_number}: expected one non-empty prompt string; "
                        "legacy prompt variants are not supported"
                    )
                if not isinstance(view, str) or not view.strip():
                    raise ValueError(f"{path}:{line_number}: expected reference_view")
                if not isinstance(frame, int) or frame < 0:
                    raise ValueError(
                        f"{path}:{line_number}: expected non-negative "
                        "reference_frame_index"
                    )
                result[episode_index] = {
                    "episode_index": episode_index,
                    "prompt": prompt.strip(),
                    "reference_view": view.strip(),
                    "reference_frame_index": frame,
                }
        return result

    @classmethod
    def read_existing(cls, path: str | Path) -> dict[int, dict[str, Any]]:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return cls._read_existing(resolved)

    @staticmethod
    def _atomic_write_rows(path: Path, rows: dict[int, dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        text = "".join(
            json.dumps(rows[index], ensure_ascii=False, separators=(",", ":")) + "\n"
            for index in sorted(rows)
        )
        temporary.write_text(text, encoding="utf-8")
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)

    def export(self, records: Iterable[SampleRecord]) -> tuple[int, int]:
        if not self.config.enabled:
            return 0, 0
        grouped: defaultdict[Path, list[SampleRecord]] = defaultdict(list)
        for record in records:
            grouped[record.dataset_root].append(record)

        dataset_count = 0
        prompt_count = 0
        for dataset_root, dataset_records in sorted(
            grouped.items(), key=lambda item: str(item[0])
        ):
            destination = dataset_root / "meta" / self.config.filename
            rows = (
                self._read_existing(destination) if self.config.merge_existing else {}
            )
            updated = 0
            for record in dataset_records:
                prompt_path = self._prompt_path(record)
                annotation_path = self._annotation_path(record)
                if not prompt_path.is_file() or not annotation_path.is_file():
                    continue
                prompt = prompt_path.read_text(encoding="utf-8").strip()
                if not prompt:
                    raise ValueError(f"Final prompt is empty: {prompt_path}")
                annotation = StructuredAnnotation.model_validate_json(
                    annotation_path.read_text(encoding="utf-8")
                )
                rows[record.episode_index] = {
                    "episode_index": record.episode_index,
                    "prompt": prompt,
                    "reference_view": annotation.reference.view,
                    "reference_frame_index": annotation.reference.frame_index,
                }
                updated += 1
            if not updated:
                continue
            self._atomic_write_rows(destination, rows)
            dataset_count += 1
            prompt_count += updated
        return dataset_count, prompt_count


@dataclass(frozen=True)
class BatchResult:
    total: int
    succeeded: int
    failed: int
    skipped: int
    exported_datasets: int
    exported_prompts: int
    complete: bool
    incomplete_samples: list[str]


class BatchPipeline:
    def __init__(self, config: PipelineConfig, client: VLMClient | None = None):
        self.config = config
        self.client = client or QwenOpenAIClient(config.provider)
        self.service = AnnotationService(config, self.client)
        self.media_preparer = MediaPreparer(config.media)
        self.renderer = PromptRenderer(config.renderer)
        self.dataset_exporter = DatasetPromptExporter(
            config.dataset_prompt_export, config.output_root
        )
        self._log_lock = threading.Lock()
        self._prepare_directories()

    def _prepare_directories(self) -> None:
        for relative in (
            "annotations",
            "annotations_raw",
            "critiques",
            "prompts",
            "logs",
        ):
            (self.config.output_root / relative).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)

    def _append_jsonl(self, path: Path, value: dict[str, Any]) -> None:
        line = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        with self._log_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def _annotation_path(self, sample_id: str) -> Path:
        return self.config.output_root / "annotations" / f"{sample_id}.json"

    def _prompt_path(self, sample_id: str) -> Path:
        return self.config.output_root / "prompts" / f"{sample_id}.txt"

    def _render_existing(self, sample_id: str) -> bool:
        path = self._annotation_path(sample_id)
        if not path.is_file():
            return False
        try:
            annotation = StructuredAnnotation.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if annotation.sample_id != sample_id:
                return False
            prompt = self.renderer.render(annotation)
        except (OSError, ValueError):
            return False
        output = self._prompt_path(sample_id)
        try:
            current = output.read_text(encoding="utf-8").strip()
        except OSError:
            current = None
        if current != prompt:
            self._atomic_write(output, prompt + "\n")
        return True

    def _audit_completion(
        self, records: Iterable[SampleRecord]
    ) -> list[dict[str, Any]]:
        exported_tables: dict[Path, dict[int, dict[str, Any]]] = {}
        incomplete: list[dict[str, Any]] = []
        for record in records:
            reasons: list[str] = []
            rendered: str | None = None
            annotation: StructuredAnnotation | None = None
            try:
                annotation = StructuredAnnotation.model_validate_json(
                    self._annotation_path(record.sample_id).read_text(encoding="utf-8")
                )
                if annotation.sample_id != record.sample_id:
                    reasons.append("canonical sample_id mismatch")
                rendered = self.renderer.render(annotation)
            except (OSError, ValueError) as error:
                reasons.append(f"invalid/missing canonical annotation: {error}")

            prompt: str | None = None
            try:
                prompt = (
                    self._prompt_path(record.sample_id)
                    .read_text(encoding="utf-8")
                    .strip()
                )
                if not prompt:
                    reasons.append("empty final prompt")
                elif rendered is not None and prompt != rendered:
                    reasons.append("prompt differs from deterministic render")
            except OSError as error:
                reasons.append(f"missing final prompt: {error}")

            if self.config.dataset_prompt_export.enabled:
                destination = (
                    record.dataset_root
                    / "meta"
                    / self.config.dataset_prompt_export.filename
                )
                try:
                    if destination not in exported_tables:
                        exported_tables[destination] = (
                            self.dataset_exporter.read_existing(destination)
                        )
                    row = exported_tables[destination].get(record.episode_index)
                except (OSError, ValueError) as error:
                    reasons.append(f"cannot validate consolidated prompt file: {error}")
                    row = None
                if row is None:
                    reasons.append("consolidated prompt row is missing")
                elif annotation is not None and (
                    row["prompt"] != prompt
                    or row["reference_view"] != annotation.reference.view
                    or row["reference_frame_index"] != annotation.reference.frame_index
                ):
                    reasons.append(
                        "consolidated prompt/reference differs from canonical output"
                    )

            if reasons:
                incomplete.append(
                    {
                        "sample_id": record.sample_id,
                        "dataset_root": str(record.dataset_root),
                        "episode_index": record.episode_index,
                        "reasons": reasons,
                    }
                )
        return incomplete

    def _log_usage(self, sample_id: str, stage: str, response: VLMResponse) -> None:
        if not self.config.batch.log_costs:
            return
        input_tokens = response.input_tokens or 0
        output_tokens = response.output_tokens or 0
        estimated_cost = None
        if (
            self.config.batch.input_cost_per_million is not None
            and self.config.batch.output_cost_per_million is not None
        ):
            estimated_cost = (
                input_tokens * self.config.batch.input_cost_per_million
                + output_tokens * self.config.batch.output_cost_per_million
            ) / 1_000_000
        self._append_jsonl(
            self.config.output_root / "logs/requests.jsonl",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sample_id": sample_id,
                "stage": stage,
                "model": response.model,
                "request_id": response.request_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "estimated_cost": estimated_cost,
            },
        )

    def _process(self, record: SampleRecord) -> str:
        if self.config.batch.skip_completed and self._render_existing(record.sample_id):
            return "skipped"

        media = self.media_preparer.prepare(record)
        feedback: ValidationResult | None = None
        annotation_response: VLMResponse | None = None
        annotation: StructuredAnnotation | None = None
        validation = ValidationResult()
        for attempt in range(self.config.annotation.severe_retry_count + 1):
            annotation_response = self.service.annotate(
                record, media, critic_feedback=feedback
            )
            self._log_usage(
                record.sample_id, f"annotation_attempt_{attempt}", annotation_response
            )
            annotation = StructuredAnnotation.model_validate(
                annotation_response.payload.model_dump()
            )
            local = find_local_issues(
                annotation,
                record.annotation_metadata(),
                confidence_threshold=self.config.annotation.confidence_threshold,
                renderer=self.renderer,
            )
            if self.config.critic.enabled:
                critique_response = self.service.critique(record, media, annotation)
                self._log_usage(
                    record.sample_id, f"critique_attempt_{attempt}", critique_response
                )
                validation = merge_validation(
                    ValidationResult.model_validate(
                        critique_response.payload.model_dump()
                    ),
                    local,
                )
            else:
                validation = local
            if (
                not validation.has_severe_errors()
                or attempt >= self.config.annotation.severe_retry_count
            ):
                break
            feedback = validation

        assert annotation_response is not None and annotation is not None
        self._atomic_write(
            self.config.output_root / "annotations_raw" / f"{record.sample_id}.json",
            annotation.model_dump_json(indent=2) + "\n",
        )
        if self.config.critic.enabled:
            self._atomic_write(
                self.config.output_root / "critiques" / f"{record.sample_id}.json",
                validation.model_dump_json(indent=2) + "\n",
            )
        if (
            validation.has_severe_errors()
            and self.config.critic.fail_on_severe_after_retries
        ):
            raise ValueError(
                f"Validation still has severe errors after retries: {record.sample_id}"
            )

        canonical = canonicalize_annotation(annotation)
        prompt = self.renderer.render(canonical)
        self._atomic_write(self._prompt_path(record.sample_id), prompt + "\n")
        # Canonical annotation is the completion marker and is written last.
        self._atomic_write(
            self._annotation_path(record.sample_id),
            canonical.model_dump_json(indent=2) + "\n",
        )
        return "succeeded"

    def run(self, records: Iterable[SampleRecord] | None = None) -> BatchResult:
        if records is None:
            discovery = self.config.discovery
            records = discover_samples(
                self.config.dataset_root,
                metadata_manifest=self.config.metadata_manifest,
                min_episode_frames=discovery.min_episode_frames,
                required_views=tuple(discovery.required_views),
            )
        records = list(records)
        if len({record.sample_id for record in records}) != len(records):
            raise ValueError("Batch selection contains duplicate sample ids")

        counts = {"succeeded": 0, "skipped": 0}
        completed_records: list[SampleRecord] = []
        pending = list(records)
        final_errors: dict[str, str] = {}
        retry_rounds = self.config.batch.failed_retry_rounds
        for retry_round in range(retry_rounds + 1):
            round_failures: list[SampleRecord] = []
            with ThreadPoolExecutor(
                max_workers=self.config.batch.concurrency
            ) as executor:
                futures = {
                    executor.submit(self._process, record): record for record in pending
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    record = futures[future]
                    try:
                        status = future.result()
                        counts[status] += 1
                        completed_records.append(record)
                        final_errors.pop(record.sample_id, None)
                        print(
                            f"[round {retry_round + 1} "
                            f"{completed}/{len(pending)}] {status}: "
                            f"{record.sample_id}",
                            flush=True,
                        )
                    except Exception as error:
                        final_round = retry_round >= retry_rounds
                        round_failures.append(record)
                        final_errors[record.sample_id] = (
                            f"{type(error).__name__}: {error}"
                        )
                        self._append_jsonl(
                            self.config.output_root / "failures.jsonl",
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "sample_id": record.sample_id,
                                "retry_round": retry_round,
                                "final": final_round,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                        )
                        print(
                            f"[round {retry_round + 1} "
                            f"{completed}/{len(pending)}] failed: "
                            f"{record.sample_id}: {type(error).__name__}: {error}",
                            flush=True,
                        )
            pending = round_failures
            if not pending:
                break
            if retry_round < retry_rounds:
                delay = self.config.batch.failed_retry_backoff_seconds
                print(
                    f"retrying {len(pending)} failed samples only "
                    f"(round {retry_round + 2}/{retry_rounds + 1}, "
                    f"delay={delay}s)",
                    flush=True,
                )
                if delay:
                    time.sleep(delay)

        exported_datasets, exported_prompts = self.dataset_exporter.export(
            completed_records
        )
        if self.config.dataset_prompt_export.enabled:
            print(
                f"exported {exported_prompts} prompt rows to "
                f"{exported_datasets} dataset metadata files",
                flush=True,
            )

        incomplete = self._audit_completion(records)
        incomplete_ids = [item["sample_id"] for item in incomplete]
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": len(records),
            "complete": not incomplete,
            "complete_samples": len(records) - len(incomplete),
            "incomplete_samples": incomplete,
            "final_errors": final_errors,
        }
        self._atomic_write(
            self.config.output_root / "logs/completion_report.json",
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        self._atomic_write(
            self.config.output_root / "logs/incomplete_samples.jsonl",
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in incomplete
            ),
        )
        failed_ids = {record.sample_id for record in pending}
        if self.config.batch.require_complete:
            failed_ids.update(incomplete_ids)
        if self.config.batch.require_complete and incomplete:
            print(
                f"completion check failed: {len(incomplete)}/{len(records)} "
                "selected samples are incomplete; see logs/incomplete_samples.jsonl",
                flush=True,
            )
        return BatchResult(
            total=len(records),
            succeeded=counts["succeeded"],
            failed=len(failed_ids),
            skipped=counts["skipped"],
            exported_datasets=exported_datasets,
            exported_prompts=exported_prompts,
            complete=not incomplete,
            incomplete_samples=incomplete_ids,
        )
