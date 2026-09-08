"""Concurrent linear pipeline for project-specific prompt annotation."""

from __future__ import annotations

import hashlib
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
from .media import (
    MediaPreparer,
    PreparedMedia,
)
from .models import StructuredAnnotation, ValidationResult
from .qwen import QwenOpenAIClient, ResponseParseError, VLMClient, VLMResponse
from .references import build_reference_artifacts
from .renderer import PromptRenderer
from .task_metadata import canonical_robot_description, task_contract
from .validation import (
    canonicalize_annotation,
    find_local_issues,
)


def _validated_reference_row(record: SampleRecord, path: Path) -> dict[str, Any]:
    """Load one canonical Multi-Reference sidecar and verify its image identities."""

    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict) or row.get("episode_index") != record.episode_index:
        raise ValueError(f"invalid episode identity in {path}")
    references = row.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError(f"missing Multi-Reference list in {path}")
    reference_ids: list[str] = []
    dataset_root = record.dataset_root.resolve()
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            raise ValueError(f"Reference {index} in {path} must be an object")
        reference_id = reference.get("reference_id")
        relative_path = reference.get("reference_path")
        if not isinstance(reference_id, str) or not reference_id:
            raise ValueError(f"Reference {index} in {path} has no reference_id")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Reference {index} in {path} has no reference_path")
        image_path = (dataset_root / relative_path).resolve()
        try:
            image_path.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(
                f"Reference path escapes dataset root: {image_path}"
            ) from error
        payload = image_path.read_bytes()
        expected_digest = reference.get("sha256")
        if (
            isinstance(expected_digest, str)
            and hashlib.sha256(payload).hexdigest() != expected_digest
        ):
            raise ValueError(f"Reference digest mismatch: {image_path}")
        reference_ids.append(reference_id)
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError(f"duplicate reference_id in {path}")
    return row


class AnnotationService:
    """Run the single annotation API call with transport/schema retries."""

    def __init__(self, config: PipelineConfig, client: VLMClient):
        self.config = config
        self.client = client
        self.annotation_prompt = self._load_prompt(config.annotation.system_prompt)

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
            except Exception as error:
                status_code = getattr(error, "status_code", None)
                if (
                    isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code not in {408, 409, 429}
                ):
                    raise
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

    def _annotation_constraints(self) -> str:
        return (
            "\n\nPROMPT-CRITICAL SLOT CONSTRAINTS:\n"
            "- task.semantics is derived ONLY from task/robot metadata. Videos and "
            "Reference must never add an action, manipulated object, destination, "
            "instrument, or completion requirement to these slots.\n"
            "- Copy metadata_task exactly. Every action/object/goal/constraint is a "
            "{text, metadata_span} object. Use a short supporting metadata fragment "
            "when the raw task is separable; compact concatenated text and reasonable "
            "overlap are allowed. Use one lowercase English base verb in action.text.\n"
            "- primary_objects contains only atomic metadata-named task objects; "
            "goal contains the destination and constraints contain only "
            "metadata-stated facts. Do not also list the goal destination as a "
            "manipulated object.\n"
            "- Visible objects absent from task metadata are incidental clutter. They "
            "may appear in detailed task.objects but never in task.semantics.\n"
            "- target_visuals uses short, coarse appearance phrases: workspace at "
            "most 6 words, background at most 8, and lighting at most 6. Do not list "
            "incidental objects or frame-specific states.\n"
            "- The supplied Reference is the exact Real first-frame crop source. "
            "reference_candidates must include separate normalized boxes for every "
            "visible task object and useful robot/workspace/background/environment "
            "region. It never supplies task semantics or dynamics.\n"
            "- The final renderer emits one natural video-content sentence without "
            "Match/Render/reference-image instructions.\n"
            f"- The deterministic renderer targets "
            f"{self.config.renderer.target_prompt_words} words and enforces at most "
            f"{self.config.renderer.max_prompt_words} words and "
            f"{self.config.renderer.max_prompt_characters} characters in total.\n"
            "- Return structured slots, not final Prompt sentences or renderer "
            "wrappers."
        )

    @staticmethod
    def _fix_identifiers(
        annotation: StructuredAnnotation,
        record: SampleRecord,
        media: PreparedMedia,
    ) -> StructuredAnnotation:
        semantics = annotation.task.semantics
        contract = task_contract(record.task)
        semantics = semantics.model_copy(
            update={
                "metadata_task": record.task or semantics.metadata_task,
                "robot": canonical_robot_description(
                    record.robot_type, semantics.robot
                ),
                "active_arm": contract.active_arm,
            }
        )
        updates: dict[str, Any] = {
            "sample_id": record.sample_id,
            "task": annotation.task.model_copy(update={"semantics": semantics}),
        }
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
    ) -> VLMResponse:
        metadata = json.dumps(
            self._prompt_metadata(record, media),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        user_text = (
            "Create one source-separated structured annotation for this paired "
            "episode. Do not write a final Prompt.\n\n"
            f"AUTHORITATIVE METADATA JSON:\n{metadata}" + self._annotation_constraints()
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


class DatasetPromptExporter:
    """Merge one prompt and its exact Reference identity into dataset metadata."""

    def __init__(self, config: DatasetPromptExportConfig, output_root: Path):
        self.config = config
        self.output_root = output_root

    def _prompt_path(self, record: SampleRecord) -> Path:
        return self.output_root / "prompts" / f"{record.sample_id}.txt"

    def _annotation_path(self, record: SampleRecord) -> Path:
        return self.output_root / "annotations" / f"{record.sample_id}.json"

    def _reference_path(self, record: SampleRecord) -> Path:
        return self.output_root / "references" / f"{record.sample_id}.json"

    @staticmethod
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
                    raise ValueError(f"{path}:{line_number}: expected an object")
                rows.append(row)
        return rows

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
                reference_ids = row.get("reference_ids")
                view = row.get("reference_view")
                frame = row.get("reference_frame_index")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        f"{path}:{line_number}: expected one non-empty prompt string; "
                        "legacy prompt variants are not supported"
                    )
                if reference_ids is not None and (
                    not isinstance(reference_ids, list)
                    or not reference_ids
                    or any(
                        not isinstance(value, str) or not value
                        for value in reference_ids
                    )
                ):
                    raise ValueError(f"{path}:{line_number}: invalid reference_ids")
                if reference_ids is None and (
                    not isinstance(view, str)
                    or not view.strip()
                    or not isinstance(frame, int)
                    or frame < 0
                ):
                    raise ValueError(
                        f"{path}:{line_number}: expected Reference identity"
                    )
                parsed = {
                    "episode_index": episode_index,
                    "prompt": prompt.strip(),
                }
                if reference_ids is not None:
                    parsed["reference_ids"] = reference_ids
                else:
                    parsed["reference_view"] = view.strip()
                    parsed["reference_frame_index"] = frame
                result[episode_index] = parsed
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
            if self.config.merge_existing:
                try:
                    rows = self.read_existing(destination)
                except (OSError, ValueError, json.JSONDecodeError):
                    rows = {}
            else:
                rows = {}
            reference_destination = dataset_root / "meta/reference_images.jsonl"
            reference_rows: dict[int, dict[str, Any]] = {}
            if self.config.merge_existing and reference_destination.is_file():
                with reference_destination.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            value = json.loads(line)
                            reference_rows[int(value["episode_index"])] = value
            updated = 0
            for record in dataset_records:
                prompt_path = self._prompt_path(record)
                annotation_path = self._annotation_path(record)
                reference_path = self._reference_path(record)
                if (
                    not prompt_path.is_file()
                    or not annotation_path.is_file()
                    or not reference_path.is_file()
                ):
                    continue
                prompt = prompt_path.read_text(encoding="utf-8").strip()
                if not prompt:
                    raise ValueError(f"Final prompt is empty: {prompt_path}")
                StructuredAnnotation.model_validate_json(
                    annotation_path.read_text(encoding="utf-8")
                )
                reference_row = _validated_reference_row(record, reference_path)
                reference_ids = [
                    item["reference_id"] for item in reference_row["references"]
                ]
                rows[record.episode_index] = {
                    "episode_index": record.episode_index,
                    "prompt": prompt,
                    "reference_ids": reference_ids,
                }
                reference_rows[record.episode_index] = reference_row
                updated += 1
            if not updated:
                continue
            self._atomic_write_rows(destination, rows)
            self._atomic_write_rows(reference_destination, reference_rows)
            dataset_count += 1
            prompt_count += updated
        return dataset_count, prompt_count


def audit_completion(
    config: PipelineConfig,
    records: Iterable[SampleRecord],
) -> list[dict[str, Any]]:
    """Validate canonical outputs and their consolidated training rows once."""

    renderer = PromptRenderer(config.renderer)
    exporter = DatasetPromptExporter(config.dataset_prompt_export, config.output_root)
    exported_tables: dict[Path, dict[int, dict[str, Any]]] = {}
    exported_reference_tables: dict[Path, dict[int, dict[str, Any]]] = {}
    incomplete: list[dict[str, Any]] = []
    for record in records:
        reasons: list[str] = []
        annotation: StructuredAnnotation | None = None
        rendered: str | None = None
        annotation_path = (
            config.output_root / "annotations" / f"{record.sample_id}.json"
        )
        try:
            annotation = StructuredAnnotation.model_validate_json(
                annotation_path.read_text(encoding="utf-8")
            )
            if annotation.sample_id != record.sample_id:
                reasons.append("canonical sample_id mismatch")
            rendered = renderer.render(annotation)
        except (OSError, ValueError) as error:
            reasons.append(f"invalid/missing canonical annotation: {error}")

        prompt: str | None = None
        prompt_path = config.output_root / "prompts" / f"{record.sample_id}.txt"
        try:
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            if not prompt:
                reasons.append("empty final prompt")
            elif rendered is not None and prompt != rendered:
                reasons.append("prompt differs from deterministic render")
        except OSError as error:
            reasons.append(f"missing final prompt: {error}")

        reference_ids: list[str] = []
        reference_path = config.output_root / "references" / f"{record.sample_id}.json"
        try:
            reference_row = _validated_reference_row(record, reference_path)
            reference_ids = [
                reference["reference_id"] for reference in reference_row["references"]
            ]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            reasons.append(f"invalid/missing canonical Multi-Reference: {error}")

        if config.dataset_prompt_export.enabled:
            destination = (
                record.dataset_root / "meta" / config.dataset_prompt_export.filename
            )
            try:
                if destination not in exported_tables:
                    exported_tables[destination] = exporter.read_existing(destination)
                row = exported_tables[destination].get(record.episode_index)
            except (OSError, ValueError) as error:
                reasons.append(f"cannot validate consolidated prompt file: {error}")
                row = None
            if row is None:
                reasons.append("consolidated prompt row is missing")
            elif annotation is not None and row["prompt"] != prompt:
                reasons.append("consolidated prompt differs from canonical output")
            elif row.get("reference_ids") != reference_ids:
                reasons.append("consolidated prompt has different Reference identities")

            reference_destination = record.dataset_root / "meta/reference_images.jsonl"
            try:
                if reference_destination not in exported_reference_tables:
                    exported_reference_tables[reference_destination] = {
                        int(item["episode_index"]): item
                        for item in DatasetPromptExporter._read_jsonl(
                            reference_destination
                        )
                    }
                consolidated_reference = exported_reference_tables[
                    reference_destination
                ].get(record.episode_index)
                consolidated_ids = [
                    reference["reference_id"]
                    for reference in consolidated_reference["references"]
                ]
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                reasons.append(f"cannot validate consolidated Reference file: {error}")
            else:
                if consolidated_ids != reference_ids:
                    reasons.append(
                        "consolidated Reference identities differ from canonical output"
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


@dataclass(frozen=True)
class BatchResult:
    total: int
    succeeded: int
    failed: int
    skipped: int
    excluded: int
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
            "references",
            "validations",
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

    def _reference_path(self, sample_id: str) -> Path:
        return self.config.output_root / "references" / f"{sample_id}.json"

    def _render_existing(self, record: SampleRecord) -> bool:
        sample_id = record.sample_id
        path = self._annotation_path(sample_id)
        reference_path = self._reference_path(sample_id)
        if not path.is_file() or not reference_path.is_file():
            return False
        try:
            annotation = StructuredAnnotation.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if annotation.sample_id != sample_id:
                return False
            _validated_reference_row(record, reference_path)
            prompt = self.renderer.render(annotation)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        output = self._prompt_path(sample_id)
        try:
            current = output.read_text(encoding="utf-8").strip()
        except OSError:
            current = None
        if current != prompt:
            self._atomic_write(output, prompt + "\n")
        return True

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

    def _quality_check(
        self,
        record: SampleRecord,
        annotation: StructuredAnnotation,
        rendered_prompt: str,
    ) -> ValidationResult:
        """Run deterministic local checks; no second model call is involved."""

        return find_local_issues(
            annotation,
            record.annotation_metadata(),
            rendered_prompt,
            renderer=self.renderer,
        )

    def _exclude(
        self,
        record: SampleRecord,
        reason: str,
        validation: ValidationResult | None = None,
    ) -> str:
        row: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_id": record.sample_id,
            "dataset_root": str(record.dataset_root),
            "episode_index": record.episode_index,
            "reason": reason,
        }
        if validation is not None:
            row["validation"] = validation.model_dump()
        self._append_jsonl(
            self.config.output_root / "logs/excluded_samples.jsonl",
            row,
        )
        return "excluded"

    def _process(self, record: SampleRecord) -> str:
        if self.config.batch.skip_completed and self._render_existing(record):
            return "skipped"

        if not record.task:
            return self._exclude(record, "missing authoritative task metadata")
        if not record.robot_type:
            return self._exclude(record, "missing authoritative robot_type metadata")

        media = self.media_preparer.prepare(record)

        response = self.service.annotate(record, media)
        self._log_usage(record.sample_id, "annotation", response)
        annotation = StructuredAnnotation.model_validate(response.payload.model_dump())
        canonical = canonicalize_annotation(annotation)
        try:
            reference_artifacts = build_reference_artifacts(
                record, canonical.reference_candidates, self.config.media
            )
        except ValueError as error:
            return self._exclude(record, str(error))
        prompt = self.renderer.render(canonical)
        validation = self._quality_check(record, canonical, prompt)
        self._atomic_write(
            self.config.output_root / "validations" / f"{record.sample_id}.json",
            validation.model_dump_json(indent=2) + "\n",
        )
        if validation.has_severe_errors():
            severe_count = sum(issue.severity == "error" for issue in validation.issues)
            return self._exclude(
                record,
                f"quality check rejected candidate with {severe_count} error(s)",
                validation,
            )

        for artifact in reference_artifacts:
            artifact.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact.path.with_name(
                f".{artifact.path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
            )
            temporary.write_bytes(artifact.jpeg)
            os.replace(temporary, artifact.path)
        self._atomic_write(
            self._reference_path(record.sample_id),
            json.dumps(
                {
                    "schema_version": 2,
                    "episode_index": record.episode_index,
                    "reference_seed": self.config.media.reference_seed,
                    "references": [artifact.row for artifact in reference_artifacts],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

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

        counts = {"succeeded": 0, "skipped": 0, "excluded": 0}
        completed_records: list[SampleRecord] = []
        excluded_ids: set[str] = set()
        final_errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.config.batch.concurrency) as executor:
            futures = {
                executor.submit(self._process, record): record for record in records
            }
            for completed, future in enumerate(as_completed(futures), 1):
                record = futures[future]
                try:
                    status = future.result()
                    counts[status] += 1
                    if status == "excluded":
                        excluded_ids.add(record.sample_id)
                    else:
                        completed_records.append(record)
                    print(
                        f"[{completed}/{len(records)}] {status}: {record.sample_id}",
                        flush=True,
                    )
                except Exception as error:
                    final_errors[record.sample_id] = f"{type(error).__name__}: {error}"
                    self._append_jsonl(
                        self.config.output_root / "logs/failures.jsonl",
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "sample_id": record.sample_id,
                            "dataset_root": str(record.dataset_root),
                            "episode_index": record.episode_index,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
                    print(
                        f"[{completed}/{len(records)}] failed: {record.sample_id}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )

        exported_datasets, exported_prompts = self.dataset_exporter.export(
            completed_records
        )
        if self.config.dataset_prompt_export.enabled:
            print(
                f"exported {exported_prompts} prompt rows to "
                f"{exported_datasets} dataset metadata files",
                flush=True,
            )

        incomplete = audit_completion(self.config, records)
        incomplete_ids = [item["sample_id"] for item in incomplete]
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": len(records),
            "complete": not incomplete,
            "complete_samples": len(records) - len(incomplete),
            "excluded_samples": sorted(excluded_ids),
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
        failed_ids = set(final_errors) | (set(incomplete_ids) - excluded_ids)
        if incomplete:
            print(
                f"completion check: {len(incomplete)}/{len(records)} selected "
                "samples have no accepted output; see logs/incomplete_samples.jsonl",
                flush=True,
            )
        return BatchResult(
            total=len(records),
            succeeded=counts["succeeded"],
            failed=len(failed_ids),
            skipped=counts["skipped"],
            excluded=counts["excluded"],
            exported_datasets=exported_datasets,
            exported_prompts=exported_prompts,
            complete=not incomplete,
            incomplete_samples=incomplete_ids,
        )
