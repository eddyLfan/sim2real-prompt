"""Single supported Python interface for the annotation package."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import PipelineConfig, load_config
from .lerobot import SampleRecord, discover_samples
from .media import MediaPreparer
from .models import StructuredAnnotation, ValidationResult
from .pipeline import BatchPipeline, DatasetPromptExporter
from .qwen import VLMClient
from .renderer import PromptRenderer


def _parse_episodes(value: str | Iterable[int] | None) -> set[int] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return {int(item) for item in value}
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            result.add(int(part))
            continue
        start_text, end_text = part.split("-", 1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError(f"Invalid episode range: {part}")
        result.update(range(start, end + 1))
    return result


def _sample_summary(record: SampleRecord) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "dataset": record.dataset_name,
        "episode": record.episode_index,
        "length": record.episode_length,
        "fps": record.fps,
        "robot_type": record.robot_type,
        "task": record.task,
        "paired_views": list(record.paired_views),
        "subtask_count": len(record.subtasks),
    }


class PromptAnnotationPipeline:
    """Inspect, annotate, audit, and render paired LeRobot episodes.

    This is the package's only supported public Python interface. ``config`` may
    be a YAML path, a validated ``PipelineConfig``, a mapping, or ``None`` for
    validated defaults. The VLM client is created lazily, so inspect, audit,
    schema, render, and dry-run operations never require an API key.
    """

    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | PipelineConfig | None = None,
        *,
        dataset_root: str | Path | None = None,
        output_root: str | Path | None = None,
        client: VLMClient | None = None,
    ) -> None:
        if config is None:
            parsed = PipelineConfig()
        elif isinstance(config, PipelineConfig):
            parsed = config.model_copy(deep=True)
        elif isinstance(config, Mapping):
            parsed = PipelineConfig.model_validate(dict(config))
        else:
            parsed = load_config(config)
        updates: dict[str, Any] = {}
        if dataset_root is not None:
            updates["dataset_root"] = Path(dataset_root).expanduser().resolve()
        if output_root is not None:
            updates["output_root"] = Path(output_root).expanduser().resolve()
        self._config = parsed.model_copy(update=updates)
        self._client = client

    def _records(
        self,
        *,
        dataset_glob: str = "*",
        episodes: str | Iterable[int] | None = None,
        limit: int | None = None,
    ) -> list[SampleRecord]:
        selection = self._config.discovery
        return discover_samples(
            self._config.dataset_root,
            dataset_glob=dataset_glob,
            episodes=_parse_episodes(episodes),
            limit=limit,
            metadata_manifest=self._config.metadata_manifest,
            min_episode_frames=selection.min_episode_frames,
            required_views=tuple(selection.required_views),
            require_later_window=selection.require_later_window,
            window_stride=selection.window_stride,
        )

    def inspect(
        self,
        *,
        dataset_glob: str = "*",
        episodes: str | Iterable[int] | None = None,
        limit: int | None = None,
        show: int = 3,
    ) -> dict[str, Any]:
        records = self._records(
            dataset_glob=dataset_glob, episodes=episodes, limit=limit
        )
        return {
            "dataset_root": str(self._config.dataset_root),
            "sample_count": len(records),
            "datasets": dict(Counter(record.dataset_name for record in records)),
            "first_samples": [_sample_summary(record) for record in records[:show]],
        }

    def run(
        self,
        *,
        dataset_glob: str = "*",
        episodes: str | Iterable[int] | None = None,
        limit: int | None = None,
        force: bool = False,
        dry_run: bool = False,
        prepare_media: bool = False,
        show: int = 3,
    ) -> dict[str, Any]:
        records = self._records(
            dataset_glob=dataset_glob, episodes=episodes, limit=limit
        )
        if not records:
            raise ValueError("No paired LeRobot samples matched the selection")
        if dry_run:
            prepared = []
            if prepare_media:
                preparer = MediaPreparer(self._config.media)
                for record in records:
                    media = preparer.prepare(record)
                    prepared.append(
                        {
                            "sample_id": record.sample_id,
                            "media_groups": len(media.groups),
                            "has_reference": media.reference is not None,
                        }
                    )
            return {
                "dry_run": True,
                "sample_count": len(records),
                "provider": self._config.provider.name,
                "model": self._config.provider.model,
                "media_mode": self._config.media.mode,
                "samples": [_sample_summary(record) for record in records[:show]],
                "prepared": prepared,
            }
        run_config = self._config.model_copy(deep=True)
        if force:
            run_config.batch.skip_completed = False
        result = BatchPipeline(run_config, client=self._client).run(records)
        return asdict(result)

    def audit(
        self,
        *,
        dataset_glob: str = "*",
        episodes: str | Iterable[int] | None = None,
        limit: int | None = None,
        show: int = 10,
    ) -> dict[str, Any]:
        records = self._records(
            dataset_glob=dataset_glob, episodes=episodes, limit=limit
        )
        renderer = PromptRenderer(self._config.renderer)
        required = tuple(self._config.dataset_prompt_export.variants)
        tables: dict[Path, dict[int, dict[str, Any]]] = {}
        incomplete = []
        for record in records:
            reasons = []
            annotation_path = (
                self._config.output_root
                / "annotations"
                / f"{record.sample_id}.json"
            )
            rendered = None
            try:
                annotation = StructuredAnnotation.model_validate_json(
                    annotation_path.read_text(encoding="utf-8")
                )
                if annotation.sample_id != record.sample_id:
                    reasons.append("canonical sample_id mismatch")
                rendered = renderer.render_all(annotation)
            except (OSError, ValueError) as error:
                reasons.append(f"invalid/missing canonical annotation: {error}")
            output_prompts: dict[str, str] = {}
            for variant in required:
                path = (
                    self._config.output_root
                    / "prompts"
                    / variant
                    / f"{record.sample_id}.txt"
                )
                try:
                    output_prompts[variant] = path.read_text(
                        encoding="utf-8"
                    ).strip()
                except OSError as error:
                    reasons.append(f"missing {variant} prompt: {error}")
                    continue
                if not output_prompts[variant]:
                    reasons.append(f"empty {variant} prompt")
                elif rendered is not None and output_prompts[variant] != rendered.get(
                    variant
                ):
                    reasons.append(
                        f"{variant} prompt differs from deterministic render"
                    )
            if self._config.dataset_prompt_export.enabled:
                table = tables.get(record.dataset_root)
                if table is None:
                    destination = (
                        record.dataset_root
                        / "meta"
                        / self._config.dataset_prompt_export.filename
                    )
                    try:
                        table = DatasetPromptExporter.read_existing(destination)
                    except (OSError, ValueError) as error:
                        reasons.append(
                            f"cannot validate consolidated prompt file: {error}"
                        )
                        table = {}
                    tables[record.dataset_root] = table
                exported = table.get(record.episode_index)
                if exported is None:
                    reasons.append(
                        "training episodes_prompt.jsonl row is missing/incomplete"
                    )
                elif any(
                    exported.get("prompts", {}).get(name)
                    != output_prompts.get(name)
                    for name in required
                ):
                    reasons.append(
                        "training export differs from rendered prompt files"
                    )
            if reasons:
                incomplete.append(
                    {
                        "sample_id": record.sample_id,
                        "dataset": record.dataset_name,
                        "episode": record.episode_index,
                        "reasons": reasons,
                    }
                )
        return {
            "complete": not incomplete,
            "selected": len(records),
            "ready": len(records) - len(incomplete),
            "incomplete": len(incomplete),
            "first_incomplete": incomplete[:show],
        }

    def render(
        self,
        annotation: str | Path | Mapping[str, Any] | StructuredAnnotation,
        *,
        variant: str = "all",
    ) -> str | dict[str, str]:
        if isinstance(annotation, StructuredAnnotation):
            parsed = annotation
        elif isinstance(annotation, Mapping):
            parsed = StructuredAnnotation.model_validate(dict(annotation))
        else:
            value = str(annotation)
            if value.lstrip().startswith("{"):
                raw = value
            else:
                candidate = Path(value)
                raw = candidate.read_text(encoding="utf-8")
            parsed = StructuredAnnotation.model_validate_json(raw)
        renderer = PromptRenderer(self._config.renderer)
        return renderer.render_all(parsed) if variant == "all" else renderer.render(
            parsed, variant
        )

    @staticmethod
    def schemas() -> dict[str, dict[str, Any]]:
        return {
            "annotation": StructuredAnnotation.model_json_schema(),
            "critique": ValidationResult.model_json_schema(),
        }
