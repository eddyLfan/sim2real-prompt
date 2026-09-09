"""Small public facade for inspection, processing, and product audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import PipelineConfig, load_config
from .dataset import discover_episodes
from .pipeline import PreprocessingPipeline
from .prompt_branch import PromptBranch
from .qwen import VLMClient
from .reference_branch import ReferenceBranch


def parse_episodes(value: str | Iterable[int] | None) -> set[int] | None:
    """Parse ``0,2,5-9`` or an integer iterable into an episode set."""

    if value is None:
        return None
    if not isinstance(value, str):
        result = {int(item) for item in value}
        if any(item < 0 for item in result):
            raise ValueError("episode indices must be non-negative")
        return result
    result: set[int] = set()
    for component in value.split(","):
        component = component.strip()
        if not component:
            continue
        if "-" not in component:
            result.add(int(component))
            continue
        start_text, end_text = component.split("-", 1)
        start, end = int(start_text), int(end_text)
        if start < 0 or end < start:
            raise ValueError(f"invalid episode range: {component}")
        result.update(range(start, end + 1))
    if any(item < 0 for item in result):
        raise ValueError("episode indices must be non-negative")
    return result


def _config(
    value: str | Path | Mapping[str, Any] | PipelineConfig | None,
) -> PipelineConfig:
    if value is None:
        return PipelineConfig()
    if isinstance(value, PipelineConfig):
        return value.model_copy(deep=True)
    if isinstance(value, Mapping):
        return PipelineConfig.model_validate(dict(value))
    return load_config(value)


class Sim2RealPreprocessingPipeline:
    """The only supported high-level Python API."""

    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | PipelineConfig | None = None,
        *,
        dataset_root: str | Path | None = None,
        output_root: str | Path | None = None,
        dataset_glob: str | None = None,
        vlm_client: VLMClient | None = None,
        prompt_branch: PromptBranch | None = None,
        reference_branch: ReferenceBranch | None = None,
    ) -> None:
        parsed = _config(config)
        dataset_updates: dict[str, Any] = {}
        if dataset_root is not None:
            dataset_updates["root"] = Path(dataset_root).expanduser().resolve()
        if dataset_glob is not None:
            dataset_updates["dataset_glob"] = dataset_glob
        if dataset_updates:
            parsed.dataset = parsed.dataset.model_copy(update=dataset_updates)
        if output_root is not None:
            parsed.output = parsed.output.model_copy(
                update={"root": Path(output_root).expanduser().resolve()}
            )
        self.config = parsed
        self._pipeline = PreprocessingPipeline(
            parsed,
            vlm_client=vlm_client,
            prompt_branch=prompt_branch,
            reference_branch=reference_branch,
        )

    def inspect(
        self,
        *,
        episodes: str | Iterable[int] | None = None,
        limit: int | None = None,
        show: int = 3,
    ) -> dict[str, Any]:
        records = discover_episodes(
            self.config.dataset,
            episodes=parse_episodes(episodes),
            limit=limit,
        )
        return {
            "dataset_root": str(self.config.dataset.root.resolve()),
            "real_view": self.config.dataset.real_view,
            "episode_count": len(records),
            "datasets": dict(Counter(item.dataset_name for item in records)),
            "domains": dict(Counter(item.domain for item in records)),
            "splits": dict(Counter(item.split for item in records)),
            "first_episodes": [
                {
                    "sample_id": item.sample_id,
                    "dataset": item.dataset_name,
                    "episode_index": item.episode_index,
                    "task": item.task,
                    "robot_type": item.robot_type,
                    "real_view": item.real_view,
                    "real_video": str(item.real_video),
                    "length": item.episode_length,
                }
                for item in records[: max(0, show)]
            ],
        }

    def run(
        self,
        *,
        episodes: str | Iterable[int] | None = None,
        limit: int | None = None,
        force: bool = False,
        audit: bool = True,
    ) -> dict[str, Any]:
        return self._pipeline.run(
            episodes=parse_episodes(episodes),
            limit=limit,
            force=force,
            audit=audit,
        )

    def audit(
        self,
        *,
        episodes: str | Iterable[int] | None = None,
        limit: int | None = None,
        show: int = 20,
    ) -> dict[str, Any]:
        return self._pipeline.audit(
            episodes=parse_episodes(episodes),
            limit=limit,
            show=show,
        )


# One release-cycle compatibility alias; both names use the new two-branch pipeline.
PromptAnnotationPipeline = Sim2RealPreprocessingPipeline
