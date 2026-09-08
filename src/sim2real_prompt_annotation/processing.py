"""End-to-end paired Sim/Real validation, annotation, and export orchestration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .api import PromptAnnotationPipeline
from .config import PipelineConfig
from .dataset_validation import inspect_dataset
from .qwen import VLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


class DatasetProcessingPipeline:
    """The single public pipeline for a paired LeRobot dataset."""

    def __init__(
        self,
        dataset: str | Path,
        *,
        config: str | Path | Mapping[str, Any] | PipelineConfig | None = None,
        output_root: str | Path | None = None,
        client: VLMClient | None = None,
    ) -> None:
        self.dataset = Path(dataset).expanduser().resolve()
        self.output_root = (
            Path(output_root).expanduser().resolve()
            if output_root
            else (DEFAULT_OUTPUT_ROOT / self.dataset.name)
        )
        self.report_path = self.output_root / "data_processing_report.json"
        self.annotation = PromptAnnotationPipeline(
            config,
            dataset_root=self.dataset,
            output_root=self.output_root,
            client=client,
        )

    @staticmethod
    def _atomic_report(path: Path, report: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    def run(
        self,
        *,
        check_only: bool = False,
        force: bool = False,
        probe_videos: bool = True,
    ) -> dict[str, Any]:
        before = inspect_dataset(self.dataset, probe_videos=probe_videos)
        report: dict[str, Any] = {
            "dataset": str(self.dataset),
            "output_root": str(self.output_root),
            "mode": "check_only" if check_only else "apply",
            "before": before,
            "actions": [],
        }
        if not before.get("core_valid", False):
            report.update(status="blocked", reason="Core dataset validation failed")
        elif check_only:
            report["status"] = (
                "complete" if before["pipeline_complete"] else "needs_processing"
            )
        else:
            try:
                result = self.annotation.run(force=force)
            except Exception as error:
                report["actions"].append(
                    {
                        "step": "multi_reference_and_prompt",
                        "status": "blocked",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                report["status"] = "blocked"
                self._atomic_report(self.report_path, report)
                return report
            report["actions"].append(
                {
                    "step": "multi_reference_and_prompt",
                    "status": "completed" if result.get("complete") else "partial",
                    **result,
                }
            )
            after = inspect_dataset(self.dataset, probe_videos=False)
            report["result"] = result
            report["after"] = after
            report["status"] = (
                "complete" if after["pipeline_complete"] else "needs_processing"
            )
        self._atomic_report(self.report_path, report)
        return report
