from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sim2real_prompt_annotation import PromptAnnotationPipeline
from sim2real_prompt_annotation.models import (
    AnnotationField,
    AppearanceEntry,
    StructuredAnnotation,
    ValidationResult,
)
from sim2real_prompt_annotation.qwen import VLMClient, VLMResponse


def _write_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create the synthetic test video")
    for color in colors:
        writer.write(np.full((48, 64, 3), color, dtype=np.uint8))
    writer.release()


def _make_dataset(root: Path) -> Path:
    dataset = root / "paired_demo"
    (dataset / "meta").mkdir(parents=True)
    template = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4"
    )
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "robot_type": "dual_arm",
                "fps": 10,
                "chunks_size": 1000,
                "video_path": template,
                "features": {
                    "camera_head": {"dtype": "video"},
                    "camera_head_sim": {"dtype": "video"},
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 6, "tasks": ["place mug"]})
        + "\n",
        encoding="utf-8",
    )
    colors = [(index * 20, 50, 100) for index in range(6)]
    _write_video(dataset / "videos/chunk-000/camera_head/episode_000000.mp4", colors)
    _write_video(
        dataset / "videos/chunk-000/camera_head_sim/episode_000000.mp4",
        list(reversed(colors)),
    )
    return dataset


def _annotation(sample_id: str) -> StructuredAnnotation:
    return StructuredAnnotation(
        sample_id=sample_id,
        robot=AnnotationField(
            text="A dual-arm robot with parallel-jaw grippers.",
            source="metadata",
            confidence=1.0,
            evidence=["metadata:robot_type"],
        ),
        task=AnnotationField(
            text="Place the mug on the target.",
            source="metadata",
            confidence=1.0,
            evidence=["metadata:task"],
        ),
        objects=AnnotationField(
            text="One handled mug and one placement target.",
            source="pair",
            confidence=0.95,
            evidence=["sim:camera_head:frame_000000"],
        ),
        environment=AnnotationField(
            text="An indoor robotics workspace.",
            source="real",
            confidence=0.9,
            evidence=["real:camera_head:frame_000000"],
        ),
        appearance=[
            AppearanceEntry(
                entity="mug",
                attributes=["glossy", "green", "ceramic"],
                source="reference",
                confidence=0.9,
                visible_in_reference=True,
                evidence=["reference:camera_head:frame_000000"],
            )
        ],
    )


class MockClient(VLMClient):
    def generate(self, **kwargs) -> VLMResponse:  # type: ignore[no-untyped-def]
        model = kwargs["response_model"]
        if model is StructuredAnnotation:
            payload = _annotation(kwargs["sample_id"])
        elif model is ValidationResult:
            payload = ValidationResult()
        else:
            raise AssertionError(model)
        return VLMResponse(
            payload=payload,
            raw_text=payload.model_dump_json(),
            model="mock-qwen",
            input_tokens=10,
            output_tokens=5,
            request_id="mock-request",
        )


class PipelineTest(unittest.TestCase):
    def test_public_api_runs_resumes_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _make_dataset(root / "data")
            output = root / "outputs"
            pipeline = PromptAnnotationPipeline(
                {
                    "dataset_root": dataset.parent,
                    "output_root": output,
                    "media": {"max_frames": 3, "resize_long_edge": 64},
                    "batch": {
                        "concurrency": 1,
                        "api_retry_count": 0,
                        "failed_retry_rounds": 0,
                    },
                },
                client=MockClient(),
            )

            result = pipeline.run(dataset_glob="paired_demo")
            self.assertEqual(result["succeeded"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertTrue(result["complete"])

            sample_id = "paired_demo__episode_000000"
            prompt_table = dataset / "meta/episodes_prompt.jsonl"
            row = json.loads(prompt_table.read_text(encoding="utf-8"))
            self.assertEqual(
                set(row["prompts"]), {"full", "reference", "semantic", "minimal"}
            )
            self.assertNotEqual(row["prompts"]["full"], row["prompts"]["reference"])
            self.assertNotIn("glossy green ceramic", row["prompts"]["reference"])

            full_path = output / "prompts/full" / f"{sample_id}.txt"
            full_path.write_text("stale prompt\n", encoding="utf-8")
            resumed = pipeline.run(dataset_glob="paired_demo")
            self.assertEqual(resumed["skipped"], 1)
            self.assertTrue(pipeline.audit(dataset_glob="paired_demo")["complete"])

            rendered = pipeline.render(output / "annotations" / f"{sample_id}.json")
            restored = full_path.read_text(encoding="utf-8").strip()
            self.assertEqual(restored, rendered["full"])
            self.assertEqual(rendered["full"], row["prompts"]["full"])


if __name__ == "__main__":
    unittest.main()
