from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from sim2real_prompt_annotation import PromptAnnotationPipeline
from sim2real_prompt_annotation.media import MediaPreparer
from sim2real_prompt_annotation.models import (
    EvidenceText,
    PromptPlan,
    ReferenceDescription,
    SimInvariants,
    StructuredAnnotation,
    TargetVisuals,
    TaskDescription,
    TaskObject,
    ValidationResult,
)
from sim2real_prompt_annotation.qwen import VLMClient, VLMResponse


def _write_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create the synthetic test video")
    for color in colors:
        writer.write(np.full((48, 64, 3), color, dtype=np.uint8))
    writer.release()


def _make_dataset(root: Path) -> Path:
    dataset = root / "paired_demo"
    (dataset / "meta").mkdir(parents=True)
    template = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
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
        json.dumps({"episode_index": 0, "length": 6, "tasks": ["place mug"]}) + "\n",
        encoding="utf-8",
    )
    colors = [(index * 20, 50, 100) for index in range(6)]
    _write_video(dataset / "videos/chunk-000/camera_head/episode_000000.mp4", colors)
    _write_video(
        dataset / "videos/chunk-000/camera_head_sim/episode_000000.mp4",
        list(reversed(colors)),
    )
    return dataset


def _field(
    text: str, source: str, evidence: list[str], confidence: float = 0.95
) -> EvidenceText:
    return EvidenceText(
        text=text,
        source=source,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=evidence,
    )


def _annotation(
    sample_id: str, reference_view: str, reference_frame_index: int
) -> StructuredAnnotation:
    return StructuredAnnotation(
        sample_id=sample_id,
        task=TaskDescription(
            summary=_field(
                "Place the mug on the tray.",
                "metadata",
                ["meta/episodes.jsonl:tasks"],
            ),
            robot=_field(
                "A dual-arm robot.",
                "metadata",
                ["meta/info.json:robot_type"],
            ),
            objects=[
                TaskObject(
                    role="manipulated object",
                    identity="mug",
                    state="placed on the tray at the end",
                    appearance="blue ceramic",
                    geometry_affordance="rigid handled vessel",
                    source="pair",
                    confidence=0.95,
                    evidence=[
                        "sim:camera_head:frame_000000",
                        "real:camera_head:frame_000000",
                    ],
                )
            ],
        ),
        sim_invariants=SimInvariants(
            robot_motion=_field(
                "The robot reaches, grasps, transports, and releases the mug.",
                "sim",
                ["sim:camera_head:frame_000000"],
            ),
            object_motion=_field(
                "The mug moves with the gripper to the tray.",
                "sim",
                ["sim:camera_head:frame_000000"],
            ),
        ),
        target_visuals=TargetVisuals(
            robot_appearance=_field(
                "dark dual-arm robot",
                "real",
                ["real:camera_head:frame_000000"],
            ),
            workspace=_field(
                "gray workbench",
                "real",
                ["real:camera_head:frame_000000"],
            ),
            background=_field(
                "robotics laboratory",
                "real",
                ["real:camera_head:frame_000000"],
            ),
            lighting=_field(
                "soft overhead lighting",
                "real",
                ["real:camera_head:frame_000000"],
            ),
        ),
        reference=ReferenceDescription(
            view=reference_view,
            frame_index=reference_frame_index,
            visible_content=["robot", "mug", "workbench", "laboratory background"],
            use_for=["robot", "objects", "workspace", "background"],
            unclear_or_occluded=[],
        ),
        prompt_plan=PromptPlan(
            task_clause=(
                "a dark dual-arm robot placing a blue ceramic mug on a white tray"
            ),
            setting_clauses=[
                "a gray workbench",
                "a robotics laboratory",
                "soft overhead lighting",
            ],
            reference_scopes=["robot", "objects", "workspace", "background"],
            text_overrides_reference=True,
        ),
    )


class MockClient(VLMClient):
    def generate(self, **kwargs) -> VLMResponse:  # type: ignore[no-untyped-def]
        model = kwargs["response_model"]
        if model is StructuredAnnotation:
            reference = kwargs["media"].reference
            assert reference is not None
            payload = _annotation(
                kwargs["sample_id"], reference.view, reference.frame_index
            )
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
    def test_lighting_cannot_be_selected_as_reference_scope(self) -> None:
        with self.assertRaises(ValidationError):
            ReferenceDescription(
                view="camera_head",
                frame_index=0,
                visible_content=["robot"],
                use_for=["lighting"],  # type: ignore[list-item]
                unclear_or_occluded=[],
            )

    def test_public_api_runs_resumes_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _make_dataset(root / "data")
            output = root / "outputs"
            pipeline = PromptAnnotationPipeline(
                {
                    "dataset_root": dataset.parent,
                    "output_root": output,
                    "media": {
                        "max_frames": 3,
                        "resize_long_edge": 64,
                        "reference_seed": 7,
                    },
                    "batch": {
                        "concurrency": 1,
                        "api_retry_count": 0,
                        "failed_retry_rounds": 0,
                    },
                },
                client=MockClient(),
            )

            references = pipeline.export_references(dataset_glob="paired_demo")
            self.assertEqual(references["written"], 1)
            reference_path = dataset / "Reference/episode_000000.jpg"
            self.assertTrue(reference_path.is_file())
            reference_rows = [
                json.loads(line)
                for line in (dataset / "meta/reference_images.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertEqual(len(reference_rows), 1)
            self.assertEqual(
                reference_rows[0]["reference_path"],
                reference_path.relative_to(dataset).as_posix(),
            )

            resumed_references = pipeline.export_references(dataset_glob="paired_demo")
            self.assertEqual(resumed_references["written"], 0)
            self.assertEqual(resumed_references["skipped"], 1)

            result = pipeline.run(dataset_glob="paired_demo")
            self.assertEqual(result["succeeded"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertTrue(result["complete"])

            sample_id = "paired_demo__episode_000000"
            prompt_table = dataset / "meta/episodes_prompt.jsonl"
            row = json.loads(prompt_table.read_text(encoding="utf-8"))
            self.assertIsInstance(row["prompt"], str)
            self.assertNotIn("prompts", row)
            self.assertEqual(row["reference_view"], "camera_head")
            self.assertGreaterEqual(row["reference_frame_index"], 0)
            self.assertLess(row["reference_frame_index"], 6)
            self.assertIn("explicit text attributes take priority", row["prompt"])
            self.assertIn("Render the scene with", row["prompt"])
            self.assertEqual(row["prompt"].count("."), 3)

            records = pipeline._records(dataset_glob="paired_demo")
            preparer = MediaPreparer(pipeline._config.media)
            first = preparer.prepare(records[0]).reference
            second = preparer.prepare(records[0]).reference
            assert first is not None and second is not None
            self.assertEqual(first.frame_index, second.frame_index)
            self.assertEqual(row["reference_frame_index"], first.frame_index)
            self.assertEqual(
                row["reference_frame_index"],
                reference_rows[0]["reference_frame_index"],
            )

            prompt_path = output / "prompts" / f"{sample_id}.txt"
            prompt_path.write_text("stale prompt\n", encoding="utf-8")
            resumed = pipeline.run(dataset_glob="paired_demo")
            self.assertEqual(resumed["skipped"], 1)
            self.assertTrue(pipeline.audit(dataset_glob="paired_demo")["complete"])

            rendered = pipeline.render(output / "annotations" / f"{sample_id}.json")
            restored = prompt_path.read_text(encoding="utf-8").strip()
            self.assertEqual(restored, rendered)
            self.assertEqual(rendered, row["prompt"])


if __name__ == "__main__":
    unittest.main()
