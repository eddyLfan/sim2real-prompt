from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from sim2real_prompt_annotation import PromptAnnotationPipeline
from sim2real_prompt_annotation.media import (
    MediaGroup,
    MediaPreparer,
    PreparedFrame,
    PreparedMedia,
)
from sim2real_prompt_annotation.models import (
    EvidenceText,
    MetadataGroundedText,
    ReferenceCandidate,
    ReferenceDescription,
    SimInvariants,
    StructuredAnnotation,
    TargetVisuals,
    TaskDescription,
    TaskObject,
    TaskSemantics,
)
from sim2real_prompt_annotation.qwen import QwenOpenAIClient, VLMClient, VLMResponse
from sim2real_prompt_annotation.renderer import PromptLengthError, PromptRenderer
from sim2real_prompt_annotation.task_metadata import task_contract
from sim2real_prompt_annotation.validation import find_local_issues


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
        json.dumps(
            {
                "episode_index": 0,
                "length": 6,
                "tasks": ["place blue mug onto black tray"],
            }
        )
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


def _field(
    text: str, source: str, evidence: list[str], confidence: float = 0.95
) -> EvidenceText:
    return EvidenceText(
        text=text,
        source=source,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=evidence,
    )


def _ground(text: str, metadata_span: str) -> MetadataGroundedText:
    return MetadataGroundedText(text=text, metadata_span=metadata_span)


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
            semantics=TaskSemantics(
                metadata_task="place blue mug onto black tray",
                robot="dual-arm robot",
                active_arm="unspecified",
                action=_ground("place", "place"),
                primary_objects=[_ground("a blue mug", "blue mug")],
                goal=_ground("onto a black tray", "onto black tray"),
                constraints=[],
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
        reference_candidates=[
            ReferenceCandidate(
                scope="objects",
                label="blue mug",
                description="a blue ceramic mug",
                bbox_xyxy=(200, 200, 500, 800),
                confidence=0.95,
            ),
            ReferenceCandidate(
                scope="workspace",
                label="workbench",
                description="a gray workbench",
                bbox_xyxy=(50, 500, 950, 950),
                confidence=0.9,
            ),
        ],
    )


class MockClient(VLMClient):
    def generate(self, **kwargs) -> VLMResponse:  # type: ignore[no-untyped-def]
        model = kwargs["response_model"]
        if model is not StructuredAnnotation:
            raise AssertionError(model)
        reference = kwargs["media"].reference
        assert reference is not None
        payload = _annotation(
            kwargs["sample_id"], reference.view, reference.frame_index
        )
        return VLMResponse(
            payload=payload,
            raw_text=payload.model_dump_json(),
            model="mock-qwen",
            input_tokens=10,
            output_tokens=5,
            request_id="mock-request",
        )


class TrackingClient(MockClient):
    def __init__(self) -> None:
        self.stages: list[str] = []

    def generate(self, **kwargs) -> VLMResponse:  # type: ignore[no-untyped-def]
        self.stages.append(kwargs["stage"])
        return super().generate(**kwargs)


class NonRetryableClient(VLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs) -> VLMResponse:  # type: ignore[no-untyped-def]
        del kwargs
        self.calls += 1
        error = RuntimeError("invalid request")
        error.status_code = 400  # type: ignore[attr-defined]
        raise error


class PipelineTest(unittest.TestCase):
    def test_metadata_contract_supports_encoded_english_and_chinese_tasks(self) -> None:
        banana = task_contract(
            "align_the_preassembled_banana_bunch_handles_in_the_same_direction"
        )
        self.assertEqual(banana.active_arm, "unspecified")
        self.assertEqual(
            banana.task_payload,
            "align_the_preassembled_banana_bunch_handles_in_the_same_direction",
        )

        tray = task_contract("双臂搬运绿色托盘")
        self.assertEqual(tray.active_arm, "both")
        self.assertEqual(tray.active_arm_span, "双臂")

        cup = task_contract("左臂将绿色杯放到黑色杯垫")
        self.assertEqual(cup.active_arm, "left")
        self.assertEqual(cup.task_payload, "左臂将绿色杯放到黑色杯垫")

    def test_prompt_media_uses_real_first_frame_as_crop_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _make_dataset(root / "data")
            pipeline = PromptAnnotationPipeline(
                {"dataset_root": dataset.parent, "output_root": root / "outputs"}
            )
            record = pipeline._records(dataset_glob="paired_demo")[0]
            reference = MediaPreparer(pipeline._config.media).prepare(record).reference
            assert reference is not None
            self.assertEqual(reference.frame_index, 0)

    def test_pipeline_builds_references_without_preexisting_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _make_dataset(root / "data")
            output = root / "outputs"
            client = TrackingClient()
            pipeline = PromptAnnotationPipeline(
                {
                    "dataset_root": dataset.parent,
                    "output_root": output,
                    "batch": {"concurrency": 1, "api_retry_count": 0},
                },
                client=client,
            )

            result = pipeline.run(dataset_glob="paired_demo")

            self.assertEqual(result["excluded"], 0)
            self.assertEqual(result["failed"], 0)
            self.assertTrue(result["complete"])
            self.assertEqual(client.stages, ["annotation"])
            self.assertTrue(
                (dataset / "Reference/episode_000000/reference_00.jpg").is_file()
            )

    def test_frame_limits_match_qwen_video_contract(self) -> None:
        self.assertFalse(hasattr(PromptAnnotationPipeline()._config, "critic"))
        with self.assertRaises(ValidationError):
            PromptAnnotationPipeline({"media": {"max_frames": 3}})

    def test_short_frame_groups_use_ordered_images_instead_of_video(self) -> None:
        frames = tuple(
            PreparedFrame(
                evidence_id=f"sim:head:frame_{index:06d}",
                frame_index=index,
                timestamp_seconds=float(index),
                jpeg=b"jpeg",
            )
            for index in range(3)
        )
        media = PreparedMedia(
            groups=(MediaGroup(source="sim", view="head", frames=frames),),
            reference=None,
        )
        client = object.__new__(QwenOpenAIClient)
        content = client._content("annotate", media)
        self.assertFalse(any(item["type"] == "video" for item in content))
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 3)

    def test_prompt_uses_40_word_target_and_56_word_hard_limit(self) -> None:
        config = PromptAnnotationPipeline()._config.renderer
        self.assertEqual(config.target_prompt_words, 40)
        self.assertEqual(config.max_prompt_words, 56)
        renderer = PromptRenderer(config)
        renderer.validate_length(" ".join(["word"] * 56))
        with self.assertRaises(PromptLengthError):
            renderer.validate_length(" ".join(["word"] * 57))

    def test_reference_frame_supports_real_target_appearance(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        assert annotation.target_visuals.workspace is not None
        annotation.target_visuals.workspace.source = "reference"
        annotation.target_visuals.workspace.evidence = [
            "reference:camera_head:frame_000000"
        ]
        renderer = PromptRenderer(PromptAnnotationPipeline()._config.renderer)
        validation = find_local_issues(
            annotation,
            {"task": "place blue mug onto black tray", "robot_type": "dual_arm"},
            renderer.render(annotation),
            renderer=renderer,
        )
        self.assertFalse(
            any(
                issue.severity == "error" and issue.field == "target_visuals.workspace"
                for issue in validation.issues
            )
        )

    def test_detailed_evidence_prefix_is_not_a_prompt_qc_gate(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        annotation.sim_invariants.camera_and_timing = _field(
            "Static head camera captures the workspace.",
            "sim",
            ["real:camera_head:frame_000000"],
        )
        renderer = PromptRenderer(PromptAnnotationPipeline()._config.renderer)
        validation = find_local_issues(
            annotation,
            {"task": "place blue mug onto black tray", "robot_type": "dual_arm"},
            renderer.render(annotation),
            renderer=renderer,
        )
        self.assertEqual(validation.issues, [])
        self.assertFalse(validation.has_severe_errors())

    def test_detailed_annotation_source_issues_do_not_reject_prompt(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        annotation.task.summary.source = "real"
        annotation.task.robot.source = "real"
        assert annotation.sim_invariants.robot_motion is not None
        annotation.sim_invariants.robot_motion.source = "real"
        assert annotation.target_visuals.workspace is not None
        annotation.target_visuals.workspace.source = "sim"
        renderer = PromptRenderer(PromptAnnotationPipeline()._config.renderer)

        validation = find_local_issues(
            annotation,
            {"task": "place blue mug onto black tray", "robot_type": "dual_arm"},
            renderer.render(annotation),
            renderer=renderer,
        )

        self.assertEqual(validation.issues, [])
        self.assertFalse(validation.has_severe_errors())

    def test_non_retryable_400_is_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _make_dataset(root / "data")
            client = NonRetryableClient()
            pipeline = PromptAnnotationPipeline(
                {
                    "dataset_root": dataset.parent,
                    "output_root": root / "outputs",
                    "media": {"max_frames": 4, "resize_long_edge": 64},
                    "batch": {"concurrency": 1, "api_retry_count": 4},
                },
                client=client,
            )
            result = pipeline.run(dataset_glob="paired_demo")

            self.assertEqual(result["failed"], 1)
            self.assertEqual(client.calls, 1)

    def test_lighting_cannot_be_selected_as_reference_scope(self) -> None:
        with self.assertRaises(ValidationError):
            ReferenceDescription(
                view="camera_head",
                frame_index=0,
                visible_content=["robot"],
                use_for=["lighting"],  # type: ignore[list-item]
                unclear_or_occluded=[],
            )

    def test_renderer_uses_slots_and_ignores_detailed_incidental_objects(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        annotation.task.objects.append(
            TaskObject(
                role="incidental clutter",
                identity="white cup",
                state="resting beside the mug",
                appearance="white paper",
                geometry_affordance=None,
                source="real",
                confidence=0.9,
                evidence=["real:camera_head:frame_000000"],
            )
        )
        prompt = PromptRenderer(PromptAnnotationPipeline()._config.renderer).render(
            annotation
        )
        self.assertNotIn("Real-world video of", prompt)
        self.assertNotIn("Render the scene with", prompt)
        self.assertIn("using its manipulators to place a blue ceramic mug", prompt)
        self.assertNotIn("reference image", prompt)
        self.assertNotIn("white cup", prompt)

    def test_renderer_is_a_single_natural_video_content_sentence(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        annotation.reference.use_for = ["robot", "workspace"]
        annotation.reference.unclear_or_occluded = ["objects", "background"]

        prompt = PromptRenderer(PromptAnnotationPipeline()._config.renderer).render(
            annotation
        )

        self.assertEqual(prompt.count("."), 1)
        self.assertIn("gray workbench", prompt)
        self.assertIn("soft overhead lighting", prompt)

    def test_annotation_requires_at_least_one_reference_candidate(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        payload = annotation.model_dump()
        payload["reference_candidates"] = []
        with self.assertRaises(ValidationError):
            StructuredAnnotation.model_validate(payload)

    def test_task_contract_does_not_use_character_coverage_as_a_gate(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        metadata_task = (
            "align_the_preassembled_banana_bunch_handles_in_the_same_direction"
        )
        annotation.task.semantics = TaskSemantics(
            metadata_task=metadata_task,
            robot="dual-arm robot",
            active_arm="unspecified",
            action=_ground("align", "align"),
            primary_objects=[
                _ground(
                    "preassembled banana bunch handles",
                    "preassembled banana bunches",
                )
            ],
            goal=None,
            constraints=[],
        )
        renderer = PromptRenderer(PromptAnnotationPipeline()._config.renderer)

        validation = find_local_issues(
            annotation,
            {"task": metadata_task, "robot_type": "dual_arm"},
            renderer.render(annotation),
            renderer=renderer,
        )

        self.assertFalse(validation.has_severe_errors())

    def test_task_contract_allows_whole_payload_for_compact_metadata(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        metadata_task = "place blue mug onto black tray"
        annotation.task.semantics.action.metadata_span = metadata_task
        renderer = PromptRenderer(PromptAnnotationPipeline()._config.renderer)

        validation = find_local_issues(
            annotation,
            {"task": metadata_task, "robot_type": "dual_arm"},
            renderer.render(annotation),
            renderer=renderer,
        )

        self.assertFalse(validation.has_severe_errors())

    def test_task_contract_rejects_reference_clutter_in_task_slots(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        metadata_task = "双臂搬运绿色托盘"
        annotation.task.semantics = TaskSemantics(
            metadata_task=metadata_task,
            robot="dual-arm robot",
            active_arm="both",
            action=_ground("carry", "搬运"),
            primary_objects=[
                _ground("a green tray", "绿色托盘"),
                _ground("a white cup", "white cup"),
            ],
            goal=_ground("onto orange mats", "orange mats"),
            constraints=[],
        )
        renderer = PromptRenderer(PromptAnnotationPipeline()._config.renderer)

        validation = find_local_issues(
            annotation,
            {"task": metadata_task, "robot_type": "dual_arm"},
            renderer.render(annotation),
            renderer=renderer,
        )

        errors = [
            issue.reason for issue in validation.issues if issue.severity == "error"
        ]
        self.assertTrue(
            any("unsupported by task metadata" in reason for reason in errors)
        )
        self.assertTrue(
            any("white cup" in reason and "orange mats" in reason for reason in errors)
        )

    def test_renderer_does_not_repeat_goal_as_manipulated_object(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        annotation.task.semantics.primary_objects.append(
            _ground("a black tray", "black tray")
        )
        prompt = PromptRenderer(PromptAnnotationPipeline()._config.renderer).render(
            annotation
        )
        self.assertIn("place a blue ceramic mug onto a black tray", prompt)
        self.assertNotIn("a blue ceramic mug and a black tray", prompt)

    def test_renderer_bounds_appearance_without_changing_annotation(self) -> None:
        annotation = _annotation("sample", "camera_head", 0)
        assert annotation.target_visuals.workspace is not None
        annotation.target_visuals.workspace.text = (
            "White rectangular tabletop with scattered small objects and tools"
        )
        original = annotation.target_visuals.workspace.text
        renderer = PromptRenderer(PromptAnnotationPipeline()._config.renderer)

        prompt = renderer.render(annotation)

        self.assertIn("White rectangular tabletop with scattered small", prompt)
        self.assertNotIn("objects and tools", prompt)
        self.assertEqual(annotation.target_visuals.workspace.text, original)

    def test_final_prompt_hard_limit_excludes_without_a_second_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _make_dataset(root / "data")
            output = root / "outputs"
            client = TrackingClient()
            pipeline = PromptAnnotationPipeline(
                {
                    "dataset_root": dataset.parent,
                    "output_root": output,
                    "media": {"max_frames": 4, "resize_long_edge": 64},
                    "renderer": {
                        "target_prompt_words": 18,
                        "max_prompt_words": 20,
                    },
                    "batch": {"concurrency": 1, "api_retry_count": 0},
                },
                client=client,
            )
            result = pipeline.run(dataset_glob="paired_demo")

            self.assertEqual(result["excluded"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(client.stages, ["annotation"])
            self.assertFalse(
                (output / "annotations/paired_demo__episode_000000.json").exists()
            )
            self.assertFalse(
                (output / "prompts/paired_demo__episode_000000.txt").exists()
            )
            validation = json.loads(
                (output / "validations/paired_demo__episode_000000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                any(issue["severity"] == "error" for issue in validation["issues"])
            )

    def test_public_api_runs_resumes_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _make_dataset(root / "data")
            output = root / "outputs"
            client = TrackingClient()
            pipeline = PromptAnnotationPipeline(
                {
                    "dataset_root": dataset.parent,
                    "output_root": output,
                    "media": {
                        "max_frames": 4,
                        "resize_long_edge": 64,
                        "reference_seed": 7,
                    },
                    "batch": {
                        "concurrency": 1,
                        "api_retry_count": 0,
                    },
                },
                client=client,
            )

            result = pipeline.run(dataset_glob="paired_demo")
            self.assertEqual(result["succeeded"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertTrue(result["complete"])
            self.assertEqual(client.stages, ["annotation"])

            sample_id = "paired_demo__episode_000000"
            prompt_table = dataset / "meta/episodes_prompt.jsonl"
            row = json.loads(prompt_table.read_text(encoding="utf-8"))
            self.assertFalse((dataset / "meta/episodes_prompt.meta.json").exists())
            self.assertIsInstance(row["prompt"], str)
            self.assertNotIn("prompts", row)
            self.assertEqual(len(row["reference_ids"]), 2)
            self.assertNotIn("reference image", row["prompt"])
            self.assertNotIn("Render the scene with", row["prompt"])
            self.assertEqual(row["prompt"].count("."), 1)

            reference_rows = [
                json.loads(line)
                for line in (dataset / "meta/reference_images.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertEqual(len(reference_rows[0]["references"]), 2)
            self.assertEqual(
                row["reference_ids"],
                [item["reference_id"] for item in reference_rows[0]["references"]],
            )

            self.assertEqual(
                reference_rows[0]["references"][0]["source_frame_index"], 0
            )

            resumed_references = pipeline.export_references(dataset_glob="paired_demo")
            self.assertEqual(resumed_references["written"], 0)
            self.assertEqual(resumed_references["skipped"], 2)

            prompt_path = output / "prompts" / f"{sample_id}.txt"
            prompt_path.write_text("stale prompt\n", encoding="utf-8")
            resumed = pipeline.run(dataset_glob="paired_demo")
            self.assertEqual(resumed["skipped"], 1)
            self.assertEqual(client.stages, ["annotation"])
            self.assertTrue(pipeline.audit(dataset_glob="paired_demo")["complete"])

            rendered = pipeline.render(output / "annotations" / f"{sample_id}.json")
            restored = prompt_path.read_text(encoding="utf-8").strip()
            self.assertEqual(restored, rendered)
            self.assertEqual(rendered, row["prompt"])


if __name__ == "__main__":
    unittest.main()
