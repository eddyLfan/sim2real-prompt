from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sim2real_prompt_annotation as package
from sim2real_prompt_annotation.config import load_config


class StandalonePackageTest(unittest.TestCase):
    def test_uses_standard_src_layout(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source_root = project_root / "src/sim2real_prompt_annotation"
        self.assertTrue((source_root / "api.py").is_file())
        self.assertTrue((source_root / "prompts/annotation_system.txt").is_file())
        self.assertFalse((project_root / "api.py").exists())

    def test_only_facade_is_public(self) -> None:
        self.assertEqual(package.__all__, ["PromptAnnotationPipeline"])
        self.assertTrue(callable(package.PromptAnnotationPipeline))

    def test_has_no_omini_s2r_compatibility_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source_root = project_root / "src/sim2real_prompt_annotation"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(source_root.glob("*.py"))
        )
        self.assertNotIn("S2R_", source)
        self.assertNotIn("/media/datasets/", source)
        self.assertNotIn("/media/unify/", source)

        config = package.PromptAnnotationPipeline()._config
        self.assertEqual(config.discovery.min_episode_frames, 1)
        self.assertEqual(config.discovery.required_views, [])
        self.assertFalse(hasattr(config.discovery, "require_later_window"))
        self.assertFalse(hasattr(config.discovery, "window_stride"))

    def test_config_paths_are_relative_to_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset_root: ./paired_data",
                        "output_root: ./outputs",
                        "annotation:",
                        "  system_prompt: ./annotation.txt",
                        "critic:",
                        "  system_prompt: ./critic.txt",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config.dataset_root, root / "paired_data")
            self.assertEqual(config.output_root, root / "outputs")
            self.assertEqual(config.annotation.system_prompt, root / "annotation.txt")
            self.assertEqual(config.critic.system_prompt, root / "critic.txt")

    def test_facade_inspects_paired_lerobot_without_repository_imports(self) -> None:
        from sim2real_prompt_annotation import PromptAnnotationPipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "paired_task"
            (dataset / "meta").mkdir(parents=True)
            video_template = (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            )
            (dataset / "meta/info.json").write_text(
                json.dumps(
                    {
                        "robot_type": "test_robot",
                        "fps": 30,
                        "chunks_size": 1000,
                        "video_path": video_template,
                        "features": {
                            "observation.images.camera_head": {"dtype": "video"},
                            "observation.images.camera_head_sim": {"dtype": "video"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (dataset / "meta/episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "length": 8, "tasks": ["move block"]})
                + "\n",
                encoding="utf-8",
            )
            for video_key in (
                "observation.images.camera_head",
                "observation.images.camera_head_sim",
            ):
                video = dataset / video_template.format(
                    episode_chunk=0, video_key=video_key, episode_index=0
                )
                video.parent.mkdir(parents=True, exist_ok=True)
                video.touch()

            pipeline = PromptAnnotationPipeline(
                {
                    "dataset_root": root,
                    "output_root": root / "outputs",
                    "discovery": {
                        "min_episode_frames": 1,
                        "required_views": ["camera_head"],
                    },
                }
            )
            summary = pipeline.inspect(dataset_glob="paired_*")
            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(summary["first_samples"][0]["episode"], 0)

        cli_source = (Path(package.__file__).parent / "cli.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("from S2R.", cli_source)


if __name__ == "__main__":
    unittest.main()
