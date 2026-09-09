"""Run the same inspect -> preprocess -> audit flow as the three CLI commands."""

from pathlib import Path

from sim2real_prompt_annotation import Sim2RealPreprocessingPipeline

DATASET = Path(
    "/media/datasets/EWM_SIM_REAL_PAIRS/model_test/"
    "test_0905_agilex_cobotmagic2_12task_5episode"
)


def main() -> None:
    pipeline = Sim2RealPreprocessingPipeline(
        "config.yaml",
        dataset_root=DATASET,
    )

    # Metadata-only: this does not open videos, initialize YOLOE, or require an API key.
    print(pipeline.inspect(show=5))

    # A normal rerun independently reuses valid Prompt and Reference checkpoints.
    report = pipeline.run(episodes="0")
    print(report)
    if report["status"] != "complete":
        raise RuntimeError("episode 0 preprocessing did not complete")

    audit = pipeline.audit(episodes="0", show=20)
    print(audit)
    if audit["status"] != "complete":
        raise RuntimeError("episode 0 products did not pass audit")


if __name__ == "__main__":
    main()
