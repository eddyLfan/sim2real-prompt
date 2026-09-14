"""Run the same inspect -> preprocess -> audit flow as the three CLI commands."""

from pathlib import Path

from sim2real_prompt_annotation import Sim2RealPreprocessingPipeline

DATASET = Path(
    "/media/datasets/EWM_SIM_REAL_PAIRS/model_train/test/train_00_hang_scissors"
)


def main() -> None:
    pipeline = Sim2RealPreprocessingPipeline(
        "config.yaml",
        dataset_root=DATASET,
    )

    # Metadata-only: no video, API, RobotSeg, Big-LaMa, or YOLOE is initialized.
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
