"""Minimal use of the package's single public Python interface."""

from sim2real_prompt_annotation import PromptAnnotationPipeline

pipeline = PromptAnnotationPipeline("config.yaml")

# Metadata-only discovery; no API key is needed.
print(pipeline.inspect(dataset_glob="paired_task_*", limit=3))

# This also reports the reproducibly random Reference frame without API access.
print(
    pipeline.run(
        dataset_glob="paired_task_*",
        limit=1,
        dry_run=True,
        prepare_media=True,
    )
)
