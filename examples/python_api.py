"""Minimal use of the package's single public Python interface."""

from sim2real_prompt_annotation import PromptAnnotationPipeline

pipeline = PromptAnnotationPipeline("config.yaml")

# Metadata-only discovery; no API key is needed.
print(pipeline.inspect(dataset_glob="paired_task_*", limit=3))

# Remove dry_run after inspecting the selection and exporting the API variables.
print(pipeline.run(dataset_glob="paired_task_*", limit=1, dry_run=True))
