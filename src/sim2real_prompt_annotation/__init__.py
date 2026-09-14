"""Real-video Prompt and robot-removed scene Reference preprocessing."""

from .api import PromptAnnotationPipeline, Sim2RealPreprocessingPipeline
from .config import PipelineConfig, load_config

__all__ = [
    "PipelineConfig",
    "PromptAnnotationPipeline",
    "Sim2RealPreprocessingPipeline",
    "load_config",
]
