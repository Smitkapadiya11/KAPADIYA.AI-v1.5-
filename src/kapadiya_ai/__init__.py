"""KAPADIYA.AI v1.5 TMKB - fast, low-VRAM virtual try-on (based on FASHN VTON v1.5)."""

__version__ = "1.5.0"
MODEL_NAME = "KAPADIYA.AI v1.5 TMKB"

from .pipeline import PRESETS, PipelineOutput, TryOnPipeline
from .tryon_mmdit import TryOnModel

__all__ = [
    "TryOnPipeline",
    "PipelineOutput",
    "TryOnModel",
    "PRESETS",
    "MODEL_NAME",
    "__version__",
]
