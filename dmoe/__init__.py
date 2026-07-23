"""Distributional mixture-of-experts research package."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
    from .model import DecoderLM

__all__ = [
    "DataConfig",
    "DecoderLM",
    "ExperimentConfig",
    "ModelConfig",
    "TrainConfig",
]


def __getattr__(name: str) -> Any:
    if name == "DecoderLM":
        return importlib.import_module(".model", __name__).DecoderLM
    if name in {"DataConfig", "ExperimentConfig", "ModelConfig", "TrainConfig"}:
        config = importlib.import_module(".config", __name__)
        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
