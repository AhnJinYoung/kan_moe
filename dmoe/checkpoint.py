from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig, ModelConfig
from .model import DecoderLM


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    candidates = list(Path(output_dir).glob("step_*.pt"))
    if not candidates:
        return None

    def step_number(path: Path) -> int:
        match = re.search(r"step_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step_number)


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    tokens_seen: int,
    model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    config: ExperimentConfig,
    sampler_states: list[dict[str, Any]],
    rng_states: list[dict[str, Any]],
    keep_last: int,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"step_{step:08d}.pt"
    temporary = directory / f".{target.name}.tmp"
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "sampler_states": sampler_states,
        "rng_states": rng_states,
        "torch_version": torch.__version__,
    }
    torch.save(payload, temporary)
    os.replace(temporary, target)

    checkpoints = sorted(directory.glob("step_*.pt"))
    if keep_last > 0:
        for old_checkpoint in checkpoints[:-keep_last]:
            old_checkpoint.unlink()
    return target


def load_training_checkpoint(
    path: str | Path,
    model: DecoderLM,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def load_model_from_checkpoint(
    path: str | Path,
    device: torch.device | str,
    top_k: int | None = None,
    strict: bool = True,
) -> tuple[DecoderLM, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True)
    config_raw = checkpoint.get("config", {})
    model_raw = config_raw.get("model", checkpoint.get("model_config"))
    if model_raw is None:
        raise ValueError("checkpoint does not contain a model configuration")
    model_config = ModelConfig(**model_raw)
    if top_k is not None:
        model_config.top_k = top_k
    model_config.gradient_checkpointing = False
    model_config.validate()
    model = DecoderLM(model_config)
    model.load_state_dict(checkpoint["model"], strict=strict)
    model.to(device)
    model.eval()
    # Evaluation must not retain several gigabytes of optimizer and model-state
    # references after the weights have been copied into the module.
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key
        not in {
            "model",
            "optimizer",
            "scaler",
            "sampler_states",
            "rng_states",
        }
    }
    del checkpoint
    return model, metadata
