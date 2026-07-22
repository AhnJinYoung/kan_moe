from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


SUPPORTED_SUFFIXES = {".bin", ".npy"}


def validate_data_manifest(
    manifest_path: str,
    *,
    vocab_size: int,
    eos_token_id: int,
    binary_dtype: str,
) -> dict[str, object]:
    """Fail early if tokenized data and the model/tokenizer contract disagree."""
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"token data manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError(f"missing tokenizer metadata in {path}")
    observed_vocab = int(tokenizer.get("vocab_size", -1))
    observed_eos = int(tokenizer.get("eos_token_id", -1))
    observed_dtype = str(manifest.get("dtype", ""))
    mismatches: list[str] = []
    if observed_vocab != vocab_size:
        mismatches.append(f"vocab_size manifest={observed_vocab} config={vocab_size}")
    if observed_eos != eos_token_id:
        mismatches.append(f"eos_token_id manifest={observed_eos} config={eos_token_id}")
    if observed_dtype != binary_dtype:
        mismatches.append(f"dtype manifest={observed_dtype} config={binary_dtype}")
    if mismatches:
        raise ValueError(f"token data contract mismatch in {path}: " + "; ".join(mismatches))
    return manifest


def resolve_token_shards(path: str, patterns: str = "*.bin,*.npy") -> list[Path]:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        shards = [candidate]
    elif candidate.is_dir():
        shards = []
        for pattern in patterns.split(","):
            shards.extend(candidate.glob(pattern.strip()))
    else:
        shards = [Path(match) for match in glob.glob(str(candidate))]
    unique = sorted({item.resolve() for item in shards if item.suffix in SUPPORTED_SUFFIXES})
    if not unique:
        raise FileNotFoundError(
            f"no token shards found for path={path!r}, patterns={patterns!r}"
        )
    return unique


def open_token_shard(path: Path, binary_dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
    elif path.suffix == ".bin":
        array = np.memmap(path, mode="r", dtype=np.dtype(binary_dtype))
    else:
        raise ValueError(f"unsupported token shard suffix: {path.suffix}")
    if array.ndim != 1:
        raise ValueError(f"token shard must be one-dimensional: {path}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"token shard must contain integer ids: {path}")
    return array


class RandomTokenBatcher:
    """Random contiguous-window sampler over memory-mapped token shards."""

    def __init__(
        self,
        path: str,
        patterns: str,
        binary_dtype: str,
        sequence_length: int,
        batch_size: int,
        seed: int,
        rank: int = 0,
        vocab_size: int | None = None,
        validate_token_ids: bool = True,
    ) -> None:
        self.paths = resolve_token_shards(path, patterns)
        self.arrays = [open_token_shard(item, binary_dtype) for item in self.paths]
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.generator = np.random.default_rng(seed + rank * 1_000_003)
        self.valid_starts = np.asarray(
            [max(0, len(array) - sequence_length) for array in self.arrays],
            dtype=np.int64,
        )
        if self.valid_starts.sum() == 0:
            raise ValueError("all token shards are shorter than sequence_length + 1")
        self.shard_probability = self.valid_starts / self.valid_starts.sum()
        self.samples_drawn = 0
        if validate_token_ids and vocab_size is not None:
            self.validate(vocab_size)

    def validate(self, vocab_size: int, sample_count: int = 4_096) -> None:
        for path, array in zip(self.paths, self.arrays):
            if len(array) == 0:
                raise ValueError(f"empty token shard: {path}")
            count = min(sample_count, len(array))
            indices = np.linspace(0, len(array) - 1, count, dtype=np.int64)
            values = np.asarray(array[indices], dtype=np.int64)
            minimum = int(values.min())
            maximum = int(values.max())
            if minimum < 0 or maximum >= vocab_size:
                raise ValueError(
                    f"token ids in {path} fall outside [0, {vocab_size}): "
                    f"sampled min={minimum}, max={maximum}"
                )

    def next_batch(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        shard_ids = self.generator.choice(
            len(self.arrays), size=self.batch_size, p=self.shard_probability
        )
        windows: list[np.ndarray] = []
        for shard_id in shard_ids:
            maximum_start = int(self.valid_starts[shard_id])
            start = int(self.generator.integers(0, maximum_start))
            window = np.asarray(
                self.arrays[shard_id][start : start + self.sequence_length + 1],
                dtype=np.int64,
            )
            windows.append(window)
        batch = torch.from_numpy(np.stack(windows, axis=0))
        self.samples_drawn += self.batch_size
        return (
            batch[:, :-1].to(device=device, non_blocking=True),
            batch[:, 1:].to(device=device, non_blocking=True),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "generator_state": self.generator.bit_generator.state,
            "samples_drawn": self.samples_drawn,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.generator.bit_generator.state = state["generator_state"]  # type: ignore[assignment]
        self.samples_drawn = int(state.get("samples_drawn", 0))


def sequential_token_batches(
    path: str,
    patterns: str,
    binary_dtype: str,
    sequence_length: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    max_batches: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield disjoint validation windows, sharded by distributed rank."""
    paths = resolve_token_shards(path, patterns)
    pending: list[np.ndarray] = []
    global_window_index = 0
    yielded_batches = 0
    for shard_path in paths:
        array = open_token_shard(shard_path, binary_dtype)
        for start in range(0, len(array) - sequence_length, sequence_length):
            take = global_window_index % world_size == rank
            global_window_index += 1
            if not take:
                continue
            window = np.asarray(
                array[start : start + sequence_length + 1], dtype=np.int64
            )
            pending.append(window)
            if len(pending) < batch_size:
                continue
            batch = torch.from_numpy(np.stack(pending, axis=0))
            yield batch[:, :-1], batch[:, 1:]
            pending.clear()
            yielded_batches += 1
            if max_batches > 0 and yielded_batches >= max_batches:
                return
    if pending and (max_batches <= 0 or yielded_batches < max_batches):
        batch = torch.from_numpy(np.stack(pending, axis=0))
        yield batch[:, :-1], batch[:, 1:]
