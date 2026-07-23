from __future__ import annotations

import glob
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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


def resolve_parquet_files(path: str, patterns: str = "*.parquet") -> list[Path]:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        files = [candidate]
    elif candidate.is_dir():
        files = []
        for pattern in patterns.split(","):
            files.extend(candidate.glob(pattern.strip()))
    else:
        files = [Path(match) for match in glob.glob(str(candidate))]
    unique = sorted({item.resolve() for item in files if item.suffix == ".parquet"})
    if not unique:
        raise FileNotFoundError(
            f"no Parquet files found for path={path!r}, patterns={patterns!r}"
        )
    return unique


def _load_validated_tokenizer(
    tokenizer_path: str,
    tokenizer_revision: str,
    vocab_size: int,
    eos_token_id: int,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "online tokenization requires `pip install -e '.[data]'`"
        ) from error
    local_candidate = Path(tokenizer_path).expanduser()
    if local_candidate.is_absolute() and not local_candidate.exists():
        raise FileNotFoundError(f"local tokenizer directory not found: {local_candidate}")
    tokenizer_kwargs: dict[str, object] = {
        "use_fast": True,
        "trust_remote_code": False,
    }
    if tokenizer_revision:
        tokenizer_kwargs["revision"] = tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    if not tokenizer.is_fast:
        raise ValueError("online pretraining requires a fast tokenizer")
    if len(tokenizer) != vocab_size:
        raise ValueError(
            f"tokenizer has {len(tokenizer)} tokens but model vocab_size={vocab_size}"
        )
    if tokenizer.eos_token_id is None or int(tokenizer.eos_token_id) != eos_token_id:
        raise ValueError(
            f"tokenizer EOS id is {tokenizer.eos_token_id}, "
            f"but data.eos_token_id={eos_token_id}"
        )
    return tokenizer


def _encode_documents(
    *,
    tokenizer: Any,
    texts: list[Any],
    eos_token_id: int,
    vocab_size: int,
    validate_token_ids: bool,
) -> list[int]:
    normalized = [text if isinstance(text, str) else str(text or "") for text in texts]
    encoded = tokenizer(
        normalized,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    if len(input_ids) != len(texts):
        raise RuntimeError(
            f"tokenizer returned {len(input_ids)} rows for {len(texts)} documents"
        )
    packed: list[int] = []
    for document_tokens in input_ids:
        tokens = [int(token) for token in document_tokens]
        tokens.append(eos_token_id)
        if validate_token_ids and tokens:
            minimum = min(tokens)
            maximum = max(tokens)
            if minimum < 0 or maximum >= vocab_size:
                raise ValueError(
                    f"online tokenizer produced ids outside [0, {vocab_size}): "
                    f"min={minimum}, max={maximum}"
                )
        packed.extend(tokens)
    return packed


class PackedTextBatcher:
    """Deterministically tokenize and pack rows from a map-style text dataset."""

    def __init__(
        self,
        *,
        dataset: Any,
        tokenizer: Any,
        text_column: str,
        eos_token_id: int,
        vocab_size: int,
        sequence_length: int,
        batch_size: int,
        tokenizer_batch_size: int,
        row_start: int,
        row_stop: int,
        rank: int = 0,
        world_size: int = 1,
        repeat: bool = True,
        validate_token_ids: bool = True,
    ) -> None:
        if not 0 <= row_start < row_stop <= len(dataset):
            raise ValueError(
                f"invalid dataset row range [{row_start}, {row_stop}) "
                f"for {len(dataset)} rows"
            )
        if not 0 <= rank < world_size:
            raise ValueError("rank must fall inside world_size")
        if row_start + rank >= row_stop:
            raise ValueError("dataset row range is too small for the distributed world")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.tokenizer_batch_size = int(tokenizer_batch_size)
        self.row_start = int(row_start)
        self.row_stop = int(row_stop)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.repeat = bool(repeat)
        self.validate_token_ids = bool(validate_token_ids)
        self.row_cursor = self.row_start + self.rank
        self.epoch = 0
        self.token_buffer: list[int] = []
        self.buffer_offset = 0
        self.samples_drawn = 0
        self.documents_consumed = 0

    def _next_row_indices(self) -> list[int]:
        indices: list[int] = []
        while len(indices) < self.tokenizer_batch_size:
            if self.row_cursor >= self.row_stop:
                if not self.repeat:
                    break
                self.epoch += 1
                self.row_cursor = self.row_start + self.rank
            indices.append(self.row_cursor)
            self.row_cursor += self.world_size
        return indices

    def _tokenize_more(self) -> bool:
        indices = self._next_row_indices()
        if not indices:
            return False
        rows = self.dataset[indices]
        if self.text_column not in rows:
            raise KeyError(
                f"text column {self.text_column!r} is absent; "
                f"available columns: {sorted(rows)}"
            )
        texts = rows[self.text_column]
        if isinstance(texts, str):
            texts = [texts]
        self.token_buffer.extend(
            _encode_documents(
                tokenizer=self.tokenizer,
                texts=list(texts),
                eos_token_id=self.eos_token_id,
                vocab_size=self.vocab_size,
                validate_token_ids=self.validate_token_ids,
            )
        )
        self.documents_consumed += len(indices)
        return True

    def _ensure_tokens(self, count: int) -> bool:
        while len(self.token_buffer) - self.buffer_offset < count:
            if not self._tokenize_more():
                return False
        return True

    def next_batch(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Adjacent batches share the final boundary token, so every packed
        # transition is used exactly once.
        token_count = self.batch_size * self.sequence_length
        if not self._ensure_tokens(token_count + 1):
            raise StopIteration
        start = self.buffer_offset
        packed = torch.tensor(
            self.token_buffer[start : start + token_count + 1], dtype=torch.long
        )
        self.buffer_offset += token_count
        if self.buffer_offset >= 1_000_000 or self.buffer_offset * 2 >= len(
            self.token_buffer
        ):
            self.token_buffer = self.token_buffer[self.buffer_offset :]
            self.buffer_offset = 0
        self.samples_drawn += self.batch_size
        return (
            packed[:-1]
            .view(self.batch_size, self.sequence_length)
            .to(device=device, non_blocking=True),
            packed[1:]
            .view(self.batch_size, self.sequence_length)
            .to(device=device, non_blocking=True),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "row_cursor": self.row_cursor,
            "epoch": self.epoch,
            "token_buffer": self.token_buffer[self.buffer_offset :],
            "samples_drawn": self.samples_drawn,
            "documents_consumed": self.documents_consumed,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected_layout = (
            self.rank,
            self.world_size,
            self.row_start,
            self.row_stop,
        )
        observed_layout = (
            int(state.get("rank", self.rank)),
            int(state.get("world_size", self.world_size)),
            int(state.get("row_start", self.row_start)),
            int(state.get("row_stop", self.row_stop)),
        )
        if observed_layout != expected_layout:
            raise ValueError(
                "checkpoint data layout is incompatible with this rank/world/split: "
                f"checkpoint={observed_layout}, current={expected_layout}"
            )
        row_cursor = int(state["row_cursor"])
        if row_cursor < self.row_start + self.rank or row_cursor > (
            self.row_stop + self.world_size - 1
        ):
            raise ValueError("checkpoint row cursor is incompatible with this data split")
        if (row_cursor - self.row_start - self.rank) % self.world_size != 0:
            raise ValueError("checkpoint row cursor is incompatible with this rank")
        self.row_cursor = row_cursor
        self.epoch = int(state.get("epoch", 0))
        stored_buffer = state.get("token_buffer", [])
        if not isinstance(stored_buffer, list):
            raise ValueError("checkpoint token buffer must be a list")
        self.token_buffer = [int(token) for token in stored_buffer]
        self.buffer_offset = 0
        self.samples_drawn = int(state.get("samples_drawn", 0))
        self.documents_consumed = int(state.get("documents_consumed", 0))


@dataclass(frozen=True)
class ParquetFileLayout:
    path: str
    row_count: int
    row_group_rows: tuple[int, ...]


class ParquetRowStream:
    """Bounded-memory sequential text reader over Parquet row groups."""

    def __init__(
        self,
        *,
        layouts: list[ParquetFileLayout],
        text_column: str,
        row_start: int,
        row_stop: int,
        read_batch_size: int,
        repeat: bool,
    ) -> None:
        total_rows = sum(layout.row_count for layout in layouts)
        if not 0 <= row_start < row_stop <= total_rows:
            raise ValueError(
                f"invalid Parquet row range [{row_start}, {row_stop}) "
                f"for {total_rows} rows"
            )
        self.layouts = layouts
        self.text_column = text_column
        self.row_start = int(row_start)
        self.row_stop = int(row_stop)
        self.read_batch_size = int(read_batch_size)
        self.repeat = bool(repeat)
        self.cursor = self.row_start
        self.epoch = 0
        self.file_starts = [0]
        for layout in layouts:
            self.file_starts.append(self.file_starts[-1] + layout.row_count)
        self._parquet_file: Any | None = None
        self._batch_iterator: Any | None = None
        self._pending: list[Any] = []
        self._pending_offset = 0

    def _clear_reader(self) -> None:
        self._batch_iterator = None
        self._parquet_file = None
        self._pending = []
        self._pending_offset = 0

    def _open_at_cursor(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "direct Parquet streaming requires `pip install -e '.[data]'`"
            ) from error
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        file_index = bisect_right(self.file_starts, self.cursor) - 1
        layout = self.layouts[file_index]
        row_in_file = self.cursor - self.file_starts[file_index]
        row_group_starts = [0]
        for count in layout.row_group_rows:
            row_group_starts.append(row_group_starts[-1] + count)
        row_group_index = bisect_right(row_group_starts, row_in_file) - 1
        offset_in_group = row_in_file - row_group_starts[row_group_index]

        self._parquet_file = pq.ParquetFile(layout.path, memory_map=True)
        self._batch_iterator = self._parquet_file.iter_batches(
            batch_size=self.read_batch_size,
            row_groups=[row_group_index],
            columns=[self.text_column],
            use_threads=False,
        )
        skip = offset_in_group
        while True:
            try:
                batch = next(self._batch_iterator)
            except StopIteration as error:
                raise RuntimeError(
                    f"could not seek to row {self.cursor} in {layout.path}"
                ) from error
            values = batch.column(0).to_pylist()
            if skip >= len(values):
                skip -= len(values)
                continue
            self._pending = values[skip:]
            self._pending_offset = 0
            return

    def next_texts(self, count: int) -> list[Any]:
        texts: list[Any] = []
        while len(texts) < count:
            if self.cursor >= self.row_stop:
                if not self.repeat:
                    break
                self.epoch += 1
                self.cursor = self.row_start
                self._clear_reader()
            if self._pending_offset < len(self._pending):
                available = len(self._pending) - self._pending_offset
                take = min(count - len(texts), available, self.row_stop - self.cursor)
                end = self._pending_offset + take
                texts.extend(self._pending[self._pending_offset : end])
                self._pending_offset = end
                self.cursor += take
                continue
            self._pending = []
            self._pending_offset = 0
            if self._batch_iterator is None:
                self._open_at_cursor()
                continue
            try:
                batch = next(self._batch_iterator)
                self._pending = batch.column(0).to_pylist()
            except StopIteration:
                self._clear_reader()
        return texts

    def state_dict(self) -> dict[str, object]:
        return {
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "read_batch_size": self.read_batch_size,
            "cursor": self.cursor,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = (self.row_start, self.row_stop, self.read_batch_size)
        observed = (
            int(state.get("row_start", -1)),
            int(state.get("row_stop", -1)),
            int(state.get("read_batch_size", -1)),
        )
        if observed != expected:
            raise ValueError(
                "checkpoint Parquet stream layout is incompatible: "
                f"checkpoint={observed}, current={expected}"
            )
        cursor = int(state["cursor"])
        if not self.row_start <= cursor <= self.row_stop:
            raise ValueError("checkpoint Parquet cursor is outside this data split")
        self.cursor = cursor
        self.epoch = int(state.get("epoch", 0))
        self._clear_reader()


class StreamingPackedTextBatcher:
    """Online tokenizer/packer backed by a bounded-memory row stream."""

    def __init__(
        self,
        *,
        source: ParquetRowStream,
        tokenizer: Any,
        eos_token_id: int,
        vocab_size: int,
        sequence_length: int,
        batch_size: int,
        tokenizer_batch_size: int,
        validate_token_ids: bool = True,
    ) -> None:
        self.source = source
        self.tokenizer = tokenizer
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.tokenizer_batch_size = int(tokenizer_batch_size)
        self.validate_token_ids = bool(validate_token_ids)
        self.token_buffer: list[int] = []
        self.buffer_offset = 0
        self.samples_drawn = 0
        self.documents_consumed = 0

    def _tokenize_more(self) -> bool:
        texts = self.source.next_texts(self.tokenizer_batch_size)
        if not texts:
            return False
        self.token_buffer.extend(
            _encode_documents(
                tokenizer=self.tokenizer,
                texts=texts,
                eos_token_id=self.eos_token_id,
                vocab_size=self.vocab_size,
                validate_token_ids=self.validate_token_ids,
            )
        )
        self.documents_consumed += len(texts)
        return True

    def _ensure_tokens(self, count: int) -> bool:
        while len(self.token_buffer) - self.buffer_offset < count:
            if not self._tokenize_more():
                return False
        return True

    def next_batch(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = self.batch_size * self.sequence_length
        if not self._ensure_tokens(token_count + 1):
            raise StopIteration
        start = self.buffer_offset
        packed = torch.tensor(
            self.token_buffer[start : start + token_count + 1], dtype=torch.long
        )
        self.buffer_offset += token_count
        if self.buffer_offset >= 1_000_000 or self.buffer_offset * 2 >= len(
            self.token_buffer
        ):
            self.token_buffer = self.token_buffer[self.buffer_offset :]
            self.buffer_offset = 0
        self.samples_drawn += self.batch_size
        return (
            packed[:-1]
            .view(self.batch_size, self.sequence_length)
            .to(device=device, non_blocking=True),
            packed[1:]
            .view(self.batch_size, self.sequence_length)
            .to(device=device, non_blocking=True),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "source": self.source.state_dict(),
            "token_buffer": self.token_buffer[self.buffer_offset :],
            "samples_drawn": self.samples_drawn,
            "documents_consumed": self.documents_consumed,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        source_state = state.get("source")
        if not isinstance(source_state, dict):
            raise ValueError("checkpoint is missing its Parquet source state")
        self.source.load_state_dict(source_state)
        stored_buffer = state.get("token_buffer", [])
        if not isinstance(stored_buffer, list):
            raise ValueError("checkpoint token buffer must be a list")
        self.token_buffer = [int(token) for token in stored_buffer]
        self.buffer_offset = 0
        self.samples_drawn = int(state.get("samples_drawn", 0))
        self.documents_consumed = int(state.get("documents_consumed", 0))


def _partition_rows(
    row_start: int, row_stop: int, rank: int, world_size: int
) -> tuple[int, int]:
    if not 0 <= rank < world_size:
        raise ValueError("rank must fall inside world_size")
    total = row_stop - row_start
    quotient, remainder = divmod(total, world_size)
    local_start = row_start + rank * quotient + min(rank, remainder)
    local_stop = local_start + quotient + (1 if rank < remainder else 0)
    if local_start >= local_stop:
        raise ValueError("data split contains fewer rows than distributed ranks")
    return local_start, local_stop


class DirectParquetTextCorpus:
    """Metadata-only Parquet corpus; no Arrow dataset is materialized."""

    def __init__(
        self,
        *,
        layouts: list[ParquetFileLayout],
        tokenizer: Any,
        metadata: dict[str, object],
        text_column: str,
        eos_token_id: int,
        vocab_size: int,
        tokenizer_batch_size: int,
        parquet_read_batch_size: int,
        validation_rows: int,
        validate_token_ids: bool,
    ) -> None:
        self.layouts = layouts
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.text_column = text_column
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.tokenizer_batch_size = tokenizer_batch_size
        self.parquet_read_batch_size = parquet_read_batch_size
        self.validation_rows = validation_rows
        self.validate_token_ids = validate_token_ids
        self.total_rows = sum(layout.row_count for layout in layouts)
        if validation_rows >= self.total_rows:
            raise ValueError(
                f"validation_rows={validation_rows} leaves no training rows "
                f"from {self.total_rows} rows"
            )

    @property
    def train_rows(self) -> int:
        return self.total_rows - self.validation_rows

    def _batcher(
        self,
        *,
        row_start: int,
        row_stop: int,
        sequence_length: int,
        batch_size: int,
        rank: int,
        world_size: int,
        repeat: bool,
    ) -> StreamingPackedTextBatcher:
        local_start, local_stop = _partition_rows(
            row_start, row_stop, rank, world_size
        )
        source = ParquetRowStream(
            layouts=self.layouts,
            text_column=self.text_column,
            row_start=local_start,
            row_stop=local_stop,
            read_batch_size=self.parquet_read_batch_size,
            repeat=repeat,
        )
        return StreamingPackedTextBatcher(
            source=source,
            tokenizer=self.tokenizer,
            eos_token_id=self.eos_token_id,
            vocab_size=self.vocab_size,
            sequence_length=sequence_length,
            batch_size=batch_size,
            tokenizer_batch_size=self.tokenizer_batch_size,
            validate_token_ids=self.validate_token_ids,
        )

    def train_batcher(
        self,
        *,
        sequence_length: int,
        batch_size: int,
        rank: int,
        world_size: int,
    ) -> StreamingPackedTextBatcher:
        return self._batcher(
            row_start=0,
            row_stop=self.train_rows,
            sequence_length=sequence_length,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            repeat=True,
        )

    def validation_batcher(
        self,
        *,
        sequence_length: int,
        batch_size: int,
        rank: int,
        world_size: int,
    ) -> StreamingPackedTextBatcher:
        return self._batcher(
            row_start=self.train_rows,
            row_stop=self.total_rows,
            sequence_length=sequence_length,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            repeat=False,
        )


class ParquetTextCorpus:
    """Raw-text Arrow cache plus tokenizer contract for online packing."""

    def __init__(
        self,
        *,
        dataset: Any,
        tokenizer: Any,
        metadata: dict[str, object],
        text_column: str,
        eos_token_id: int,
        vocab_size: int,
        tokenizer_batch_size: int,
        validation_rows: int,
        validate_token_ids: bool,
    ) -> None:
        if validation_rows >= len(dataset):
            raise ValueError(
                f"validation_rows={validation_rows} leaves no training rows "
                f"from a dataset with {len(dataset)} rows"
            )
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.text_column = text_column
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.tokenizer_batch_size = tokenizer_batch_size
        self.validation_rows = validation_rows
        self.validate_token_ids = validate_token_ids

    @property
    def train_rows(self) -> int:
        return len(self.dataset) - self.validation_rows

    def train_batcher(
        self,
        *,
        sequence_length: int,
        batch_size: int,
        rank: int,
        world_size: int,
    ) -> PackedTextBatcher:
        return PackedTextBatcher(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            text_column=self.text_column,
            eos_token_id=self.eos_token_id,
            vocab_size=self.vocab_size,
            sequence_length=sequence_length,
            batch_size=batch_size,
            tokenizer_batch_size=self.tokenizer_batch_size,
            row_start=0,
            row_stop=self.train_rows,
            rank=rank,
            world_size=world_size,
            repeat=True,
            validate_token_ids=self.validate_token_ids,
        )

    def validation_batcher(
        self,
        *,
        sequence_length: int,
        batch_size: int,
        rank: int,
        world_size: int,
    ) -> PackedTextBatcher:
        return PackedTextBatcher(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            text_column=self.text_column,
            eos_token_id=self.eos_token_id,
            vocab_size=self.vocab_size,
            sequence_length=sequence_length,
            batch_size=batch_size,
            tokenizer_batch_size=self.tokenizer_batch_size,
            row_start=self.train_rows,
            row_stop=len(self.dataset),
            rank=rank,
            world_size=world_size,
            repeat=False,
            validate_token_ids=self.validate_token_ids,
        )


def _load_direct_parquet_text_corpus(
    *,
    parquet_files: list[Path],
    tokenizer: Any,
    text_column: str,
    tokenizer_path: str,
    tokenizer_revision: str,
    tokenizer_batch_size: int,
    parquet_read_batch_size: int,
    validation_rows: int,
    vocab_size: int,
    eos_token_id: int,
    validate_token_ids: bool,
) -> DirectParquetTextCorpus:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "direct Parquet streaming requires `pip install -e '.[data]'`"
        ) from error
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    layouts: list[ParquetFileLayout] = []
    for path in parquet_files:
        parquet_file = pq.ParquetFile(path, memory_map=True)
        if text_column not in parquet_file.schema_arrow.names:
            raise KeyError(
                f"text column {text_column!r} is absent from {path}; "
                f"available columns: {parquet_file.schema_arrow.names}"
            )
        metadata = parquet_file.metadata
        layouts.append(
            ParquetFileLayout(
                path=str(path),
                row_count=metadata.num_rows,
                row_group_rows=tuple(
                    metadata.row_group(index).num_rows
                    for index in range(metadata.num_row_groups)
                ),
            )
        )
    total_rows = sum(layout.row_count for layout in layouts)
    if validation_rows >= total_rows:
        raise ValueError(
            f"validation_rows={validation_rows} leaves no training rows "
            f"from {total_rows} rows"
        )
    corpus_metadata: dict[str, object] = {
        "input_format": "parquet_text",
        "parquet_backend": "direct",
        "materializes_arrow_cache": False,
        "parquet_files": [
            {
                "path": layout.path,
                "rows": layout.row_count,
                "row_groups": len(layout.row_group_rows),
            }
            for layout in layouts
        ],
        "total_rows": total_rows,
        "train_rows": total_rows - validation_rows,
        "validation_rows": validation_rows,
        "text_column": text_column,
        "tokenizer_batch_size": tokenizer_batch_size,
        "parquet_read_batch_size": parquet_read_batch_size,
        "tokenizer": {
            "source": tokenizer_path,
            "revision": tokenizer_revision,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "is_fast": tokenizer.is_fast,
        },
        "packing": {
            "add_special_tokens": False,
            "append_eos_per_document": True,
            "shuffle": False,
        },
    }
    return DirectParquetTextCorpus(
        layouts=layouts,
        tokenizer=tokenizer,
        metadata=corpus_metadata,
        text_column=text_column,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        tokenizer_batch_size=tokenizer_batch_size,
        parquet_read_batch_size=parquet_read_batch_size,
        validation_rows=validation_rows,
        validate_token_ids=validate_token_ids,
    )


def load_parquet_text_corpus(
    *,
    path: str,
    patterns: str,
    tokenizer_path: str,
    tokenizer_revision: str,
    text_column: str,
    cache_dir: str,
    dataset_num_proc: int,
    tokenizer_batch_size: int,
    parquet_backend: str,
    parquet_read_batch_size: int,
    validation_rows: int,
    vocab_size: int,
    eos_token_id: int,
    validate_token_ids: bool,
) -> ParquetTextCorpus | DirectParquetTextCorpus:
    """Open raw Parquet directly or reuse an explicitly requested HF cache."""
    parquet_files = resolve_parquet_files(path, patterns)
    tokenizer = _load_validated_tokenizer(
        tokenizer_path,
        tokenizer_revision,
        vocab_size,
        eos_token_id,
    )
    if parquet_backend == "direct":
        return _load_direct_parquet_text_corpus(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            text_column=text_column,
            tokenizer_path=tokenizer_path,
            tokenizer_revision=tokenizer_revision,
            tokenizer_batch_size=tokenizer_batch_size,
            parquet_read_batch_size=parquet_read_batch_size,
            validation_rows=validation_rows,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            validate_token_ids=validate_token_ids,
        )
    if parquet_backend != "hf_cache":
        raise ValueError("parquet_backend must be direct or hf_cache")
    try:
        from datasets import config as datasets_config
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "hf_cache Parquet backend requires `pip install -e '.[data]'`"
        ) from error
    effective_cache = (
        str(Path(cache_dir).expanduser())
        if cache_dir
        else os.environ.get(
            "HF_DATASETS_CACHE", str(datasets_config.HF_DATASETS_CACHE)
        )
    )

    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(item) for item in parquet_files]},
        split="train",
        num_proc=dataset_num_proc,
        cache_dir=effective_cache,
    )
    if text_column not in dataset.column_names:
        raise KeyError(
            f"text column {text_column!r} is absent; "
            f"available columns: {dataset.column_names}"
        )
    source_fingerprint = getattr(dataset, "_fingerprint", "")
    source_cache_files = list(getattr(dataset, "cache_files", []))
    # Keep the existing Arrow buffers but avoid materializing unused FineWeb
    # metadata columns for every tokenizer batch.
    dataset = dataset.select_columns([text_column])

    cache_files = [
        str(Path(item["filename"]).resolve())
        for item in source_cache_files
        if item.get("filename")
    ]
    metadata: dict[str, object] = {
        "input_format": "parquet_text",
        "parquet_backend": "hf_cache",
        "materializes_arrow_cache": True,
        "parquet_files": [str(item) for item in parquet_files],
        "hf_cache_dir": effective_cache,
        "hf_cache_files": cache_files,
        "dataset_fingerprint": source_fingerprint,
        "total_rows": len(dataset),
        "train_rows": len(dataset) - validation_rows,
        "validation_rows": validation_rows,
        "text_column": text_column,
        "tokenizer": {
            "source": tokenizer_path,
            "revision": tokenizer_revision,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "is_fast": tokenizer.is_fast,
        },
        "packing": {
            "add_special_tokens": False,
            "append_eos_per_document": True,
            "shuffle": False,
        },
    }
    return ParquetTextCorpus(
        dataset=dataset,
        tokenizer=tokenizer,
        metadata=metadata,
        text_column=text_column,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        tokenizer_batch_size=tokenizer_batch_size,
        validation_rows=validation_rows,
        validate_token_ids=validate_token_ids,
    )


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
