from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TOKENIZER = "mistralai/Mistral-7B-v0.3"
# Pin the exact public snapshot used by the experiment. Passing another revision is
# supported, but it will be recorded in the output manifest.
DEFAULT_TOKENIZER_REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"
DEFAULT_EXPECTED_VOCAB_SIZE = 32_768


@dataclass(frozen=True)
class ShardJob:
    source: str
    destination: str
    metadata_path: str
    split: str


@dataclass(frozen=True)
class ShardResult:
    source: str
    source_bytes: int
    source_mtime_ns: int
    destination: str
    split: str
    documents: int
    null_documents: int
    tokens: int
    bytes: int
    dtype: str


_WORKER_TOKENIZER: Any = None
_WORKER_TEXT_COLUMN = "text"
_WORKER_BATCH_SIZE = 256
_WORKER_DTYPE = "uint16"
_WORKER_EOS_ID = 2
_WORKER_OVERWRITE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream FineWeb-Edu parquet documents into memory-mappable token shards"
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--input-glob", default="*.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument(
        "--expected-vocab-size", type=int, default=DEFAULT_EXPECTED_VOCAB_SIZE
    )
    parser.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(16, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--validation-files",
        type=int,
        default=1,
        help="Reserve the last N lexicographically sorted parquet files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace already completed output shards.",
    )
    return parser.parse_args()


def split_source_files(
    input_dir: str | Path, pattern: str, validation_files: int
) -> tuple[list[Path], list[Path]]:
    files = sorted(Path(input_dir).expanduser().glob(pattern))
    if len(files) < 2:
        raise ValueError(f"expected at least two input files, found {len(files)}")
    if validation_files < 1 or validation_files >= len(files):
        raise ValueError(
            "validation-files must be at least 1 and smaller than the input file count"
        )
    return files[:-validation_files], files[-validation_files:]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def enforce_preparation_spec(output_dir: Path, spec: dict[str, Any]) -> None:
    """Prevent completed shards from being relabeled with another tokenizer/split."""
    path = output_dir / "preprocessing_spec.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != spec:
            raise ValueError(
                f"preprocessing specification differs from {path}; use a new empty "
                "output directory instead of mixing token shards"
            )
        return
    if any(output_dir.glob("train/*.bin")) or any(
        output_dir.glob("validation/*.bin")
    ):
        raise ValueError(
            f"{output_dir} contains token shards without preprocessing_spec.json; "
            "use a new empty output directory"
        )
    _write_json_atomic(path, spec)


def _load_completed_result(job: ShardJob, dtype: str) -> ShardResult | None:
    destination = Path(job.destination)
    metadata_path = Path(job.metadata_path)
    if not destination.is_file() or not metadata_path.is_file():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        result = ShardResult(**raw)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    source = Path(job.source)
    if not source.is_file():
        return None
    source_stat = source.stat()
    expected_bytes = result.tokens * np.dtype(dtype).itemsize
    if (
        result.source != job.source
        or result.source_bytes != source_stat.st_size
        or result.source_mtime_ns != source_stat.st_mtime_ns
        or result.destination != job.destination
        or result.split != job.split
        or result.dtype != dtype
        or result.bytes != expected_bytes
        or destination.stat().st_size != expected_bytes
    ):
        return None
    return result


def _initialize_worker(
    tokenizer_path: str,
    text_column: str,
    batch_size: int,
    dtype: str,
    eos_id: int,
    overwrite: bool,
) -> None:
    global _WORKER_TOKENIZER
    global _WORKER_TEXT_COLUMN, _WORKER_BATCH_SIZE, _WORKER_DTYPE
    global _WORKER_EOS_ID, _WORKER_OVERWRITE

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required; install the project data extra"
        ) from error
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    _WORKER_TEXT_COLUMN = text_column
    _WORKER_BATCH_SIZE = batch_size
    _WORKER_DTYPE = dtype
    _WORKER_EOS_ID = eos_id
    _WORKER_OVERWRITE = overwrite


def _tokenize_shard(job: ShardJob) -> ShardResult:
    if not _WORKER_OVERWRITE:
        completed = _load_completed_result(job, _WORKER_DTYPE)
        if completed is not None:
            return completed

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required; install the project data extra"
        ) from error

    destination = Path(job.destination)
    metadata_path = Path(job.metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    dtype = np.dtype(_WORKER_DTYPE)
    maximum_token_id = np.iinfo(dtype).max
    document_count = 0
    null_document_count = 0
    token_count = 0

    parquet_file = pq.ParquetFile(job.source)
    if _WORKER_TEXT_COLUMN not in parquet_file.schema.names:
        raise ValueError(
            f"column {_WORKER_TEXT_COLUMN!r} is missing from {job.source}; "
            f"available columns: {parquet_file.schema.names}"
        )

    try:
        with temporary.open("wb") as output:
            for record_batch in parquet_file.iter_batches(
                batch_size=_WORKER_BATCH_SIZE, columns=[_WORKER_TEXT_COLUMN]
            ):
                raw_texts = record_batch.column(0).to_pylist()
                texts: list[str] = []
                for text in raw_texts:
                    if text is None:
                        null_document_count += 1
                        continue
                    if not isinstance(text, str):
                        raise TypeError(
                            f"non-string value in {_WORKER_TEXT_COLUMN!r} of {job.source}"
                        )
                    texts.append(text)
                if not texts:
                    continue
                encoded = _WORKER_TOKENIZER(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    truncation=False,
                )["input_ids"]
                pieces: list[np.ndarray] = []
                for token_ids in encoded:
                    array = np.asarray(token_ids, dtype=np.int64)
                    if array.size and (
                        int(array.min()) < 0 or int(array.max()) > maximum_token_id
                    ):
                        raise ValueError(
                            f"token id does not fit {_WORKER_DTYPE} in {job.source}"
                        )
                    pieces.append(array.astype(dtype, copy=False))
                    pieces.append(np.asarray([_WORKER_EOS_ID], dtype=dtype))
                    token_count += int(array.size) + 1
                np.concatenate(pieces).tofile(output)
                document_count += len(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    result = ShardResult(
        source=job.source,
        source_bytes=Path(job.source).stat().st_size,
        source_mtime_ns=Path(job.source).stat().st_mtime_ns,
        destination=job.destination,
        split=job.split,
        documents=document_count,
        null_documents=null_document_count,
        tokens=token_count,
        bytes=destination.stat().st_size,
        dtype=_WORKER_DTYPE,
    )
    _write_json_atomic(metadata_path, asdict(result))
    return result


def _build_jobs(
    train_files: list[Path], validation_files: list[Path], output_dir: Path
) -> list[ShardJob]:
    jobs: list[ShardJob] = []
    for split, files in (("train", train_files), ("validation", validation_files)):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for source in files:
            destination = split_dir / f"{source.stem}.bin"
            jobs.append(
                ShardJob(
                    source=str(source.resolve()),
                    destination=str(destination.resolve()),
                    metadata_path=str(destination.with_suffix(".bin.json").resolve()),
                    split=split,
                )
            )
    return jobs


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("batch-size and workers must be positive")
    if re.fullmatch(r"[0-9a-f]{40}", args.tokenizer_revision) is None:
        raise ValueError("tokenizer-revision must be an exact 40-character commit SHA")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_files, validation_files = split_source_files(
        args.input_dir, args.input_glob, args.validation_files
    )

    try:
        import pyarrow
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "preprocessing dependencies are missing; run "
            "`pip install -e '.[data]'`"
        ) from error

    print(
        f"loading tokenizer {args.tokenizer}@{args.tokenizer_revision}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        use_fast=True,
        trust_remote_code=False,
    )
    vocab_size = len(tokenizer)
    if not tokenizer.is_fast:
        raise ValueError("the preprocessing pipeline requires a fast tokenizer")
    if args.expected_vocab_size > 0 and vocab_size != args.expected_vocab_size:
        raise ValueError(
            f"tokenizer vocabulary is {vocab_size}, expected {args.expected_vocab_size}"
        )
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define an EOS token")
    dtype = np.dtype(args.dtype)
    if vocab_size - 1 > np.iinfo(dtype).max:
        raise ValueError(f"vocabulary size {vocab_size} does not fit {args.dtype}")

    source_records = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "split": "validation" if path in validation_files else "train",
        }
        for path in train_files + validation_files
    ]
    preparation_spec = {
        "format_version": 1,
        "input_dir": str(Path(args.input_dir).expanduser().resolve()),
        "input_glob": args.input_glob,
        "text_column": args.text_column,
        "sources": source_records,
        "tokenizer_source": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "numpy_version": np.__version__,
        "pyarrow_version": pyarrow.__version__,
        "tokenizers_version": tokenizers.__version__,
        "transformers_version": transformers.__version__,
        "vocab_size": vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "add_special_tokens": False,
        "document_separator": "eos",
        "dtype": args.dtype,
    }
    enforce_preparation_spec(output_dir, preparation_spec)

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    jobs = _build_jobs(train_files, validation_files, output_dir)

    print(
        f"tokenizing {len(train_files)} train + {len(validation_files)} validation "
        f"parquet files with {args.workers} workers",
        flush=True,
    )
    results: list[ShardResult] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(
            str(tokenizer_dir),
            args.text_column,
            args.batch_size,
            args.dtype,
            int(tokenizer.eos_token_id),
            args.overwrite,
        ),
    ) as executor:
        futures = {executor.submit(_tokenize_shard, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(jobs)}] {result.split}/{Path(result.destination).name}: "
                f"{result.documents:,} docs, {result.tokens:,} tokens",
                flush=True,
            )

    results.sort(key=lambda item: (item.split, item.source))
    manifest = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(Path(args.input_dir).expanduser().resolve()),
        "input_glob": args.input_glob,
        "text_column": args.text_column,
        "preprocessing_spec": preparation_spec,
        "tokenizer": {
            "source": args.tokenizer,
            "revision": args.tokenizer_revision,
            "local_path": str(tokenizer_dir),
            "vocab_size": vocab_size,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "add_special_tokens": False,
            "document_separator": "eos",
        },
        "dtype": args.dtype,
        "validation_files": [str(path.resolve()) for path in validation_files],
        "splits": {
            split: {
                "documents": sum(item.documents for item in results if item.split == split),
                "null_documents": sum(
                    item.null_documents for item in results if item.split == split
                ),
                "tokens": sum(item.tokens for item in results if item.split == split),
                "shards": sum(item.split == split for item in results),
            }
            for split in ("train", "validation")
        },
        "shard_results": [asdict(item) for item in results],
        "software": {
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "tokenizers": tokenizers.__version__,
            "transformers": transformers.__version__,
        },
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["splits"], indent=2), flush=True)
    print(f"wrote {output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; completed shards can be reused on the next run", file=sys.stderr)
        raise SystemExit(130)
