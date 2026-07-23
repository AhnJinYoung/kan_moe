from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterator

from dmoe.resources import (
    configure_conservative_cpu_runtime,
    configure_torch_threads,
)

RESOURCE_LIMITS = configure_conservative_cpu_runtime()

import torch
import torch.nn.functional as F

from dmoe.checkpoint import load_model_from_checkpoint
from dmoe.data import load_parquet_text_corpus, sequential_token_batches
from dmoe.distributed import (
    all_reduce_sum,
    cleanup_distributed,
    gather_objects,
    initialize_distributed,
)

configure_torch_threads(torch, RESOURCE_LIMITS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute held-out token perplexity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--data-glob", default="")
    parser.add_argument("--binary-dtype", default="")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--bootstrap-iters", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = initialize_distributed()
    model, checkpoint = load_model_from_checkpoint(
        args.checkpoint, context.device, top_k=args.top_k
    )
    stored = checkpoint["config"]
    data_config = stored["data"]
    train_config = stored["train"]
    data_path = args.data_path or data_config["validation_path"]
    data_glob = args.data_glob or data_config["validation_glob"]
    binary_dtype = args.binary_dtype or data_config["binary_dtype"]
    sequence_length = args.sequence_length or train_config["sequence_length"]
    if sequence_length > model.config.max_seq_len:
        raise ValueError("evaluation sequence length exceeds model maximum")
    global_tokens_per_batch = (
        args.batch_size * sequence_length * context.world_size
    )
    max_batches = (
        math.ceil(args.max_tokens / global_tokens_per_batch)
        if args.max_tokens > 0
        else 0
    )
    if args.precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif args.precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    total = torch.zeros(2, dtype=torch.float64, device=context.device)
    local_block_nll: list[float] = []
    start_time = time.perf_counter()
    model.eval()
    data_metadata = None
    if data_config.get("input_format", "binary") == "parquet_text":
        data_path = args.data_path or data_config["train_path"]
        data_glob = args.data_glob or data_config["train_glob"]
        corpus = load_parquet_text_corpus(
            path=data_path,
            patterns=data_glob,
            tokenizer_path=data_config["tokenizer_path"],
            tokenizer_revision=data_config.get("tokenizer_revision", ""),
            text_column=data_config.get("text_column", "text"),
            cache_dir=data_config.get("hf_cache_dir", ""),
            dataset_num_proc=min(
                int(data_config.get("dataset_num_proc", 1)),
                RESOURCE_LIMITS.data_workers,
            ),
            tokenizer_batch_size=min(
                int(data_config.get("tokenizer_batch_size", 4)),
                RESOURCE_LIMITS.tokenizer_batch_limit,
            ),
            parquet_backend=data_config.get("parquet_backend", "direct"),
            parquet_read_batch_size=min(
                int(data_config.get("parquet_read_batch_size", 4)),
                RESOURCE_LIMITS.parquet_batch_limit,
            ),
            validation_rows=int(data_config.get("validation_rows", 10_000)),
            vocab_size=model.config.vocab_size,
            eos_token_id=int(data_config["eos_token_id"]),
            validate_token_ids=bool(data_config.get("validate_token_ids", True)),
        )
        online_batcher = corpus.validation_batcher(
            sequence_length=sequence_length,
            batch_size=args.batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )
        data_metadata = corpus.metadata

        def evaluation_batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            yielded = 0
            while max_batches <= 0 or yielded < max_batches:
                try:
                    yield online_batcher.next_batch()
                except StopIteration:
                    return
                yielded += 1

        batches = evaluation_batches()
    else:
        batches = sequential_token_batches(
            path=data_path,
            patterns=data_glob,
            binary_dtype=binary_dtype,
            sequence_length=sequence_length,
            batch_size=args.batch_size,
            rank=context.rank,
            world_size=context.world_size,
            max_batches=max_batches,
        )
    with torch.inference_mode():
        for input_ids, labels in batches:
            input_ids = input_ids.to(context.device, non_blocking=True)
            labels = labels.to(context.device, non_blocking=True)
            autocast = (
                torch.autocast(context.device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else torch.autocast(context.device.type, enabled=False)
            )
            with autocast:
                output = model(input_ids, labels)
            block_nll = F.cross_entropy(
                output["logits"].float().reshape(
                    -1, output["logits"].shape[-1]
                ),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape).mean(dim=1)
            local_block_nll.extend(block_nll.double().cpu().tolist())
            count = labels.numel()
            total[0] += output["lm_loss"].double() * count
            total[1] += count
    all_reduce_sum(total, context)
    elapsed = time.perf_counter() - start_time
    nll = (total[0] / total[1].clamp_min(1)).item()
    gathered_block_nll = gather_objects(local_block_nll, context)
    block_nll_tensor = torch.tensor(
        [
            value
            for rank_values in gathered_block_nll
            for value in rank_values
        ],
        dtype=torch.float64,
    )
    bootstrap_ci = (nll, nll)
    if context.is_main and len(block_nll_tensor) > 1 and args.bootstrap_iters > 0:
        generator = torch.Generator().manual_seed(args.bootstrap_seed)
        estimates = torch.empty(args.bootstrap_iters, dtype=torch.float64)
        for index in range(args.bootstrap_iters):
            sample = torch.randint(
                len(block_nll_tensor),
                (len(block_nll_tensor),),
                generator=generator,
            )
            estimates[index] = block_nll_tensor[sample].mean()
        low, high = torch.quantile(
            estimates, torch.tensor([0.025, 0.975], dtype=torch.float64)
        )
        bootstrap_ci = (float(low), float(high))
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "training_tokens_seen": checkpoint.get("tokens_seen"),
        "training_seed": train_config.get("seed"),
        "model_type": model.config.model_type,
        "data_path": data_path,
        "data_metadata": data_metadata,
        "resource_limits": RESOURCE_LIMITS.to_dict(),
        "top_k": model.config.top_k,
        "aggregation": model.config.aggregation,
        "aggregation_rho_by_layer": model.aggregation_rho_values(),
        **model.parameter_report(),
        "tokens": int(total[1].item()),
        "mean_nll": nll,
        "sequence_blocks": len(block_nll_tensor),
        "block_nll_std": (
            float(block_nll_tensor.std(unbiased=True))
            if len(block_nll_tensor) > 1
            else 0.0
        ),
        "mean_nll_95ci": bootstrap_ci,
        "perplexity": math.exp(min(nll, 50.0)),
        "perplexity_95ci": tuple(
            math.exp(min(value, 50.0)) for value in bootstrap_ci
        ),
        "elapsed_seconds": elapsed,
        "tokens_per_second": total[1].item() / max(elapsed, 1e-9),
    }
    if context.is_main:
        print(json.dumps(result, indent=2))
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
    cleanup_distributed()


if __name__ == "__main__":
    main()
