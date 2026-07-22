from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from dmoe.checkpoint import load_model_from_checkpoint
from dmoe.data import sequential_token_batches
from dmoe.distributed import (
    all_reduce_sum,
    cleanup_distributed,
    initialize_distributed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute held-out token perplexity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--data-glob", default="")
    parser.add_argument("--binary-dtype", default="")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
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
    start_time = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for input_ids, labels in sequential_token_batches(
            path=data_path,
            patterns=data_glob,
            binary_dtype=binary_dtype,
            sequence_length=sequence_length,
            batch_size=args.batch_size,
            rank=context.rank,
            world_size=context.world_size,
            max_batches=max_batches,
        ):
            input_ids = input_ids.to(context.device, non_blocking=True)
            labels = labels.to(context.device, non_blocking=True)
            autocast = (
                torch.autocast(context.device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else torch.autocast(context.device.type, enabled=False)
            )
            with autocast:
                output = model(input_ids, labels)
            count = labels.numel()
            total[0] += output["lm_loss"].double() * count
            total[1] += count
    all_reduce_sum(total, context)
    elapsed = time.perf_counter() - start_time
    nll = (total[0] / total[1].clamp_min(1)).item()
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "training_tokens_seen": checkpoint.get("tokens_seen"),
        "model_type": model.config.model_type,
        "data_path": data_path,
        "top_k": model.config.top_k,
        "aggregation": model.config.aggregation,
        **model.parameter_report(),
        "tokens": int(total[1].item()),
        "mean_nll": nll,
        "perplexity": math.exp(min(nll, 50.0)),
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
