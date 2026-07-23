from __future__ import annotations

import argparse
import contextlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dmoe.resources import (
    configure_conservative_cpu_runtime,
    configure_torch_threads,
)

RESOURCE_LIMITS = configure_conservative_cpu_runtime()

import torch
import torch.nn.functional as F

from dmoe.checkpoint import load_model_from_checkpoint
from dmoe.data import load_parquet_text_corpus, sequential_token_batches
from dmoe.model import DecoderLM
from dmoe.simplex import DistributionAggregator

configure_torch_threads(torch, RESOURCE_LIMITS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired held-out analysis of expert disagreement, token loss gain, "
            "and router-gradient changes."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        default="",
        help="Same-seed trained vanilla checkpoint for direct token-loss gains.",
    )
    parser.add_argument("--data-path", default="")
    parser.add_argument("--data-glob", default="")
    parser.add_argument("--binary-dtype", default="")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--router-gradient-batches", type=int, default=8)
    parser.add_argument("--bootstrap-iters", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="mechanism_analysis.json")
    return parser.parse_args()


def _rank(values: torch.Tensor) -> torch.Tensor:
    sorted_values, order = torch.sort(values)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    sorted_ranks = torch.empty(len(values), dtype=torch.float64)
    start = 0
    for count in counts.tolist():
        stop = start + count
        sorted_ranks[start:stop] = (start + stop - 1) / 2.0
        start = stop
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = sorted_ranks
    return ranks


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double()
    right = right.double()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) == 0.0:
        return 0.0
    return float((left * right).sum() / denominator)


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    return correlation(_rank(left), _rank(right))


def bootstrap_interval(
    values: torch.Tensor, iterations: int, seed: int
) -> tuple[float, float]:
    if len(values) < 2 or iterations <= 0:
        mean = float(values.double().mean())
        return mean, mean
    generator = torch.Generator().manual_seed(seed)
    estimates = torch.empty(iterations, dtype=torch.float64)
    values = values.double()
    for index in range(iterations):
        sample = torch.randint(
            len(values), (len(values),), generator=generator
        )
        estimates[index] = values[sample].mean()
    low, high = torch.quantile(
        estimates, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return float(low), float(high)


def bootstrap_spearman_interval(
    left: torch.Tensor, right: torch.Tensor, iterations: int, seed: int
) -> tuple[float, float]:
    if len(left) < 3 or iterations <= 0:
        estimate = spearman(left, right)
        return estimate, estimate
    generator = torch.Generator().manual_seed(seed)
    estimates = torch.empty(iterations, dtype=torch.float64)
    for index in range(iterations):
        sample = torch.randint(len(left), (len(left),), generator=generator)
        estimates[index] = spearman(left[sample], right[sample])
    low, high = torch.quantile(
        estimates, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return float(low), float(high)


def disagreement_deciles(
    disagreement: torch.Tensor,
    gain: torch.Tensor,
    correction: torch.Tensor,
) -> list[dict[str, float | int]]:
    order = torch.argsort(disagreement)
    bins = torch.tensor_split(order, 10)
    rows: list[dict[str, float | int]] = []
    for index, positions in enumerate(bins):
        if len(positions) == 0:
            continue
        rows.append(
            {
                "decile": index + 1,
                "tokens": len(positions),
                "mean_disagreement": float(disagreement[positions].mean()),
                "mean_nll_gain": float(gain[positions].mean()),
                "mean_correction_ratio": float(correction[positions].mean()),
            }
        )
    return rows


@contextlib.contextmanager
def aggregation_method(
    model: DecoderLM, method: str
) -> Iterator[None]:
    aggregators: list[DistributionAggregator] = []
    previous: list[str] = []
    for block in model.blocks:
        if block.is_moe and isinstance(
            block.ffn.aggregator, DistributionAggregator
        ):
            aggregators.append(block.ffn.aggregator)
            previous.append(block.ffn.aggregator.method)
            block.ffn.aggregator.method = method
    try:
        yield
    finally:
        for aggregator, old_method in zip(aggregators, previous):
            aggregator.method = old_method


def _autocast(device: torch.device, precision: str) -> Any:
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shape = labels.shape
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).reshape(shape)


def _router_gradient_matrix(
    loss: torch.Tensor, router_logits: list[torch.Tensor]
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, router_logits, retain_graph=False, allow_unused=False
    )
    return torch.cat([gradient.float() for gradient in gradients], dim=-1)


def _validate_checkpoint_pair(
    model: DecoderLM,
    baseline: DecoderLM,
    checkpoint: dict[str, Any],
    baseline_checkpoint: dict[str, Any],
) -> None:
    if baseline.config.model_type != "vanilla_moe":
        raise ValueError("--baseline-checkpoint must contain vanilla_moe")
    fields = (
        "vocab_size",
        "max_seq_len",
        "n_layers",
        "d_model",
        "n_heads",
        "ffn_dim",
        "n_experts",
        "top_k",
        "moe_layers",
    )
    mismatched = [
        field
        for field in fields
        if getattr(model.config, field) != getattr(baseline.config, field)
    ]
    if mismatched:
        raise ValueError(
            f"checkpoint architecture mismatch in fields: {mismatched}"
        )
    expected_config = checkpoint["config"]
    actual_config = baseline_checkpoint["config"]
    if expected_config["train"]["seed"] != actual_config["train"]["seed"]:
        raise ValueError("checkpoint pair must use the same training seed")
    if (
        expected_config["train"]["sequence_length"]
        != actual_config["train"]["sequence_length"]
    ):
        raise ValueError("checkpoint pair must use the same sequence length")
    data_fields = (
        "input_format",
        "train_path",
        "train_glob",
        "tokenizer_path",
        "eos_token_id",
        "validation_rows",
        "test_rows",
    )
    mismatched_data = [
        field
        for field in data_fields
        if expected_config["data"].get(field)
        != actual_config["data"].get(field)
    ]
    if mismatched_data:
        raise ValueError(
            f"checkpoint data contract mismatch in fields: {mismatched_data}"
        )
    if checkpoint.get("tokens_seen") != baseline_checkpoint.get("tokens_seen"):
        raise ValueError(
            "checkpoint pair must be aligned at the same training token count"
        )


def _evaluation_batches(
    stored: dict[str, Any],
    args: argparse.Namespace,
    model: DecoderLM,
) -> tuple[Iterator[tuple[torch.Tensor, torch.Tensor]], dict[str, Any] | None]:
    data_config = stored["data"]
    train_config = stored["train"]
    sequence_length = args.sequence_length or train_config["sequence_length"]
    if sequence_length > model.config.max_seq_len:
        raise ValueError("evaluation sequence length exceeds model maximum")
    max_batches = math.ceil(
        args.max_tokens / (args.batch_size * sequence_length)
    )
    if data_config.get("input_format", "binary") == "parquet_text":
        corpus = load_parquet_text_corpus(
            path=args.data_path or data_config["train_path"],
            patterns=args.data_glob or data_config["train_glob"],
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
            test_rows=int(data_config.get("test_rows", 0)),
            vocab_size=model.config.vocab_size,
            eos_token_id=int(data_config["eos_token_id"]),
            validate_token_ids=bool(data_config.get("validate_token_ids", True)),
        )
        batcher_factory = (
            corpus.validation_batcher
            if args.split == "validation"
            else corpus.test_batcher
        )
        batcher = batcher_factory(
            sequence_length=sequence_length,
            batch_size=args.batch_size,
            rank=0,
            world_size=1,
        )

        def iterator() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for _ in range(max_batches):
                try:
                    yield batcher.next_batch()
                except StopIteration:
                    return

        return iterator(), corpus.metadata

    if args.split == "test":
        raise ValueError(
            "--split test is only supported for parquet_text checkpoints"
        )
    return (
        sequential_token_batches(
            path=args.data_path or data_config["validation_path"],
            patterns=args.data_glob or data_config["validation_glob"],
            binary_dtype=args.binary_dtype or data_config["binary_dtype"],
            sequence_length=sequence_length,
            batch_size=args.batch_size,
            max_batches=max_batches,
        ),
        None,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    if model.config.model_type != "distributional_moe":
        raise ValueError("--checkpoint must contain distributional_moe")
    model.set_mechanism_collection(True)

    baseline = None
    baseline_metadata = None
    if args.baseline_checkpoint:
        baseline, baseline_metadata = load_model_from_checkpoint(
            args.baseline_checkpoint, device
        )
        _validate_checkpoint_pair(
            model, baseline, checkpoint, baseline_metadata
        )

    batches, data_metadata = _evaluation_batches(
        checkpoint["config"], args, model
    )
    token_disagreement: list[torch.Tensor] = []
    token_correction: list[torch.Tensor] = []
    token_counterfactual_gain: list[torch.Tensor] = []
    token_trained_gain: list[torch.Tensor] = []
    block_disagreement: list[torch.Tensor] = []
    block_counterfactual_gain: list[torch.Tensor] = []
    block_trained_gain: list[torch.Tensor] = []
    router_disagreement: list[torch.Tensor] = []
    router_actual_norm: list[torch.Tensor] = []
    router_geometric_norm: list[torch.Tensor] = []
    router_gradient_cosine: list[torch.Tensor] = []
    tokens = 0

    for batch_index, (input_ids, labels) in enumerate(batches):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, args.precision):
            actual = model(input_ids, labels)
        actual_snapshot = model.mechanism_snapshot()
        actual_nll = _token_nll(actual["logits"], labels).detach()
        disagreement = actual_snapshot["expert_js_divergence"].reshape_as(
            labels
        )
        correction = actual_snapshot["nonlinear_correction_ratio"].reshape_as(
            labels
        )
        actual_router_gradient = None
        if batch_index < args.router_gradient_batches:
            actual_router_gradient = _router_gradient_matrix(
                actual["lm_loss"], actual_snapshot["router_logits"]
            ).detach()
        del actual

        with aggregation_method(model, "geometric"):
            with _autocast(device, args.precision):
                geometric = model(input_ids, labels)
            geometric_snapshot = model.mechanism_snapshot()
            geometric_nll = _token_nll(
                geometric["logits"], labels
            ).detach()
            geometric_router_gradient = None
            if batch_index < args.router_gradient_batches:
                geometric_router_gradient = _router_gradient_matrix(
                    geometric["lm_loss"],
                    geometric_snapshot["router_logits"],
                ).detach()
            del geometric

        counterfactual_gain = geometric_nll - actual_nll
        trained_gain = None
        if baseline is not None:
            with torch.inference_mode(), _autocast(device, args.precision):
                baseline_output = baseline(input_ids, labels)
            trained_gain = (
                _token_nll(baseline_output["logits"], labels) - actual_nll
            )
            del baseline_output

        token_disagreement.append(disagreement.flatten().float().cpu())
        token_correction.append(correction.flatten().float().cpu())
        token_counterfactual_gain.append(
            counterfactual_gain.flatten().float().cpu()
        )
        block_disagreement.append(disagreement.mean(dim=1).float().cpu())
        block_counterfactual_gain.append(
            counterfactual_gain.mean(dim=1).float().cpu()
        )
        if trained_gain is not None:
            token_trained_gain.append(trained_gain.flatten().float().cpu())
            block_trained_gain.append(trained_gain.mean(dim=1).float().cpu())

        if (
            actual_router_gradient is not None
            and geometric_router_gradient is not None
        ):
            actual_norm = actual_router_gradient.norm(dim=-1)
            geometric_norm = geometric_router_gradient.norm(dim=-1)
            cosine = F.cosine_similarity(
                actual_router_gradient, geometric_router_gradient, dim=-1
            )
            router_disagreement.append(disagreement.flatten().float().cpu())
            router_actual_norm.append(actual_norm.float().cpu())
            router_geometric_norm.append(geometric_norm.float().cpu())
            router_gradient_cosine.append(cosine.float().cpu())
        tokens += labels.numel()
        if tokens >= args.max_tokens:
            break

    disagreement = torch.cat(token_disagreement)
    correction = torch.cat(token_correction)
    counterfactual_gain = torch.cat(token_counterfactual_gain)
    block_disagreement_tensor = torch.cat(block_disagreement)
    block_counterfactual_gain_tensor = torch.cat(block_counterfactual_gain)

    result: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "training_tokens_seen": checkpoint.get("tokens_seen"),
        "baseline_checkpoint": (
            str(Path(args.baseline_checkpoint).resolve())
            if args.baseline_checkpoint
            else None
        ),
        "baseline_checkpoint_step": (
            baseline_metadata.get("step")
            if baseline_metadata is not None
            else None
        ),
        "tokens": len(disagreement),
        "sequence_blocks": len(block_disagreement_tensor),
        "evaluation_split": args.split,
        "data_metadata": data_metadata,
        "resource_limits": RESOURCE_LIMITS.to_dict(),
        "model": {
            "model_type": model.config.model_type,
            "top_k": model.config.top_k,
            "distribution_k": model.config.distribution_k,
            "aggregation": model.config.aggregation,
            "learnable_rho": model.config.learnable_rho,
            "aggregation_rho_by_layer": model.aggregation_rho_values(),
        },
        "counterfactual_geometric": {
            "mean_nll_gain": float(counterfactual_gain.mean()),
            "mean_nll_gain_95ci": bootstrap_interval(
                block_counterfactual_gain_tensor,
                args.bootstrap_iters,
                args.bootstrap_seed,
            ),
            "token_spearman_disagreement_gain": spearman(
                disagreement, counterfactual_gain
            ),
            "block_spearman_disagreement_gain": spearman(
                block_disagreement_tensor, block_counterfactual_gain_tensor
            ),
            "block_spearman_95ci": bootstrap_spearman_interval(
                block_disagreement_tensor,
                block_counterfactual_gain_tensor,
                args.bootstrap_iters,
                args.bootstrap_seed + 1,
            ),
            "disagreement_deciles": disagreement_deciles(
                disagreement, counterfactual_gain, correction
            ),
        },
    }

    if token_trained_gain:
        trained_gain_tensor = torch.cat(token_trained_gain)
        block_trained_gain_tensor = torch.cat(block_trained_gain)
        result["trained_vanilla_pair"] = {
            "mean_nll_gain": float(trained_gain_tensor.mean()),
            "mean_nll_gain_95ci": bootstrap_interval(
                block_trained_gain_tensor,
                args.bootstrap_iters,
                args.bootstrap_seed + 2,
            ),
            "token_spearman_disagreement_gain": spearman(
                disagreement, trained_gain_tensor
            ),
            "block_spearman_disagreement_gain": spearman(
                block_disagreement_tensor, block_trained_gain_tensor
            ),
            "block_spearman_95ci": bootstrap_spearman_interval(
                block_disagreement_tensor,
                block_trained_gain_tensor,
                args.bootstrap_iters,
                args.bootstrap_seed + 3,
            ),
            "disagreement_deciles": disagreement_deciles(
                disagreement, trained_gain_tensor, correction
            ),
        }

    if router_actual_norm:
        router_disagreement_tensor = torch.cat(router_disagreement)
        actual_norm = torch.cat(router_actual_norm)
        geometric_norm = torch.cat(router_geometric_norm)
        cosine = torch.cat(router_gradient_cosine)
        norm_change = actual_norm - geometric_norm
        result["router_gradient"] = {
            "tokens": len(actual_norm),
            "mean_actual_norm": float(actual_norm.mean()),
            "mean_geometric_norm": float(geometric_norm.mean()),
            "mean_norm_ratio": float(
                (actual_norm / geometric_norm.clamp_min(1e-12)).mean()
            ),
            "mean_cosine": float(cosine.mean()),
            "spearman_disagreement_norm_change": spearman(
                router_disagreement_tensor, norm_change
            ),
            "spearman_disagreement_direction_change": spearman(
                router_disagreement_tensor, 1.0 - cosine
            ),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
