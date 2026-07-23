from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from dmoe.gpu_select import GPUSelection, configure_cuda_visibility
from dmoe.resources import (
    configure_conservative_cpu_runtime,
    configure_torch_threads,
)


def _automatic_gpu_selection_enabled() -> bool:
    disabled_by_cli = "--no-auto-select-gpu" in sys.argv
    disabled_by_environment = os.environ.get(
        "DMOE_AUTO_SELECT_GPU", "1"
    ).lower() in {"0", "false", "no", "off"}
    return not disabled_by_cli and not disabled_by_environment


RESOURCE_LIMITS = configure_conservative_cpu_runtime()
GPU_SELECTION: GPUSelection = configure_cuda_visibility(
    enabled=_automatic_gpu_selection_enabled()
)

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from dmoe.checkpoint import (
    find_latest_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from dmoe.config import ExperimentConfig, load_experiment_config
from dmoe.data import (
    RandomTokenBatcher,
    load_parquet_text_corpus,
    sequential_token_batches,
    validate_data_manifest,
)
from dmoe.distributed import (
    DistributedContext,
    all_reduce_sum,
    barrier,
    capture_rng_state,
    cleanup_distributed,
    gather_objects,
    initialize_distributed,
    restore_rng_state,
    seed_everything,
)
from dmoe.model import DecoderLM

configure_torch_threads(torch, RESOURCE_LIMITS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dense or MoE decoder LM")
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument(
        "--no-auto-select-gpu",
        action="store_true",
        help=(
            "Disable idle-GPU discovery. An existing CUDA_VISIBLE_DEVICES value "
            "is always respected regardless of this flag."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Configuration override, e.g. model.top_k=4",
    )
    return parser.parse_args()


def build_optimizer(model: DecoderLM, config: ExperimentConfig) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {
        "lr": config.train.learning_rate,
        "betas": (config.train.beta1, config.train.beta2),
        "eps": config.train.adam_epsilon,
    }
    if torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__annotations__:
        kwargs["fused"] = True
    else:
        # PyTorch does not reliably expose keyword annotations on every version.
        try:
            return torch.optim.AdamW(groups, fused=torch.cuda.is_available(), **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(groups, **kwargs)


def learning_rate_at_step(step: int, total_steps: int, config: ExperimentConfig) -> float:
    if step < config.train.warmup_steps:
        return config.train.learning_rate * (step + 1) / max(config.train.warmup_steps, 1)
    decay_steps = max(total_steps - config.train.warmup_steps, 1)
    progress = min(1.0, (step - config.train.warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.train.min_learning_rate + cosine * (
        config.train.learning_rate - config.train.min_learning_rate
    )


def autocast_factory(
    device: torch.device, precision: str
) -> Callable[[], contextlib.AbstractContextManager[Any]]:
    if precision == "fp32":
        return contextlib.nullcontext
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    def factory() -> contextlib.AbstractContextManager[Any]:
        return torch.autocast(device_type=device.type, dtype=dtype)

    return factory


@torch.no_grad()
def evaluate_validation(
    model: DecoderLM,
    config: ExperimentConfig,
    context: DistributedContext,
    autocast: Callable[[], contextlib.AbstractContextManager[Any]],
    online_batcher_factory: Callable[[], Any] | None = None,
) -> dict[str, float]:
    model.eval()
    total = torch.zeros(2, dtype=torch.float64, device=context.device)
    if online_batcher_factory is None:
        batches = sequential_token_batches(
            path=config.data.validation_path,
            patterns=config.data.validation_glob,
            binary_dtype=config.data.binary_dtype,
            sequence_length=config.train.sequence_length,
            batch_size=config.train.micro_batch_size,
            rank=context.rank,
            world_size=context.world_size,
            max_batches=config.train.eval_batches,
        )
    else:
        online_batcher = online_batcher_factory()

        def online_batches() -> Any:
            for _ in range(config.train.eval_batches):
                try:
                    yield online_batcher.next_batch()
                except StopIteration:
                    return

        batches = online_batches()
    for input_ids, labels in batches:
        input_ids = input_ids.to(context.device, non_blocking=True)
        labels = labels.to(context.device, non_blocking=True)
        with autocast():
            outputs = model(input_ids, labels)
        token_count = labels.numel()
        total[0] += outputs["lm_loss"].double() * token_count
        total[1] += token_count
    all_reduce_sum(total, context)
    mean_nll = (total[0] / total[1].clamp_min(1)).item()
    model.train()
    return {
        "validation_nll": mean_nll,
        "validation_ppl": math.exp(min(mean_nll, 50.0)),
        "validation_tokens": int(total[1].item()),
    }


def maybe_initialize_wandb(
    config: ExperimentConfig,
    context: DistributedContext,
    runtime_info: dict[str, Any],
) -> Any:
    if not context.is_main or not config.train.wandb_project:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb_project is set but wandb is not installed") from error
    return wandb.init(
        project=config.train.wandb_project,
        name=config.train.wandb_run_name or None,
        config={**config.to_dict(), "runtime": runtime_info},
    )


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.override)
    expects_cuda = GPU_SELECTION.auto_selected or (
        GPU_SELECTION.mode == "explicit"
        and GPU_SELECTION.visible_devices.strip() not in {"", "-1"}
    )
    if expects_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "a CUDA device was selected, but this PyTorch process cannot access "
            "CUDA. This is independent of GPU auto-selection. "
            f"Installed torch={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda!r}. Check the CUDA-enabled "
            "PyTorch wheel against the NVIDIA driver's supported CUDA version. "
            "For a driver supporting CUDA 12.8, install torch==2.11.0 from "
            "https://download.pytorch.org/whl/cu128."
        )
    if GPU_SELECTION.mode == "unavailable" and torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is available, but idle-GPU discovery could not run: "
            f"{GPU_SELECTION.detail}. Set CUDA_VISIBLE_DEVICES explicitly or fix "
            "nvidia-smi."
        )
    context = initialize_distributed()
    seed_everything(config.train.seed, 0)
    runtime_info = {
        "gpu_selection": asdict(GPU_SELECTION),
        "resource_limits": RESOURCE_LIMITS.to_dict(),
        "world_size": context.world_size,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }

    online_corpus = None
    validation_batcher_factory: Callable[[], Any] | None = None
    if config.data.input_format == "parquet_text":
        online_corpus = load_parquet_text_corpus(
            path=config.data.train_path,
            patterns=config.data.train_glob,
            tokenizer_path=config.data.tokenizer_path,
            tokenizer_revision=config.data.tokenizer_revision,
            text_column=config.data.text_column,
            cache_dir=config.data.hf_cache_dir,
            dataset_num_proc=min(
                config.data.dataset_num_proc, RESOURCE_LIMITS.data_workers
            ),
            tokenizer_batch_size=min(
                config.data.tokenizer_batch_size,
                RESOURCE_LIMITS.tokenizer_batch_limit,
            ),
            parquet_backend=config.data.parquet_backend,
            parquet_read_batch_size=min(
                config.data.parquet_read_batch_size,
                RESOURCE_LIMITS.parquet_batch_limit,
            ),
            validation_rows=config.data.validation_rows,
            vocab_size=config.model.vocab_size,
            eos_token_id=config.data.eos_token_id,
            validate_token_ids=config.data.validate_token_ids,
        )
        runtime_info["data"] = online_corpus.metadata
        train_batcher = online_corpus.train_batcher(
            sequence_length=config.train.sequence_length,
            batch_size=config.train.micro_batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )

        def make_validation_batcher() -> Any:
            assert online_corpus is not None
            return online_corpus.validation_batcher(
                sequence_length=config.train.sequence_length,
                batch_size=config.train.micro_batch_size,
                rank=context.rank,
                world_size=context.world_size,
            )

        validation_batcher_factory = make_validation_batcher
        if context.is_main:
            print(
                "resource limits: "
                f"effective_cpus={RESOURCE_LIMITS.effective_cpus}, "
                f"cpu_threads={RESOURCE_LIMITS.cpu_threads}, "
                f"memory_limit_gib="
                f"{RESOURCE_LIMITS.to_dict()['cgroup_memory_limit_gib']}"
            )
            cache_files = online_corpus.metadata.get("hf_cache_files", [])
            print(
                "loaded raw Parquet text source: "
                f"{online_corpus.metadata['total_rows']} rows; "
                f"backend={online_corpus.metadata['parquet_backend']}; "
                f"materializes_arrow_cache="
                f"{online_corpus.metadata['materializes_arrow_cache']}; "
                f"read_batch={online_corpus.metadata.get('parquet_read_batch_size')}; "
                f"tokenizer_batch="
                f"{online_corpus.metadata.get('tokenizer_batch_size')}"
            )
            if cache_files:
                print("HF cache files:")
                for cache_file in cache_files:
                    print(f"  {cache_file}")
            else:
                print("HF cache_files is empty; the dataset may be held in memory")
    elif config.data.manifest_path:
        manifest = validate_data_manifest(
            config.data.manifest_path,
            vocab_size=config.model.vocab_size,
            eos_token_id=config.data.eos_token_id,
            binary_dtype=config.data.binary_dtype,
        )
        if context.is_main:
            tokenizer = manifest["tokenizer"]
            print(
                "validated token data manifest: "
                f"{tokenizer['source']}@{tokenizer['revision']}"
            )
        runtime_info["data"] = {
            "input_format": "binary",
            "manifest_path": str(Path(config.data.manifest_path).resolve()),
            "manifest": manifest,
        }

    if config.data.input_format == "binary":
        train_batcher = RandomTokenBatcher(
            path=config.data.train_path,
            patterns=config.data.train_glob,
            binary_dtype=config.data.binary_dtype,
            sequence_length=config.train.sequence_length,
            batch_size=config.train.micro_batch_size,
            seed=config.train.seed,
            rank=context.rank,
            vocab_size=config.model.vocab_size,
            validate_token_ids=config.data.validate_token_ids,
        )

    if context.device.type != "cuda" and context.is_main:
        print("warning: CUDA is unavailable; this run will use CPU")
    if config.train.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_dir = Path(config.train.output_dir)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
        with (output_dir / "runtime.json").open("w", encoding="utf-8") as handle:
            json.dump(runtime_info, handle, indent=2)
    barrier(context)

    model = DecoderLM(config.model).to(context.device)
    optimizer = build_optimizer(model, config)
    fp16_enabled = config.train.precision == "fp16" and context.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=fp16_enabled)

    start_step = 0
    tokens_seen = 0
    resume_path = config.train.resume
    if resume_path == "latest":
        latest = find_latest_checkpoint(output_dir)
        resume_path = str(latest) if latest is not None else ""
    resumed = bool(resume_path)
    if resume_path:
        checkpoint_state = load_training_checkpoint(
            resume_path, model, optimizer=optimizer, scaler=scaler
        )
        start_step = int(checkpoint_state["step"])
        tokens_seen = int(checkpoint_state.get("tokens_seen", 0))
        sampler_states = checkpoint_state.get("sampler_states", [])
        rng_states = checkpoint_state.get("rng_states", [])
        if context.rank < len(sampler_states):
            train_batcher.load_state_dict(sampler_states[context.rank])
        if context.rank < len(rng_states):
            restore_rng_state(rng_states[context.rank])
        if context.is_main:
            print(f"resumed training from {resume_path} at step {start_step}")
        del checkpoint_state
    elif config.train.init_from:
        load_training_checkpoint(config.train.init_from, model)
        if context.is_main:
            print(f"initialized model weights from {config.train.init_from}")

    if context.is_main:
        print(json.dumps(model.parameter_report(), indent=2))

    raw_model = model
    train_model: torch.nn.Module = model
    if context.distributed:
        train_model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    if config.train.compile_model:
        train_model = torch.compile(train_model)

    # Use rank-dependent stochastic streams after DDP has synchronized parameters.
    # A resumed run has already restored the exact per-rank RNG state.
    if not resumed:
        seed_everything(config.train.seed, context.rank)
    autocast = autocast_factory(context.device, config.train.precision)
    tokens_per_step = (
        config.train.sequence_length
        * config.train.micro_batch_size
        * config.train.gradient_accumulation_steps
        * context.world_size
    )
    total_steps = config.train.max_steps
    if config.train.max_tokens > 0:
        token_limited_steps = math.ceil(config.train.max_tokens / tokens_per_step)
        total_steps = min(total_steps, token_limited_steps) if total_steps > 0 else token_limited_steps

    wandb_run = maybe_initialize_wandb(config, context, runtime_info)
    metrics_path = output_dir / "metrics.jsonl"
    last_log_time = time.perf_counter()
    last_logged_tokens = tokens_seen
    optimizer.zero_grad(set_to_none=True)

    try:
        for step_index in range(start_step, total_steps):
            current_step = step_index + 1
            learning_rate = learning_rate_at_step(step_index, total_steps, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            accumulated = torch.zeros(8, dtype=torch.float64, device=context.device)
            for micro_step in range(config.train.gradient_accumulation_steps):
                input_ids, labels = train_batcher.next_batch(context.device)
                synchronize = micro_step == config.train.gradient_accumulation_steps - 1
                sync_context = contextlib.nullcontext()
                if context.distributed and not synchronize:
                    sync_context = train_model.no_sync()  # type: ignore[union-attr]
                with sync_context, autocast():
                    outputs = train_model(input_ids, labels)
                    loss = outputs["loss"] / config.train.gradient_accumulation_steps
                scaler.scale(loss).backward()
                accumulated[0] += outputs["lm_loss"].detach().double()
                accumulated[1] += outputs["aux_loss"].detach().double()
                for metric_index, metric_name in enumerate(
                    [
                        "router_entropy",
                        "load_cv",
                        "max_load_fraction",
                        "distribution_entropy",
                        "nonlinear_correction_ratio",
                    ],
                    start=2,
                ):
                    accumulated[metric_index] += outputs["metrics"][metric_name].detach().double()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), config.train.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            tokens_seen += tokens_per_step

            if current_step % config.train.log_interval == 0:
                all_reduce_sum(accumulated, context)
                accumulated /= config.train.gradient_accumulation_steps * context.world_size
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-9)
                throughput = (tokens_seen - last_logged_tokens) / elapsed
                record = {
                    "step": current_step,
                    "tokens_seen": tokens_seen,
                    "learning_rate": learning_rate,
                    "train_nll": accumulated[0].item(),
                    "aux_loss": accumulated[1].item(),
                    "router_entropy": accumulated[2].item(),
                    "load_cv": accumulated[3].item(),
                    "max_load_fraction": accumulated[4].item(),
                    "distribution_entropy": accumulated[5].item(),
                    "nonlinear_correction_ratio": accumulated[6].item(),
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": throughput,
                }
                if context.is_main:
                    print(json.dumps(record, sort_keys=True))
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    if wandb_run is not None:
                        wandb_run.log(record, step=current_step)
                last_log_time = now
                last_logged_tokens = tokens_seen

            if config.train.eval_interval > 0 and current_step % config.train.eval_interval == 0:
                validation = evaluate_validation(
                    raw_model,
                    config,
                    context,
                    autocast,
                    validation_batcher_factory,
                )
                validation.update({"step": current_step, "tokens_seen": tokens_seen})
                if context.is_main:
                    print(json.dumps(validation, sort_keys=True))
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(validation) + "\n")
                    if wandb_run is not None:
                        wandb_run.log(validation, step=current_step)

            should_save = config.train.save_interval > 0 and current_step % config.train.save_interval == 0
            is_final = current_step == total_steps
            if should_save or is_final:
                sampler_states = gather_objects(train_batcher.state_dict(), context)
                rng_states = gather_objects(capture_rng_state(), context)
                if context.is_main:
                    path = save_checkpoint(
                        output_dir=output_dir,
                        step=current_step,
                        tokens_seen=tokens_seen,
                        model=raw_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        sampler_states=sampler_states,
                        rng_states=rng_states,
                        keep_last=config.train.keep_last_checkpoints,
                    )
                    print(f"saved checkpoint: {path}")
                barrier(context)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
