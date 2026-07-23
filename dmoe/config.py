from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    model_type: str = "distributional_moe"
    vocab_size: int = 32_768
    max_seq_len: int = 2_048
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    ffn_dim: int = 1_920
    n_experts: int = 16
    top_k: int = 2
    moe_layers: list[int] = field(
        default_factory=lambda: [1, 3, 5, 7, 9, 11]
    )
    distribution_k: int = 9
    aggregation: str = "hellinger"
    power_rho: float = 0.5
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iterations: int = 8
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_jitter: float = 0.0
    tie_embeddings: bool = True
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        valid_types = {"dense", "vanilla_moe", "distributional_moe"}
        if self.model_type not in valid_types:
            raise ValueError(
                f"model_type must be one of {sorted(valid_types)}, got {self.model_type}"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        if any(index < 0 or index >= self.n_layers for index in self.moe_layers):
            raise ValueError("moe_layers contains an out-of-range layer index")
        if len(set(self.moe_layers)) != len(self.moe_layers):
            raise ValueError("moe_layers must not contain duplicates")
        if self.model_type == "dense" and self.moe_layers:
            raise ValueError("dense config must set moe_layers to []")
        if self.model_type != "dense" and not self.moe_layers:
            raise ValueError("MoE configs require at least one MoE layer")
        if self.model_type == "distributional_moe":
            if self.distribution_k < 2:
                raise ValueError("distribution_k must be at least 2")
            if self.d_model % (self.distribution_k - 1) != 0:
                raise ValueError(
                    "d_model must be divisible by distribution_k - 1"
                )
            valid_aggregations = {
                "geometric",
                "power",
                "hellinger",
                "arithmetic",
                "wasserstein",
            }
            if self.aggregation not in valid_aggregations:
                raise ValueError(
                    f"aggregation must be one of {sorted(valid_aggregations)}"
                )
        if self.vocab_size <= 0 or self.max_seq_len <= 1:
            raise ValueError("vocab_size and max_seq_len must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class DataConfig:
    input_format: str = "binary"
    train_path: str = "data/fineweb_edu/train"
    validation_path: str = "data/fineweb_edu/validation"
    train_glob: str = "*.bin,*.npy"
    validation_glob: str = "*.bin,*.npy"
    binary_dtype: str = "uint16"
    eos_token_id: int = 2
    tokenizer_path: str = ""
    tokenizer_revision: str = ""
    manifest_path: str = ""
    validate_token_ids: bool = True
    text_column: str = "text"
    hf_cache_dir: str = ""
    parquet_backend: str = "direct"
    parquet_read_batch_size: int = 4
    dataset_num_proc: int = 1
    tokenizer_batch_size: int = 4
    validation_rows: int = 10_000

    def validate(self) -> None:
        if self.input_format not in {"binary", "parquet_text"}:
            raise ValueError("input_format must be binary or parquet_text")
        if self.binary_dtype not in {"uint16", "uint32", "int32", "int64"}:
            raise ValueError(f"unsupported binary_dtype: {self.binary_dtype}")
        if self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")
        if self.input_format == "parquet_text":
            if not self.tokenizer_path:
                raise ValueError("tokenizer_path is required for parquet_text input")
            if not self.text_column:
                raise ValueError("text_column must be non-empty")
            if self.parquet_backend not in {"direct", "hf_cache"}:
                raise ValueError("parquet_backend must be direct or hf_cache")
            if (
                self.parquet_read_batch_size <= 0
                or self.dataset_num_proc <= 0
                or self.tokenizer_batch_size <= 0
            ):
                raise ValueError(
                    "Parquet, dataset worker, and tokenizer batch sizes must be positive"
                )
            if self.validation_rows <= 0:
                raise ValueError("validation_rows must be positive")


@dataclass
class TrainConfig:
    output_dir: str = "outputs/distributional_moe_500m"
    seed: int = 1337
    sequence_length: int = 2_048
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 32
    max_steps: int = 10_000
    max_tokens: int = 0
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0
    precision: str = "bf16"
    use_tf32: bool = True
    compile_model: bool = False
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 50
    save_interval: int = 500
    keep_last_checkpoints: int = 3
    resume: str = ""
    init_from: str = ""
    wandb_project: str = ""
    wandb_run_name: str = ""

    def validate(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.micro_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch size and accumulation steps must be positive")
        if self.max_steps <= 0 and self.max_tokens <= 0:
            raise ValueError("at least one of max_steps or max_tokens must be positive")
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be bf16, fp16, or fp32")
        if not 0 <= self.warmup_steps:
            raise ValueError("warmup_steps must be non-negative")


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.train.validate()
        if self.train.sequence_length > self.model.max_seq_len:
            raise ValueError("training sequence_length exceeds model max_seq_len")
        if self.data.eos_token_id >= self.model.vocab_size:
            raise ValueError("eos_token_id must be smaller than vocab_size")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        config = cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
        )
        config.validate()
        return config


def _parse_override_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        return value


def _apply_override(raw: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must have key=value form: {override}")
    path, value = override.split("=", 1)
    keys = path.split(".")
    cursor: dict[str, Any] = raw
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = _parse_override_value(value)


def load_experiment_config(
    path: str | Path, overrides: list[str] | None = None
) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    for override in overrides or []:
        _apply_override(raw, override)
    return ExperimentConfig.from_dict(raw)
