from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .moe import (
    OutputGatedAggregator,
    ResidualMLPAggregator,
    SparseMoE,
    SwiGLU,
)
from .simplex import DistributionAggregator


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(inputs.dtype) * self.weight


def rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_seq_len).float()
        frequencies = torch.outer(positions, inverse_frequency)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cosine", embedding.cos(), persistent=False)
        self.register_buffer("sine", embedding.sin(), persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        cosine = self.cosine[:sequence_length].to(query.dtype)[None, None, :, :]
        sine = self.sine[:sequence_length].to(query.dtype)[None, None, :, :]
        return (
            query * cosine + rotate_half(query) * sine,
            key * cosine + rotate_half(key) * sine,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dim, config.max_seq_len, config.rope_theta
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        query, key, value = self.qkv_proj(inputs).chunk(3, dim=-1)

        def shape_projection(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(shape_projection, (query, key, value))
        query, key = self.rotary(query, key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_model
        )
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.is_moe = config.model_type != "dense" and layer_index in config.moe_layers
        if self.is_moe:
            self.ffn: nn.Module = SparseMoE(config)
        else:
            self.ffn = SwiGLU(config.d_model, config.ffn_dim, config.dropout)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        if self.is_moe:
            ffn_output, auxiliary_loss, metrics = self.ffn(self.ffn_norm(hidden))
        else:
            ffn_output = self.ffn(self.ffn_norm(hidden))
            zero = hidden.new_zeros((), dtype=torch.float32)
            auxiliary_loss = zero
            metrics = {
                "router_entropy": zero,
                "load_cv": zero,
                "max_load_fraction": zero,
                "distribution_entropy": zero,
                "nonlinear_correction_ratio": zero,
                "aggregation_rho": zero,
                "expert_js_divergence": zero,
            }
        hidden = hidden + ffn_output
        return (
            hidden,
            auxiliary_loss,
            metrics["router_entropy"],
            metrics["load_cv"],
            metrics["max_load_fraction"],
            metrics["distribution_entropy"],
            metrics["nonlinear_correction_ratio"],
            metrics["aggregation_rho"],
            metrics["expert_js_divergence"],
        )


class DecoderLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, index) for index in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = None
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._initialize_parameters()

    def _module_seed(self, name: str, purpose: str) -> int:
        payload = (
            f"{self.config.initialization_seed}:{name}:{purpose}"
        ).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "little") % (2**63 - 1)

    def _normal_weight(
        self, module: nn.Module, name: str, standard_deviation: float
    ) -> None:
        weight = module.weight
        if weight.device.type == "meta":
            nn.init.normal_(weight, mean=0.0, std=standard_deviation)
            return
        generator = torch.Generator(device=weight.device)
        generator.manual_seed(self._module_seed(name, "normal"))
        nn.init.normal_(
            weight,
            mean=0.0,
            std=standard_deviation,
            generator=generator,
        )

    def _initialize_parameters(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                self._normal_weight(module, name, 0.02)

        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block_index, block in enumerate(self.blocks):
            self._normal_weight(
                block.attention.out_proj,
                f"blocks.{block_index}.attention.out_proj.residual",
                residual_std,
            )
            if block.is_moe:
                for expert_index, expert in enumerate(block.ffn.experts):
                    self._normal_weight(
                        expert.down_proj,
                        (
                            f"blocks.{block_index}.ffn.experts."
                            f"{expert_index}.down_proj.residual"
                        ),
                        residual_std,
                    )
                if isinstance(block.ffn.aggregator, ResidualMLPAggregator):
                    block.ffn.aggregator.zero_initialize_output()
                if isinstance(block.ffn.aggregator, OutputGatedAggregator):
                    block.ffn.aggregator.zero_initialize_score()
            else:
                self._normal_weight(
                    block.ffn.down_proj,
                    f"blocks.{block_index}.ffn.down_proj.residual",
                    residual_std,
                )

    def set_top_k(self, top_k: int) -> None:
        if not 1 <= top_k <= self.config.n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.config.top_k = top_k
        for block in self.blocks:
            if block.is_moe:
                block.ffn.set_top_k(top_k)

    def set_mechanism_collection(self, enabled: bool) -> None:
        if self.config.model_type != "distributional_moe" and enabled:
            raise ValueError(
                "token-level mechanism collection requires distributional_moe"
            )
        for block in self.blocks:
            if block.is_moe:
                block.ffn.set_collect_mechanism_metrics(enabled)

    def mechanism_snapshot(self) -> dict[str, Any]:
        token_metrics: dict[str, list[torch.Tensor]] = {
            "expert_js_divergence": [],
            "nonlinear_correction_ratio": [],
        }
        router_logits: list[torch.Tensor] = []
        for block in self.blocks:
            if not block.is_moe:
                continue
            if block.ffn.last_router_logits is not None:
                router_logits.append(block.ffn.last_router_logits)
            aggregator = block.ffn.aggregator
            if isinstance(aggregator, DistributionAggregator):
                for name in token_metrics:
                    value = aggregator.last_token_metrics.get(name)
                    if value is not None:
                        token_metrics[name].append(value)
        result: dict[str, Any] = {"router_logits": router_logits}
        for name, values in token_metrics.items():
            if values:
                result[name] = torch.stack(values, dim=0).mean(dim=0)
        return result

    def aggregation_rho_values(self) -> list[float]:
        values: list[float] = []
        for block in self.blocks:
            if block.is_moe and isinstance(
                block.ffn.aggregator, DistributionAggregator
            ):
                values.append(block.ffn.aggregator.reported_rho())
        return values

    def parameter_report(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        expert_total = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if ".experts." in name
        )
        if expert_total == 0:
            active = total
        else:
            active = total - expert_total + expert_total * self.config.top_k // self.config.n_experts
        return {
            "total_parameters": total,
            "expert_parameters": expert_total,
            "estimated_active_parameters": active,
            "active_fraction": active / total,
        }

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have [batch, sequence] shape")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds model max_seq_len")
        hidden = self.token_embedding(input_ids)
        metric_sums = [hidden.new_zeros((), dtype=torch.float32) for _ in range(7)]
        auxiliary_loss = hidden.new_zeros((), dtype=torch.float32)
        moe_layer_count = 0

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                outputs = checkpoint(block, hidden, use_reentrant=False)
            else:
                outputs = block(hidden)
            hidden = outputs[0]
            auxiliary_loss = auxiliary_loss + outputs[1]
            if block.is_moe:
                moe_layer_count += 1
                for index in range(7):
                    metric_sums[index] = metric_sums[index] + outputs[index + 2]

        hidden = self.final_norm(hidden)
        output_weight = (
            self.token_embedding.weight if self.lm_head is None else self.lm_head.weight
        )
        logits = F.linear(hidden, output_weight)
        language_model_loss = None
        total_loss = None
        if labels is not None:
            language_model_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
            total_loss = language_model_loss + auxiliary_loss

        divisor = max(moe_layer_count, 1)
        metric_names = [
            "router_entropy",
            "load_cv",
            "max_load_fraction",
            "distribution_entropy",
            "nonlinear_correction_ratio",
            "aggregation_rho",
            "expert_js_divergence",
        ]
        metrics = {
            name: value / divisor for name, value in zip(metric_names, metric_sums)
        }
        metrics["auxiliary_loss"] = auxiliary_loss.detach()
        return {
            "logits": logits,
            "loss": total_loss,
            "lm_loss": language_model_loss,
            "aux_loss": auxiliary_loss,
            "metrics": metrics,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            model_input = generated[:, -self.config.max_seq_len :]
            logits = self(model_input)["logits"][:, -1, :]
            next_token = logits.argmax(dim=-1)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished, torch.full_like(next_token, eos_token_id), next_token
                )
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if finished.all():
                    break
        self.train(was_training)
        return generated
