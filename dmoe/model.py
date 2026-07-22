from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .moe import SparseMoE, SwiGLU


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.apply(self._initialize_module)
        self._initialize_residual_projections()

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _initialize_residual_projections(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=residual_std)
            if block.is_moe:
                for expert in block.ffn.experts:
                    nn.init.normal_(
                        expert.down_proj.weight, mean=0.0, std=residual_std
                    )
            else:
                nn.init.normal_(
                    block.ffn.down_proj.weight, mean=0.0, std=residual_std
                )

    def set_top_k(self, top_k: int) -> None:
        if not 1 <= top_k <= self.config.n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.config.top_k = top_k
        for block in self.blocks:
            if block.is_moe:
                block.ffn.set_top_k(top_k)

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
        metric_sums = [hidden.new_zeros((), dtype=torch.float32) for _ in range(5)]
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
                for index in range(5):
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

