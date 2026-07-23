from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .simplex import DistributionAggregator


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = dropout

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.down_proj(hidden)


class SparseMoE(nn.Module):
    """Dropless token-routed MoE that exposes selected expert outputs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.model_type = config.model_type
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_jitter = config.router_jitter
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLU(config.d_model, config.ffn_dim, config.dropout)
                for _ in range(config.n_experts)
            ]
        )
        self.aggregator = None
        if config.model_type == "distributional_moe":
            self.aggregator = DistributionAggregator(
                d_model=config.d_model,
                distribution_k=config.distribution_k,
                method=config.aggregation,
                power_rho=config.power_rho,
                sinkhorn_epsilon=config.sinkhorn_epsilon,
                sinkhorn_iterations=config.sinkhorn_iterations,
            )

    def set_top_k(self, top_k: int) -> None:
        if not 1 <= top_k <= self.n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.top_k = top_k

    def _dispatch(
        self, inputs: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        n_tokens, top_k = expert_indices.shape
        flat_indices = expert_indices.reshape(-1)
        output_chunks: list[torch.Tensor] = []
        position_chunks: list[torch.Tensor] = []
        unused_parameter_zero = inputs.new_zeros(())

        for expert_id, expert in enumerate(self.experts):
            positions = torch.nonzero(
                flat_indices == expert_id, as_tuple=False
            ).flatten()
            if positions.numel() == 0:
                for parameter in expert.parameters():
                    unused_parameter_zero = unused_parameter_zero + parameter.sum() * 0.0
                continue
            token_indices = torch.div(positions, top_k, rounding_mode="floor")
            expert_inputs = inputs.index_select(0, token_indices)
            output_chunks.append(expert(expert_inputs))
            position_chunks.append(positions)

        if output_chunks:
            outputs = torch.cat(output_chunks, dim=0)
            positions = torch.cat(position_chunks, dim=0)
            # Under autocast the normalized residual input remains FP32 while
            # expert Linear outputs are BF16/FP16. index_copy requires exact
            # dtype agreement, so allocate in the actual expert output dtype.
            flat_output = outputs.new_zeros(n_tokens * top_k, self.d_model)
            flat_output = flat_output.index_copy(0, positions, outputs)
        else:
            flat_output = inputs.new_zeros(n_tokens * top_k, self.d_model)
        flat_output = flat_output + unused_parameter_zero.to(flat_output.dtype)
        return flat_output.reshape(n_tokens, top_k, self.d_model)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        original_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.d_model)
        router_inputs = flat_inputs
        if self.training and self.router_jitter > 0.0:
            noise = torch.empty_like(router_inputs).uniform_(
                1.0 - self.router_jitter, 1.0 + self.router_jitter
            )
            router_inputs = router_inputs * noise

        router_logits = self.router(router_inputs)
        router_probabilities = F.softmax(router_logits.float(), dim=-1)
        router_weights, expert_indices = torch.topk(
            router_probabilities, k=self.top_k, dim=-1
        )
        router_weights = router_weights / router_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        selected_outputs = self._dispatch(flat_inputs, expert_indices)
        if self.aggregator is None:
            flat_output = (
                selected_outputs
                * router_weights.to(selected_outputs.dtype)[..., None]
            ).sum(dim=1)
            distribution_metrics = {
                "distribution_entropy": flat_inputs.new_zeros((), dtype=torch.float32),
                "nonlinear_correction_ratio": flat_inputs.new_zeros(
                    (), dtype=torch.float32
                ),
            }
        else:
            flat_output, distribution_metrics = self.aggregator(
                selected_outputs, router_weights
            )

        assignment_fraction = F.one_hot(
            expert_indices, num_classes=self.n_experts
        ).float().mean(dim=(0, 1))
        mean_router_probability = router_probabilities.mean(dim=0)
        load_balance_loss = self.n_experts * (
            assignment_fraction * mean_router_probability
        ).sum()
        router_z_loss = torch.logsumexp(
            router_logits.float(), dim=-1
        ).square().mean()
        auxiliary_loss = (
            self.router_aux_loss_coef * load_balance_loss
            + self.router_z_loss_coef * router_z_loss
        )

        load_mean = assignment_fraction.mean().clamp_min(1e-12)
        metrics = {
            "router_entropy": -(
                router_probabilities
                * router_probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1).mean(),
            "load_cv": assignment_fraction.std(unbiased=False) / load_mean,
            "max_load_fraction": assignment_fraction.max(),
            "distribution_entropy": distribution_metrics["distribution_entropy"],
            "nonlinear_correction_ratio": distribution_metrics[
                "nonlinear_correction_ratio"
            ],
        }
        return flat_output.reshape(original_shape), auxiliary_loss, metrics
