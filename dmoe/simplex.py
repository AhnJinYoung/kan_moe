from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def helmert_contrast(size: int) -> torch.Tensor:
    """Return an orthonormal (size - 1) x size Helmert contrast matrix."""
    if size < 2:
        raise ValueError("Helmert matrix size must be at least 2")
    matrix = torch.zeros(size - 1, size, dtype=torch.float32)
    for row in range(size - 1):
        denominator = math.sqrt((row + 1) * (row + 2))
        matrix[row, : row + 1] = 1.0 / denominator
        matrix[row, row + 1] = -(row + 1) / denominator
    return matrix


class SimplexCodec(nn.Module):
    """Lossless ILR coordinate transform between R^d and product simplexes."""

    def __init__(self, d_model: int, distribution_k: int) -> None:
        super().__init__()
        if d_model % (distribution_k - 1) != 0:
            raise ValueError("d_model must be divisible by distribution_k - 1")
        self.d_model = d_model
        self.distribution_k = distribution_k
        self.n_groups = d_model // (distribution_k - 1)
        self.register_buffer(
            "contrast", helmert_contrast(distribution_k), persistent=True
        )

    def inverse_ilr(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Map [..., d_model] Euclidean coordinates to log probabilities."""
        if coordinates.shape[-1] != self.d_model:
            raise ValueError(
                f"expected last dimension {self.d_model}, got {coordinates.shape[-1]}"
            )
        grouped = coordinates.reshape(
            *coordinates.shape[:-1], self.n_groups, self.distribution_k - 1
        )
        contrast = self.contrast.to(device=coordinates.device, dtype=coordinates.dtype)
        logits = grouped @ contrast
        return F.log_softmax(logits, dim=-1)

    def ilr(self, log_probabilities: torch.Tensor) -> torch.Tensor:
        """Map normalized or unnormalized log masses back to [..., d_model]."""
        expected = (self.n_groups, self.distribution_k)
        if log_probabilities.shape[-2:] != expected:
            raise ValueError(
                f"expected trailing dimensions {expected}, "
                f"got {tuple(log_probabilities.shape[-2:])}"
            )
        contrast = self.contrast.to(
            device=log_probabilities.device, dtype=log_probabilities.dtype
        )
        coordinates = log_probabilities @ contrast.transpose(0, 1)
        return coordinates.flatten(start_dim=-2)


class DistributionAggregator(nn.Module):
    """Aggregate selected expert product distributions and return ILR vectors."""

    def __init__(
        self,
        d_model: int,
        distribution_k: int,
        method: str,
        power_rho: float = 0.5,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 8,
    ) -> None:
        super().__init__()
        self.codec = SimplexCodec(d_model, distribution_k)
        self.method = method
        self.power_rho = power_rho
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        centers = torch.linspace(-1.0, 1.0, distribution_k)
        cost = (centers[:, None] - centers[None, :]).square()
        self.register_buffer("transport_cost", cost, persistent=True)

    def _rho(self) -> float:
        if self.method == "geometric":
            return 0.0
        if self.method == "hellinger":
            return 0.5
        if self.method == "arithmetic":
            return 1.0
        return self.power_rho

    def _sinkhorn_barycenter(
        self, probabilities: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Entropic discrete Wasserstein barycenter by iterative scaling."""
        dtype = probabilities.dtype
        probabilities = probabilities.float().clamp_min(1e-12)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        weights = weights.float()
        kernel = torch.exp(
            -self.transport_cost.float() / self.sinkhorn_epsilon
        ).clamp_min(1e-20)
        scaling = torch.ones_like(probabilities)
        barycenter = probabilities.mean(dim=1)
        for _ in range(self.sinkhorn_iterations):
            kernel_v = torch.einsum("ij,negj->negi", kernel, scaling).clamp_min(
                1e-12
            )
            left_scaling = probabilities / kernel_v
            kernel_t_u = torch.einsum(
                "ij,negi->negj", kernel, left_scaling
            ).clamp_min(1e-12)
            log_barycenter = (
                weights[:, :, None, None] * kernel_t_u.log()
            ).sum(dim=1)
            barycenter = F.softmax(log_barycenter, dim=-1)
            scaling = barycenter[:, None, :, :] / kernel_t_u
        return barycenter.to(dtype=dtype)

    def forward(
        self, selected_outputs: torch.Tensor, router_weights: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            selected_outputs: [tokens, top_k, d_model]
            router_weights: [tokens, top_k], rows sum to one
        """
        if selected_outputs.ndim != 3 or router_weights.ndim != 2:
            raise ValueError("invalid selected output or router weight rank")
        if selected_outputs.shape[:2] != router_weights.shape:
            raise ValueError("selected output and router weight shapes disagree")

        linear_output = (
            selected_outputs * router_weights.to(selected_outputs.dtype)[..., None]
        ).sum(dim=1)
        log_probabilities = self.codec.inverse_ilr(selected_outputs).float()
        normalized_weights = router_weights.float()
        normalized_weights = normalized_weights / normalized_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        if selected_outputs.shape[1] == 1 or self.method == "geometric":
            pooled_output = linear_output
            pooled_log_prob = self.codec.inverse_ilr(pooled_output).float()
        elif self.method == "wasserstein":
            pooled_probability = self._sinkhorn_barycenter(
                log_probabilities.exp(), normalized_weights
            )
            pooled_log_prob = pooled_probability.float().clamp_min(1e-12).log()
            pooled_output = self.codec.ilr(pooled_log_prob).to(selected_outputs.dtype)
        else:
            rho = self._rho()
            if abs(rho) < 1e-7:
                pooled_output = linear_output
                pooled_log_prob = self.codec.inverse_ilr(pooled_output).float()
            else:
                log_weight = normalized_weights.clamp_min(1e-12).log()
                pooled_log_mass = torch.logsumexp(
                    log_weight[:, :, None, None] + rho * log_probabilities,
                    dim=1,
                ) / rho
                pooled_output = self.codec.ilr(pooled_log_mass).to(
                    selected_outputs.dtype
                )
                pooled_log_prob = pooled_log_mass - torch.logsumexp(
                    pooled_log_mass, dim=-1, keepdim=True
                )

        normalized_log_prob = pooled_log_prob - torch.logsumexp(
            pooled_log_prob, dim=-1, keepdim=True
        )
        probability = normalized_log_prob.exp()
        entropy = -(probability * normalized_log_prob).sum(dim=-1).mean()
        correction = (pooled_output.float() - linear_output.float()).norm(dim=-1)
        baseline_norm = linear_output.float().norm(dim=-1).clamp_min(1e-6)
        correction_ratio = (correction / baseline_norm).mean()
        return pooled_output, {
            "distribution_entropy": entropy,
            "nonlinear_correction_ratio": correction_ratio,
        }

