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
        learnable_rho: bool = False,
        rho_limit: float = 2.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 8,
    ) -> None:
        super().__init__()
        self.codec = SimplexCodec(d_model, distribution_k)
        self.method = method
        self.power_rho = power_rho
        self.learnable_rho = learnable_rho
        self.rho_limit = rho_limit
        if learnable_rho:
            self.rho_parameter = nn.Parameter(
                torch.tensor(float(power_rho), dtype=torch.float32)
            )
        else:
            self.register_parameter("rho_parameter", None)
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.collect_token_metrics = False
        self.last_token_metrics: dict[str, torch.Tensor] = {}
        centers = torch.linspace(-1.0, 1.0, distribution_k)
        cost = (centers[:, None] - centers[None, :]).square()
        self.register_buffer("transport_cost", cost, persistent=True)

    def set_collect_token_metrics(self, enabled: bool) -> None:
        self.collect_token_metrics = enabled
        if not enabled:
            self.last_token_metrics = {}

    def reported_rho(self) -> float:
        if self.method == "geometric":
            return 0.0
        if self.learnable_rho:
            if self.rho_parameter is None:
                raise RuntimeError("learnable rho parameter is missing")
            return float(
                self.rho_parameter.detach().clamp(
                    -self.rho_limit, self.rho_limit
                )
            )
        return float(
            {
                "hellinger": 0.5,
                "arithmetic": 1.0,
                "power": self.power_rho,
            }.get(self.method, 0.0)
        )

    def _rho(self, reference: torch.Tensor) -> torch.Tensor:
        if self.method == "geometric":
            return reference.new_zeros((), dtype=torch.float32)
        if self.learnable_rho:
            if self.rho_parameter is None:
                raise RuntimeError("learnable rho parameter is missing")
            return self.rho_parameter.clamp(-self.rho_limit, self.rho_limit).to(
                device=reference.device
            )
        value = {
            "hellinger": 0.5,
            "arithmetic": 1.0,
            "power": self.power_rho,
        }.get(self.method, 0.0)
        return reference.new_tensor(value, dtype=torch.float32)

    @staticmethod
    def _power_log_pool(
        log_probabilities: torch.Tensor,
        normalized_weights: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Stable power pool, including a differentiable limit at rho=0."""
        log_weight = normalized_weights.clamp_min(1e-12).log()
        if abs(float(rho.detach())) >= 1e-3:
            return torch.logsumexp(
                log_weight[:, :, None, None] + rho * log_probabilities,
                dim=1,
            ) / rho

        # log E[exp(rho X)] / rho is the cumulant-generating function
        # divided by rho. The expansion lets a learned rho cross zero while
        # preserving gradients with respect to rho.
        weights = normalized_weights[:, :, None, None]
        mean = (weights * log_probabilities).sum(dim=1)
        centered = log_probabilities - mean[:, None, :, :]
        variance = (weights * centered.square()).sum(dim=1)
        third_cumulant = (weights * centered.pow(3)).sum(dim=1)
        return (
            mean
            + 0.5 * rho * variance
            + (rho.square() / 6.0) * third_cumulant
        )

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
            rho = self._rho(log_probabilities)
            if not self.learnable_rho and abs(float(rho)) < 1e-7:
                pooled_output = linear_output
                pooled_log_prob = self.codec.inverse_ilr(pooled_output).float()
            else:
                pooled_log_mass = self._power_log_pool(
                    log_probabilities, normalized_weights, rho
                )
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
        token_correction_ratio = correction / baseline_norm
        correction_ratio = token_correction_ratio.mean()
        effective_rho = self._rho(log_probabilities)
        expert_js_divergence = selected_outputs.new_zeros((), dtype=torch.float32)
        if self.collect_token_metrics:
            log_mixture = torch.logsumexp(
                normalized_weights.clamp_min(1e-12).log()[:, :, None, None]
                + log_probabilities,
                dim=1,
            )
            expert_kl = (
                log_probabilities.exp()
                * (log_probabilities - log_mixture[:, None, :, :])
            ).sum(dim=-1).mean(dim=-1)
            token_js = (normalized_weights * expert_kl).sum(dim=1).clamp_min(0.0)
            expert_js_divergence = token_js.mean()
            self.last_token_metrics = {
                "expert_js_divergence": token_js.detach(),
                "nonlinear_correction_ratio": token_correction_ratio.detach(),
            }
        else:
            self.last_token_metrics = {}
        return pooled_output, {
            "distribution_entropy": entropy,
            "nonlinear_correction_ratio": correction_ratio,
            "aggregation_rho": effective_rho,
            "expert_js_divergence": expert_js_divergence,
        }
