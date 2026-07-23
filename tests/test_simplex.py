import unittest

import torch

from dmoe.simplex import DistributionAggregator, SimplexCodec, helmert_contrast


class SimplexCodecTest(unittest.TestCase):
    def test_helmert_is_orthonormal_and_centered(self) -> None:
        contrast = helmert_contrast(9)
        self.assertTrue(
            torch.allclose(contrast @ contrast.T, torch.eye(8), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(contrast.sum(dim=-1), torch.zeros(8), atol=1e-6)
        )

    def test_round_trip(self) -> None:
        torch.manual_seed(0)
        codec = SimplexCodec(d_model=32, distribution_k=5)
        coordinates = torch.randn(3, 2, 32)
        reconstructed = codec.ilr(codec.inverse_ilr(coordinates))
        self.assertTrue(torch.allclose(coordinates, reconstructed, atol=2e-6))

    def test_geometric_is_linear_with_matching_gradients(self) -> None:
        torch.manual_seed(1)
        selected = torch.randn(7, 3, 32, requires_grad=True)
        weights = torch.softmax(torch.randn(7, 3), dim=-1)
        aggregator = DistributionAggregator(32, 5, method="geometric")
        actual, _ = aggregator(selected, weights)
        expected = (selected * weights[..., None]).sum(dim=1)
        self.assertTrue(torch.equal(actual, expected))
        actual.square().sum().backward()
        actual_gradient = selected.grad.detach().clone()
        selected.grad = None
        expected.square().sum().backward()
        self.assertTrue(torch.equal(actual_gradient, selected.grad))

    def test_top_one_is_identity_for_all_aggregators(self) -> None:
        torch.manual_seed(2)
        selected = torch.randn(5, 1, 32)
        weights = torch.ones(5, 1)
        for method in ("hellinger", "arithmetic", "power", "wasserstein"):
            aggregator = DistributionAggregator(
                32, 5, method=method, power_rho=0.75
            )
            actual, _ = aggregator(selected, weights)
            self.assertTrue(torch.equal(actual, selected[:, 0]), method)

    def test_hellinger_is_finite_and_nonlinear(self) -> None:
        selected = torch.tensor(
            [[[2.0, -1.0, 0.5, -0.3], [-1.0, 1.5, -0.2, 0.7]]]
        )
        weights = torch.tensor([[0.6, 0.4]])
        aggregator = DistributionAggregator(4, 3, method="hellinger")
        actual, metrics = aggregator(selected, weights)
        linear = (selected * weights[..., None]).sum(dim=1)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertFalse(torch.allclose(actual, linear, atol=1e-5))
        self.assertGreater(float(metrics["distribution_entropy"]), 0.0)
        self.assertGreater(float(metrics["nonlinear_correction_ratio"]), 0.0)

    def test_atom_count_sweep_preserves_dimension_and_round_trip(self) -> None:
        torch.manual_seed(5)
        coordinates = torch.randn(2, 768)
        for distribution_k in (5, 9, 17):
            codec = SimplexCodec(
                d_model=768, distribution_k=distribution_k
            )
            self.assertEqual(
                codec.n_groups * (distribution_k - 1), 768
            )
            reconstructed = codec.ilr(codec.inverse_ilr(coordinates))
            self.assertTrue(
                torch.allclose(coordinates, reconstructed, atol=3e-6),
                distribution_k,
            )

    def test_learnable_rho_is_differentiable_through_zero(self) -> None:
        torch.manual_seed(6)
        selected = torch.randn(7, 3, 32, requires_grad=True)
        weights = torch.softmax(torch.randn(7, 3), dim=-1)
        aggregator = DistributionAggregator(
            32,
            5,
            method="power",
            power_rho=0.0,
            learnable_rho=True,
        )
        output, metrics = aggregator(selected, weights)
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(float(metrics["aggregation_rho"]), 0.0)
        self.assertIsNotNone(aggregator.rho_parameter.grad)
        self.assertTrue(torch.isfinite(aggregator.rho_parameter.grad))

    def test_token_js_collection_is_finite_and_bounded(self) -> None:
        torch.manual_seed(7)
        selected = torch.randn(11, 3, 32)
        weights = torch.softmax(torch.randn(11, 3), dim=-1)
        aggregator = DistributionAggregator(32, 5, method="hellinger")
        aggregator.set_collect_token_metrics(True)
        _, metrics = aggregator(selected, weights)
        token_js = aggregator.last_token_metrics["expert_js_divergence"]
        self.assertEqual(token_js.shape, (11,))
        self.assertTrue(torch.isfinite(token_js).all())
        self.assertTrue((token_js >= -1e-7).all())
        self.assertTrue((token_js <= torch.log(torch.tensor(3.0)) + 1e-6).all())
        self.assertAlmostEqual(
            float(metrics["expert_js_divergence"]),
            float(token_js.mean()),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
