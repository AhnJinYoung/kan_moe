import unittest

import torch

from dmoe.config import ModelConfig
from dmoe.model import DecoderLM
from dmoe.moe import OutputGatedAggregator, ResidualMLPAggregator


def tiny_config(model_type: str, aggregation: str = "hellinger") -> ModelConfig:
    return ModelConfig(
        model_type=model_type,
        vocab_size=101,
        max_seq_len=32,
        n_layers=2,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        n_experts=4,
        top_k=2,
        moe_layers=[] if model_type == "dense" else [1],
        distribution_k=5,
        aggregation=aggregation,
        gradient_checkpointing=False,
    )


class ModelTest(unittest.TestCase):
    def test_all_variants_forward_backward(self) -> None:
        for model_type in (
            "dense",
            "vanilla_moe",
            "distributional_moe",
            "output_gated_moe",
            "residual_mlp_moe",
        ):
            torch.manual_seed(3)
            model = DecoderLM(tiny_config(model_type))
            inputs = torch.randint(0, 101, (2, 8))
            labels = torch.randint(0, 101, (2, 8))
            outputs = model(inputs, labels)
            self.assertEqual(outputs["logits"].shape, (2, 8, 101))
            self.assertTrue(torch.isfinite(outputs["loss"]))
            outputs["loss"].backward()

    def test_vanilla_and_geometric_models_match(self) -> None:
        torch.manual_seed(4)
        vanilla = DecoderLM(tiny_config("vanilla_moe"))
        distributional = DecoderLM(
            tiny_config("distributional_moe", aggregation="geometric")
        )
        distributional.load_state_dict(vanilla.state_dict(), strict=False)
        inputs = torch.randint(0, 101, (2, 8))
        vanilla.eval()
        distributional.eval()
        expected = vanilla(inputs)["logits"]
        actual = distributional(inputs)["logits"]
        self.assertTrue(torch.equal(actual, expected))

    def test_runtime_top_k_change(self) -> None:
        model = DecoderLM(tiny_config("distributional_moe"))
        model.set_top_k(4)
        self.assertEqual(model.config.top_k, 4)
        for block in model.blocks:
            if block.is_moe:
                self.assertEqual(block.ffn.top_k, 4)

    def test_activation_checkpointed_backward(self) -> None:
        config = tiny_config("distributional_moe")
        config.gradient_checkpointing = True
        model = DecoderLM(config)
        model.train()
        inputs = torch.randint(0, 101, (2, 8))
        outputs = model(inputs, inputs)
        outputs["loss"].backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_moe_variants_support_bf16_autocast(self) -> None:
        for model_type in (
            "vanilla_moe",
            "distributional_moe",
            "output_gated_moe",
            "residual_mlp_moe",
        ):
            torch.manual_seed(11)
            model = DecoderLM(tiny_config(model_type))
            model.train()
            inputs = torch.randint(0, 101, (2, 8))
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                outputs = model(inputs, inputs)
            self.assertTrue(torch.isfinite(outputs["loss"]), model_type)
            outputs["loss"].backward()
            self.assertTrue(
                all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ),
                model_type,
            )

    def test_500m_parameter_contract_on_meta_device(self) -> None:
        common = dict(
            vocab_size=32_000,
            max_seq_len=2_048,
            n_layers=12,
            d_model=768,
            n_heads=12,
            n_experts=16,
            top_k=2,
            gradient_checkpointing=True,
        )
        with torch.device("meta"):
            dense = DecoderLM(
                ModelConfig(
                    **common,
                    model_type="dense",
                    ffn_dim=16_320,
                    moe_layers=[],
                )
            )
            vanilla = DecoderLM(
                ModelConfig(
                    **common,
                    model_type="vanilla_moe",
                    ffn_dim=1_920,
                    moe_layers=[1, 3, 5, 7, 9, 11],
                )
            )
            distributional = DecoderLM(
                ModelConfig(
                    **common,
                    model_type="distributional_moe",
                    ffn_dim=1_920,
                    moe_layers=[1, 3, 5, 7, 9, 11],
                    distribution_k=9,
                )
            )
        counts = [
            model.parameter_report()["total_parameters"]
            for model in (dense, vanilla, distributional)
        ]
        self.assertEqual(counts[1], counts[2])
        self.assertEqual(counts[0], 504_122_112)
        self.assertEqual(counts[1], 504_195_840)
        self.assertLess((max(counts) - min(counts)) / min(counts), 0.05)

    def test_learned_reducers_are_permutation_invariant(self) -> None:
        torch.manual_seed(12)
        selected = torch.randn(7, 3, 32)
        weights = torch.softmax(torch.randn(7, 3), dim=-1)
        permutation = torch.tensor([2, 0, 1])
        for reducer in (
            OutputGatedAggregator(32),
            ResidualMLPAggregator(32, 8),
        ):
            if isinstance(reducer, ResidualMLPAggregator):
                torch.nn.init.normal_(reducer.down.weight)
            expected, _ = reducer(selected, weights)
            actual, _ = reducer(
                selected[:, permutation], weights[:, permutation]
            )
            self.assertTrue(torch.allclose(actual, expected, atol=2e-6))

    def test_residual_mlp_starts_at_vanilla_and_top_one_is_identity(self) -> None:
        torch.manual_seed(13)
        model = DecoderLM(tiny_config("residual_mlp_moe"))
        reducer = model.blocks[1].ffn.aggregator
        self.assertIsInstance(reducer, ResidualMLPAggregator)
        selected = torch.randn(5, 2, 32)
        weights = torch.softmax(torch.randn(5, 2), dim=-1)
        actual, _ = reducer(selected, weights)
        expected = (selected * weights[..., None]).sum(dim=1)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-7, rtol=0.0))
        torch.nn.init.normal_(reducer.down.weight)
        top_one, _ = reducer(selected[:, :1], torch.ones(5, 1))
        self.assertTrue(torch.equal(top_one, selected[:, 0]))

    def test_output_gate_starts_at_vanilla(self) -> None:
        torch.manual_seed(15)
        model = DecoderLM(tiny_config("output_gated_moe"))
        reducer = model.blocks[1].ffn.aggregator
        self.assertIsInstance(reducer, OutputGatedAggregator)
        selected = torch.randn(5, 2, 32)
        weights = torch.softmax(torch.randn(5, 2), dim=-1)
        actual, _ = reducer(selected, weights)
        expected = (selected * weights[..., None]).sum(dim=1)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-7, rtol=0.0))

    def test_common_parameters_share_exact_initialization_across_reducers(self) -> None:
        models = {
            model_type: DecoderLM(tiny_config(model_type))
            for model_type in (
                "vanilla_moe",
                "distributional_moe",
                "output_gated_moe",
                "residual_mlp_moe",
            )
        }
        reference = dict(models["vanilla_moe"].named_parameters())
        for model_type, model in models.items():
            if model_type == "vanilla_moe":
                continue
            parameters = dict(model.named_parameters())
            for name, expected in reference.items():
                if ".aggregator." in name:
                    continue
                self.assertTrue(
                    torch.equal(parameters[name], expected),
                    f"{model_type}:{name}",
                )

    def test_mechanism_snapshot_exposes_token_metrics_and_router_logits(self) -> None:
        torch.manual_seed(14)
        model = DecoderLM(tiny_config("distributional_moe"))
        model.eval()
        model.set_mechanism_collection(True)
        inputs = torch.randint(0, 101, (2, 8))
        output = model(inputs, inputs)
        snapshot = model.mechanism_snapshot()
        self.assertEqual(snapshot["expert_js_divergence"].shape, (16,))
        self.assertEqual(
            snapshot["nonlinear_correction_ratio"].shape, (16,)
        )
        self.assertEqual(len(snapshot["router_logits"]), 1)
        router_gradient = torch.autograd.grad(
            output["lm_loss"], snapshot["router_logits"]
        )[0]
        self.assertEqual(router_gradient.shape, (16, 4))
        self.assertTrue(torch.isfinite(router_gradient).all())

    def test_parameter_matching_at_three_scales(self) -> None:
        scales = (
            (512, 8, 12, 768, 6_528, [1, 3, 5, 7, 9, 11]),
            (768, 12, 12, 1_920, 16_320, [1, 3, 5, 7, 9, 11]),
            (
                1_024,
                16,
                16,
                3_328,
                28_288,
                [1, 3, 5, 7, 9, 11, 13, 15],
            ),
        )
        for d_model, heads, layers, expert_ffn, dense_ffn, moe_layers in scales:
            common = dict(
                vocab_size=32_000,
                max_seq_len=2_048,
                n_layers=layers,
                d_model=d_model,
                n_heads=heads,
                n_experts=16,
                top_k=2,
            )
            with torch.device("meta"):
                dense = DecoderLM(
                    ModelConfig(
                        **common,
                        model_type="dense",
                        ffn_dim=dense_ffn,
                        moe_layers=[],
                    )
                )
                vanilla = DecoderLM(
                    ModelConfig(
                        **common,
                        model_type="vanilla_moe",
                        ffn_dim=expert_ffn,
                        moe_layers=moe_layers,
                    )
                )
                distributional = DecoderLM(
                    ModelConfig(
                        **common,
                        model_type="distributional_moe",
                        ffn_dim=expert_ffn,
                        moe_layers=moe_layers,
                        distribution_k=9,
                    )
                )
            counts = [
                model.parameter_report()["total_parameters"]
                for model in (dense, vanilla, distributional)
            ]
            self.assertEqual(counts[1], counts[2])
            self.assertLess((max(counts) - min(counts)) / min(counts), 0.001)


if __name__ == "__main__":
    unittest.main()
