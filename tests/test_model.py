import unittest

import torch

from dmoe.config import ModelConfig
from dmoe.model import DecoderLM


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
        for model_type in ("dense", "vanilla_moe", "distributional_moe"):
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


if __name__ == "__main__":
    unittest.main()
