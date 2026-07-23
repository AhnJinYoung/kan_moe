import math
import unittest
from pathlib import Path

import torch

from analyze_mechanism import (
    bootstrap_interval,
    disagreement_deciles,
    spearman,
)
from dmoe.config import load_experiment_config
from scripts.experiment_matrix import (
    _confirmation_runs,
    _control_runs,
    _pilot_runs,
    _profiling_runs,
    _scaling_runs,
    _screening_runs,
    _seed_runs,
)


class RevisedProtocolTest(unittest.TestCase):
    def test_all_experiment_configs_validate(self) -> None:
        for path in sorted(Path("configs").glob("*.yaml")):
            config = load_experiment_config(path)
            self.assertGreater(config.model.vocab_size, 0, path)
            if config.data.input_format == "parquet_text":
                self.assertGreater(config.data.validation_rows, 0, path)
                self.assertGreater(config.data.test_rows, 0, path)

    def test_experiment_matrix_names_and_outputs_are_unique(self) -> None:
        for build_stage in (
            _pilot_runs,
            _profiling_runs,
            _screening_runs,
            _confirmation_runs,
            _control_runs,
            _scaling_runs,
            _seed_runs,
        ):
            runs = build_stage()
            names = [run.name for run in runs]
            self.assertEqual(len(names), len(set(names)))
            for run in runs:
                self.assertTrue(Path(run.config).is_file(), run.config)

    def test_scale_configs_use_ten_tokens_per_parameter_with_step_headroom(
        self,
    ) -> None:
        budgets = {
            "150m": 1_500_000_000,
            "500m": 5_000_000_000,
            "1_5b": 15_000_000_000,
        }
        for path in sorted(Path("configs").glob("*.yaml")):
            scale = next(
                (name for name in budgets if name in path.stem), None
            )
            if scale is None:
                continue
            config = load_experiment_config(path)
            self.assertEqual(config.train.max_tokens, budgets[scale], path)
            tokens_per_step = (
                config.train.sequence_length
                * config.train.micro_batch_size
                * config.train.gradient_accumulation_steps
            )
            required_steps = math.ceil(
                config.train.max_tokens / tokens_per_step
            )
            self.assertGreaterEqual(
                config.train.max_steps, required_steps, path
            )

    def test_early_screen_is_500m_and_excludes_learned_controls(self) -> None:
        runs = _screening_runs()
        self.assertTrue(all("500m" in run.config for run in runs))
        self.assertFalse(
            any(
                "output_gated" in run.config
                or "residual_mlp" in run.config
                or "learned_rho" in run.config
                for run in runs
            )
        )

    def test_late_controls_include_dense_and_confirmation_can_reuse_baseline(
        self,
    ) -> None:
        controls = _control_runs()
        self.assertTrue(any("dense_500m" in run.config for run in controls))
        candidate = _confirmation_runs(role="candidate")
        self.assertEqual(len(candidate), 1)
        self.assertIn("distributional_moe_500m", candidate[0].config)

    def test_mechanism_statistics_have_expected_direction(self) -> None:
        disagreement = torch.arange(100, dtype=torch.float32)
        gain = disagreement * 0.01
        correction = torch.ones_like(gain)
        self.assertAlmostEqual(spearman(disagreement, gain), 1.0)
        low, high = bootstrap_interval(gain, iterations=100, seed=3)
        self.assertLess(low, float(gain.mean()))
        self.assertGreater(high, float(gain.mean()))
        deciles = disagreement_deciles(disagreement, gain, correction)
        self.assertEqual(len(deciles), 10)
        self.assertLess(
            deciles[0]["mean_nll_gain"], deciles[-1]["mean_nll_gain"]
        )


if __name__ == "__main__":
    unittest.main()
