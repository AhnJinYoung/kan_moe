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

    def test_experiment_matrix_names_and_outputs_are_unique(self) -> None:
        for build_stage in (
            _pilot_runs,
            _profiling_runs,
            _screening_runs,
            _scaling_runs,
            _seed_runs,
        ):
            runs = build_stage()
            names = [run.name for run in runs]
            self.assertEqual(len(names), len(set(names)))
            for run in runs:
                self.assertTrue(Path(run.config).is_file(), run.config)

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
