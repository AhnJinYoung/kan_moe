import os
import unittest
from unittest.mock import patch

from dmoe.resources import configure_conservative_cpu_runtime, detect_resource_limits


class ResourceLimitTest(unittest.TestCase):
    def test_detected_limits_are_positive_and_conservative(self) -> None:
        limits = detect_resource_limits()
        self.assertGreaterEqual(limits.affinity_cpus, 1)
        self.assertGreaterEqual(limits.effective_cpus, 1)
        self.assertLessEqual(limits.effective_cpus, limits.affinity_cpus)
        self.assertIn(limits.cpu_threads, {1, 2})
        self.assertEqual(limits.data_workers, 1)
        self.assertLessEqual(limits.tokenizer_batch_limit, 4)
        self.assertLessEqual(limits.parquet_batch_limit, 4)

    def test_thread_environment_is_capped(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "128",
                "MKL_NUM_THREADS": "128",
                "RAYON_NUM_THREADS": "128",
                "TOKENIZERS_PARALLELISM": "true",
            },
        ):
            limits = configure_conservative_cpu_runtime()
            self.assertLessEqual(int(os.environ["OMP_NUM_THREADS"]), limits.cpu_threads)
            self.assertLessEqual(int(os.environ["MKL_NUM_THREADS"]), limits.cpu_threads)
            self.assertLessEqual(int(os.environ["RAYON_NUM_THREADS"]), limits.cpu_threads)
            self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")


if __name__ == "__main__":
    unittest.main()
