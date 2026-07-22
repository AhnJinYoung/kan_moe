import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dmoe.data import (
    RandomTokenBatcher,
    sequential_token_batches,
    validate_data_manifest,
)


class DataTest(unittest.TestCase):
    def test_manifest_contract_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "dtype": "uint16",
                        "tokenizer": {"vocab_size": 32_768, "eos_token_id": 2},
                    }
                ),
                encoding="utf-8",
            )
            validate_data_manifest(
                str(path),
                vocab_size=32_768,
                eos_token_id=2,
                binary_dtype="uint16",
            )
            with self.assertRaisesRegex(ValueError, "vocab_size"):
                validate_data_manifest(
                    str(path),
                    vocab_size=32_000,
                    eos_token_id=2,
                    binary_dtype="uint16",
                )

    def test_random_and_sequential_binary_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = (np.arange(257, dtype=np.uint16) % 101).astype(np.uint16)
            tokens.tofile(root / "tokens.bin")
            batcher = RandomTokenBatcher(
                path=str(root),
                patterns="*.bin",
                binary_dtype="uint16",
                sequence_length=16,
                batch_size=3,
                seed=9,
                vocab_size=101,
            )
            inputs, labels = batcher.next_batch()
            self.assertEqual(tuple(inputs.shape), (3, 16))
            self.assertTrue((inputs[:, 1:] == labels[:, :-1]).all())

            batches = list(
                sequential_token_batches(
                    path=str(root),
                    patterns="*.bin",
                    binary_dtype="uint16",
                    sequence_length=16,
                    batch_size=4,
                    max_batches=2,
                )
            )
            self.assertEqual(len(batches), 2)
            self.assertEqual(tuple(batches[0][0].shape), (4, 16))


if __name__ == "__main__":
    unittest.main()
