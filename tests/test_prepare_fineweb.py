import tempfile
import unittest
from pathlib import Path

from prepare_fineweb import enforce_preparation_spec, split_source_files


class PrepareFineWebTest(unittest.TestCase):
    def test_preparation_spec_prevents_mixed_tokenizers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            enforce_preparation_spec(root, {"tokenizer_revision": "first"})
            enforce_preparation_spec(root, {"tokenizer_revision": "first"})
            with self.assertRaisesRegex(ValueError, "specification differs"):
                enforce_preparation_spec(root, {"tokenizer_revision": "second"})

    def test_split_is_file_level_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("002.parquet", "000.parquet", "001.parquet"):
                (root / name).touch()
            train, validation = split_source_files(root, "*.parquet", 1)
            self.assertEqual([path.name for path in train], ["000.parquet", "001.parquet"])
            self.assertEqual([path.name for path in validation], ["002.parquet"])

    def test_split_rejects_overlap_or_empty_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "000.parquet").touch()
            (root / "001.parquet").touch()
            with self.assertRaises(ValueError):
                split_source_files(root, "*.parquet", 0)
            with self.assertRaises(ValueError):
                split_source_files(root, "*.parquet", 2)


if __name__ == "__main__":
    unittest.main()
