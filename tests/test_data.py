import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dmoe.data import (
    PackedTextBatcher,
    ParquetFileLayout,
    ParquetRowStream,
    ParquetTextCorpus,
    RandomTokenBatcher,
    StreamingPackedTextBatcher,
    sequential_token_batches,
    validate_data_manifest,
)


class _FakeDataset:
    column_names = ["text"]

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, indices: list[int]) -> dict[str, list[str]]:
        return {"text": [self.texts[index] for index in indices]}


class _FakeTokenizer:
    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
        if kwargs["add_special_tokens"] is not False:
            raise AssertionError("online packing must disable automatic special tokens")
        return {
            "input_ids": [
                [3 + (ord(character) - ord("a")) for character in text]
                for text in texts
            ]
        }


class DataTest(unittest.TestCase):
    def test_parquet_corpus_reserves_disjoint_validation_and_test_rows(
        self,
    ) -> None:
        dataset = _FakeDataset([chr(ord("a") + index) for index in range(20)])
        corpus = ParquetTextCorpus(
            dataset=dataset,
            tokenizer=_FakeTokenizer(),
            metadata={},
            text_column="text",
            eos_token_id=2,
            vocab_size=32,
            tokenizer_batch_size=2,
            validation_rows=4,
            test_rows=3,
            validate_token_ids=True,
        )
        train = corpus.train_batcher(
            sequence_length=2, batch_size=1, rank=0, world_size=1
        )
        validation = corpus.validation_batcher(
            sequence_length=2, batch_size=1, rank=0, world_size=1
        )
        test = corpus.test_batcher(
            sequence_length=2, batch_size=1, rank=0, world_size=1
        )
        self.assertEqual(corpus.train_rows, 13)
        self.assertEqual((train.row_start, train.row_stop), (0, 13))
        self.assertEqual((validation.row_start, validation.row_stop), (13, 17))
        self.assertEqual((test.row_start, test.row_stop), (17, 20))

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

    def test_online_packing_and_exact_resume(self) -> None:
        dataset = _FakeDataset(["abc", "defg", "h", "ijk", "lmno"])
        kwargs = {
            "dataset": dataset,
            "tokenizer": _FakeTokenizer(),
            "text_column": "text",
            "eos_token_id": 2,
            "vocab_size": 32,
            "sequence_length": 3,
            "batch_size": 2,
            "tokenizer_batch_size": 2,
            "row_start": 0,
            "row_stop": len(dataset),
            "rank": 0,
            "world_size": 1,
            "repeat": True,
        }
        batcher = PackedTextBatcher(**kwargs)
        inputs, labels = batcher.next_batch()
        self.assertEqual(tuple(inputs.shape), (2, 3))
        self.assertTrue((inputs.reshape(-1)[1:] == labels.reshape(-1)[:-1]).all())

        state = batcher.state_dict()
        expected_inputs, expected_labels = batcher.next_batch()
        resumed = PackedTextBatcher(**kwargs)
        resumed.load_state_dict(state)
        actual_inputs, actual_labels = resumed.next_batch()
        self.assertTrue((expected_inputs == actual_inputs).all())
        self.assertTrue((expected_labels == actual_labels).all())

    def test_direct_parquet_stream_is_bounded_and_resumable(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is an optional data dependency")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = [
                ["a", "bb", "ccc", "dddd", "eeeee"],
                ["f", "gg", "hhh", "iiii", "jjjjj", "kkkkkk"],
            ]
            layouts = []
            for index, texts in enumerate(documents):
                path = root / f"{index}.parquet"
                pq.write_table(
                    pa.table({"text": texts}),
                    path,
                    row_group_size=2,
                )
                parquet = pq.ParquetFile(path)
                layouts.append(
                    ParquetFileLayout(
                        path=str(path),
                        row_count=parquet.metadata.num_rows,
                        row_group_rows=tuple(
                            parquet.metadata.row_group(group).num_rows
                            for group in range(parquet.metadata.num_row_groups)
                        ),
                    )
                )

            stream = ParquetRowStream(
                layouts=layouts,
                text_column="text",
                row_start=2,
                row_stop=10,
                read_batch_size=2,
                repeat=False,
            )
            self.assertEqual(stream.next_texts(4), ["ccc", "dddd", "eeeee", "f"])
            state = stream.state_dict()
            expected = stream.next_texts(3)

            resumed = ParquetRowStream(
                layouts=layouts,
                text_column="text",
                row_start=2,
                row_stop=10,
                read_batch_size=2,
                repeat=False,
            )
            resumed.load_state_dict(state)
            self.assertEqual(resumed.next_texts(3), expected)
            self.assertEqual(expected, ["gg", "hhh", "iiii"])

            batcher = StreamingPackedTextBatcher(
                source=ParquetRowStream(
                    layouts=layouts,
                    text_column="text",
                    row_start=0,
                    row_stop=11,
                    read_batch_size=2,
                    repeat=True,
                ),
                tokenizer=_FakeTokenizer(),
                eos_token_id=2,
                vocab_size=32,
                sequence_length=3,
                batch_size=2,
                tokenizer_batch_size=2,
            )
            batcher.next_batch()
            batcher_state = batcher.state_dict()
            expected_batch = batcher.next_batch()
            resumed_batcher = StreamingPackedTextBatcher(
                source=ParquetRowStream(
                    layouts=layouts,
                    text_column="text",
                    row_start=0,
                    row_stop=11,
                    read_batch_size=2,
                    repeat=True,
                ),
                tokenizer=_FakeTokenizer(),
                eos_token_id=2,
                vocab_size=32,
                sequence_length=3,
                batch_size=2,
                tokenizer_batch_size=2,
            )
            resumed_batcher.load_state_dict(batcher_state)
            actual_batch = resumed_batcher.next_batch()
            self.assertTrue((expected_batch[0] == actual_batch[0]).all())
            self.assertTrue((expected_batch[1] == actual_batch[1]).all())


if __name__ == "__main__":
    unittest.main()
