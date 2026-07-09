import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from freq_table import (  # noqa: E402
        counts_sha256,
        count_frequency_tokens,
        frequency_exclusion_token_ids,
        load_frequency_table,
        load_frequency_table_metadata,
        make_frequency_table_metadata,
        save_frequency_table,
        special_token_ids,
    )
    from splits import (  # noqa: E402
        DocumentManifest,
        ManifestDocument,
        SourceDocument,
        manifest_sha256,
    )
else:
    counts_sha256 = None


def source_manifest():
    return DocumentManifest(
        protocol_version="icml2027-pr1a",
        role="freq",
        source="fixture",
        documents=(
            ManifestDocument(
                "doc-f",
                hashlib.sha256(b"content-f").hexdigest(),
                "cluster-f",
            ),
        ),
    )


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class FrequencyTableTests(unittest.TestCase):
    def make_metadata(self, counts, exclusions=(1,)):
        return make_frequency_table_metadata(
            counts,
            model_id="fixture/model",
            tokenizer_id="fixture/tokenizer",
            tokenizer_revision="revision-1",
            source_manifest_sha256=manifest_sha256(source_manifest()),
            exclusion_token_ids=exclusions,
            num_documents=1,
            eos_token_id=2,
        )

    def load(self, path, **overrides):
        expected = {
            "expected_model_id": "fixture/model",
            "expected_tokenizer_id": "fixture/tokenizer",
            "expected_tokenizer_revision": "revision-1",
            "expected_vocab_size": 3,
            "expected_exclusion_token_ids": (1,),
            "expected_eos_token_id": 2,
        }
        expected.update(overrides)
        return load_frequency_table(path, **expected)

    def test_round_trip_is_deterministic_and_zeros_exclusions(self):
        counts = torch.tensor([4, 9, 1], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            first_path = save_frequency_table(counts, metadata, Path(tmp))
            second_path = save_frequency_table(counts, metadata, Path(tmp))
            loaded_counts, loaded_metadata = self.load(first_path)

        self.assertEqual(first_path, second_path)
        self.assertEqual(loaded_counts.tolist(), [4, 0, 1])
        self.assertEqual(loaded_metadata, metadata)
        self.assertEqual(metadata.num_tokens, 5)
        self.assertEqual(metadata.counts_sha256, counts_sha256(loaded_counts))

    def test_tampered_counts_fail_hash_validation(self):
        counts = torch.tensor([4, 0, 1], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = save_frequency_table(counts, metadata, Path(tmp))
            payload = json.loads(sidecar.read_text())
            counts_path = sidecar.parent / payload["counts_file"]
            torch.save(torch.tensor([5, 0, 1], dtype=torch.int64), counts_path)

            with self.assertRaisesRegex(ValueError, "counts_sha256"):
                self.load(sidecar)

    def test_tampered_metadata_fails_artifact_identity(self):
        counts = torch.tensor([4, 0, 1], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = save_frequency_table(counts, metadata, Path(tmp))
            payload = json.loads(sidecar.read_text())
            payload["metadata"]["num_documents"] = 2
            sidecar.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "artifact_id"):
                load_frequency_table_metadata(sidecar)

    def test_identity_mismatches_fail_closed(self):
        counts = torch.tensor([4, 0, 1], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = save_frequency_table(counts, metadata, Path(tmp))
            mismatches = [
                ({"expected_model_id": "other/model"}, "model_id"),
                ({"expected_tokenizer_id": "other/tokenizer"}, "tokenizer_id"),
                ({"expected_tokenizer_revision": "other-revision"}, "tokenizer_revision"),
                ({"expected_vocab_size": 4}, "vocab_size"),
                ({"expected_exclusion_token_ids": ()}, "exclusion_token_ids"),
                ({"expected_eos_token_id": 0}, "eos_token_id"),
            ]
            for overrides, message in mismatches:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        self.load(sidecar, **overrides)

    def test_invalid_counts_are_rejected(self):
        invalid = [
            torch.tensor([1, -1, 2], dtype=torch.int64),
            torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32),
            torch.ones((1, 3), dtype=torch.int64),
        ]
        messages = ["non-negative", "integer-valued", "one-dimensional"]

        for counts, message in zip(invalid, messages):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.make_metadata(counts)

    def test_out_of_range_exclusion_is_rejected(self):
        counts = torch.tensor([4, 0, 1], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "exclusion_token_ids"):
            self.make_metadata(counts, exclusions=(3,))

    def test_special_token_ids_are_sorted_unique_integers(self):
        class Tokenizer:
            eos_token_id = 2
            all_special_ids = [2, 0, 2, None, "1"]

        self.assertEqual(special_token_ids(Tokenizer()), (0,))
        self.assertEqual(frequency_exclusion_token_ids(Tokenizer()), (0,))

    def test_document_counter_adds_one_eos_boundary_per_document(self):
        class Tokenizer:
            eos_token_id = 5
            all_special_ids = [0, 5]

            @staticmethod
            def encode(text, add_special_tokens, truncation, padding):
                if (add_special_tokens, truncation, padding) != (False, False, False):
                    raise AssertionError("builder must disable implicit special tokens")
                return {"alpha": [1, 0, 2, 5], "beta": [2, 3, 2]}[text]

        documents = (
            SourceDocument("doc-a", "alpha"),
            SourceDocument("doc-b", "beta"),
        )

        counts = count_frequency_tokens(Tokenizer(), documents, vocab_size=6)

        self.assertEqual(counts.tolist(), [0, 1, 3, 1, 0, 2])

    def test_document_counter_rejects_missing_eos_and_out_of_vocab_ids(self):
        class MissingEos:
            eos_token_id = None
            all_special_ids = []

        with self.assertRaisesRegex(ValueError, "eos_token_id"):
            count_frequency_tokens(
                MissingEos(),
                (SourceDocument("doc", "text"),),
                vocab_size=4,
            )

        class BadToken:
            eos_token_id = 3
            all_special_ids = [3]

            @staticmethod
            def encode(_text, add_special_tokens, truncation, padding):
                return [4]

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            count_frequency_tokens(
                BadToken(),
                (SourceDocument("doc", "text"),),
                vocab_size=4,
            )

    def test_metadata_types_are_not_coerced_before_identity_check(self):
        counts = torch.tensor([4, 0, 1], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = save_frequency_table(counts, metadata, Path(tmp))
            payload = json.loads(sidecar.read_text())
            payload["metadata"]["num_documents"] = 1.9
            sidecar.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "num_documents.*integer"):
                load_frequency_table_metadata(sidecar)

            exact_string_metadata = replace(metadata, tokenizer_revision="1")
            sidecar = save_frequency_table(counts, exact_string_metadata, Path(tmp))
            payload = json.loads(sidecar.read_text())
            payload["metadata"]["tokenizer_revision"] = 1
            sidecar.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "tokenizer_revision.*string"):
                load_frequency_table_metadata(sidecar)

    def test_writer_rejects_other_protocol_versions(self):
        counts = torch.tensor([4, 0, 1], dtype=torch.int64)
        metadata = replace(self.make_metadata(counts), protocol_version="other-protocol")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "protocol_version"):
                save_frequency_table(counts, metadata, Path(tmp))

    def test_loader_rejects_serialized_float_tensor(self):
        counts = torch.tensor([4, 0, 1], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = save_frequency_table(counts, metadata, Path(tmp))
            payload = json.loads(sidecar.read_text())
            counts_path = sidecar.parent / payload["counts_file"]
            torch.save(counts.float(), counts_path)
            with self.assertRaisesRegex(ValueError, "serialized counts dtype"):
                self.load(sidecar)

    def test_safe_load_does_not_fallback_after_type_error(self):
        counts = torch.tensor([4, 0, 1], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = save_frequency_table(counts, metadata, Path(tmp))
            with patch(
                "freq_table.torch.load",
                side_effect=TypeError("unsafe fixture"),
            ) as mocked:
                with self.assertRaisesRegex(TypeError, "unsafe fixture"):
                    self.load(sidecar)
            self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
