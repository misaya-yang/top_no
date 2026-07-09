import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from freq_table import (  # noqa: E402
        counts_sha256,
        load_frequency_table,
        load_frequency_table_metadata,
        make_frequency_table_metadata,
        save_frequency_table,
        special_token_ids,
    )
    from splits import DocumentManifest, ManifestDocument, manifest_sha256  # noqa: E402
else:
    counts_sha256 = None


def source_manifest():
    return DocumentManifest(
        protocol_version="icml2027-pr1a",
        role="freq",
        source="fixture",
        documents=(ManifestDocument("doc-f", "content-f", "cluster-f"),),
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
        )

    def load(self, path, **overrides):
        expected = {
            "expected_model_id": "fixture/model",
            "expected_tokenizer_id": "fixture/tokenizer",
            "expected_vocab_size": 3,
            "expected_exclusion_token_ids": (1,),
        }
        expected.update(overrides)
        return load_frequency_table(path, **expected)

    def test_round_trip_is_deterministic_and_zeros_exclusions(self):
        counts = torch.tensor([4, 9, 7], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            first_path = save_frequency_table(counts, metadata, Path(tmp))
            second_path = save_frequency_table(counts, metadata, Path(tmp))
            loaded_counts, loaded_metadata = self.load(first_path)

        self.assertEqual(first_path, second_path)
        self.assertEqual(loaded_counts.tolist(), [4, 0, 7])
        self.assertEqual(loaded_metadata, metadata)
        self.assertEqual(metadata.num_tokens, 11)
        self.assertEqual(metadata.counts_sha256, counts_sha256(loaded_counts))

    def test_tampered_counts_fail_hash_validation(self):
        counts = torch.tensor([4, 0, 7], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = save_frequency_table(counts, metadata, Path(tmp))
            payload = json.loads(sidecar.read_text())
            counts_path = sidecar.parent / payload["counts_file"]
            torch.save(torch.tensor([5, 0, 7], dtype=torch.int64), counts_path)

            with self.assertRaisesRegex(ValueError, "counts_sha256"):
                self.load(sidecar)

    def test_tampered_metadata_fails_artifact_identity(self):
        counts = torch.tensor([4, 0, 7], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = save_frequency_table(counts, metadata, Path(tmp))
            payload = json.loads(sidecar.read_text())
            payload["metadata"]["num_documents"] = 2
            sidecar.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "artifact_id"):
                load_frequency_table_metadata(sidecar)

    def test_identity_mismatches_fail_closed(self):
        counts = torch.tensor([4, 0, 7], dtype=torch.int64)
        metadata = self.make_metadata(counts)

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = save_frequency_table(counts, metadata, Path(tmp))
            mismatches = [
                ({"expected_model_id": "other/model"}, "model_id"),
                ({"expected_tokenizer_id": "other/tokenizer"}, "tokenizer_id"),
                ({"expected_vocab_size": 4}, "vocab_size"),
                ({"expected_exclusion_token_ids": ()}, "exclusion_token_ids"),
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
        counts = torch.tensor([4, 0, 7], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "exclusion_token_ids"):
            self.make_metadata(counts, exclusions=(3,))

    def test_special_token_ids_are_sorted_unique_integers(self):
        class Tokenizer:
            all_special_ids = [2, 0, 2, None, "1"]

        self.assertEqual(special_token_ids(Tokenizer()), (0, 2))


if __name__ == "__main__":
    unittest.main()
