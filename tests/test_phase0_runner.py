import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from document_store import BoundDocument  # noqa: E402
from phase0_reliability import (  # noqa: E402
    _frequency_sidecar,
    consume_document_rows,
    iter_document_logits,
    load_checkpoint,
    save_checkpoint,
)
from phase0_stats import GridSpec  # noqa: E402


class FakeTokenizer:
    pad_token_id = 99

    def __init__(self, tokens_by_text):
        self.tokens_by_text = tokens_by_text

    def __call__(self, texts, **kwargs):
        self.kwargs = kwargs
        rows = [self.tokens_by_text[text][: kwargs["max_length"]] for text in texts]
        width = max(len(row) for row in rows)
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class FakeModel:
    def __call__(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 128)
        logits[..., 0] = input_ids
        logits[..., 1] = attention_mask
        return type("Output", (), {"logits": logits})()


class Phase0RunnerTests(unittest.TestCase):
    def documents(self):
        return (
            BoundDocument("tune", "doc-b", "1" * 64, "cluster-b", "text-b"),
            BoundDocument("tune", "doc-a", "0" * 64, "cluster-a", "text-a"),
        )

    def test_iter_document_logits_uses_causal_previous_position(self):
        tokens = {
            "text-a": list(range(4, 44)),
            "text-b": list(range(50, 90)),
        }
        tokenizer = FakeTokenizer(tokens)

        rows = list(
            iter_document_logits(
                FakeModel(),
                tokenizer,
                self.documents(),
                device=torch.device("cpu"),
                max_length=64,
                min_context=16,
                stride=4,
                batch_size=2,
                excluded_target_ids={24},
            )
        )

        self.assertEqual([row.doc_id for row in rows], ["doc-a", "doc-b"])
        self.assertFalse(tokenizer.kwargs["add_special_tokens"])
        self.assertTrue(tokenizer.kwargs["truncation"])
        for row in rows:
            source = tokens[f"text-{row.doc_id[-1]}"]
            for index, selection in enumerate(row.selections):
                self.assertEqual(row.targets[index].item(), source[selection.target_index])
                self.assertEqual(row.logits[index, 0].item(), source[selection.target_index - 1])
                self.assertEqual(selection.evidence_grade, "E")
        self.assertNotIn(24, rows[0].targets.tolist())

    def test_consumer_stops_gracefully_at_wall_time(self):
        class Clock:
            def __init__(self):
                self.values = iter((0.0, 0.1, 2.0))

            def __call__(self):
                return next(self.values)

        row_type = type("Row", (), {})
        rows = []
        for doc_id in ("a", "b"):
            row = row_type()
            row.doc_id = doc_id
            row.logits = torch.tensor([[3.0, 2.0, 1.0]])
            row.targets = torch.tensor([1])
            rows.append(row)

        result = consume_document_rows(
            rows,
            token_counts=torch.tensor([0, 1, 10], dtype=torch.int64),
            grid=GridSpec.default(),
            excluded_token_ids=set(),
            permutation_seed=17,
            wall_seconds=1.0,
            clock=Clock(),
        )

        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(len(result.document_stats), 1)
        self.assertEqual(result.document_stats[0].doc_id, "a")

    def test_left_padding_is_rejected_before_forward(self):
        tokenizer = FakeTokenizer({"text-a": list(range(20)), "text-b": list(range(20))})
        tokenizer.padding_side = "left"

        with self.assertRaisesRegex(ValueError, "right padding"):
            list(
                iter_document_logits(
                    FakeModel(),
                    tokenizer,
                    self.documents(),
                    device=torch.device("cpu"),
                    max_length=64,
                    min_context=16,
                    stride=4,
                    batch_size=2,
                    excluded_target_ids=set(),
                )
            )

    def test_checkpoint_round_trip_binds_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            payload = {
                "schema_version": "icml2027-phase0-checkpoint-v1",
                "identity_sha256": "a" * 64,
                "processed_doc_ids": ["doc-a"],
            }

            save_checkpoint(path, payload)

            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            self.assertEqual(load_checkpoint(path, expected_identity="a" * 64), payload)
            with self.assertRaisesRegex(ValueError, "identity"):
                load_checkpoint(path, expected_identity="b" * 64)

    def test_frequency_sidecar_accepts_one_sha256_named_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar_dir = root / "frequency"
            sidecar_dir.mkdir()
            sidecar = sidecar_dir / ("a" * 64 + ".json")
            sidecar.write_text("{}", encoding="utf-8")

            self.assertEqual(
                _frequency_sidecar(root, "frequency"),
                sidecar.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
