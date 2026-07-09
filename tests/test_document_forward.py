import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from document_store import BoundDocument  # noqa: E402
    from eval_prediction_sets import batch_document_position_logits  # noqa: E402


@unittest.skipIf(torch is None, "torch is not installed")
class DocumentForwardTests(unittest.TestCase):
    class Tokenizer:
        pad_token_id = 999

        def __init__(self, tokens_by_text):
            self.tokens_by_text = tokens_by_text

        def __call__(self, texts, **kwargs):
            self.kwargs = kwargs
            return {"input_ids": [self.tokens_by_text[text] for text in texts]}

    class Model:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, attention_mask, position_ids):
            self.calls.append(
                (
                    input_ids.detach().cpu(),
                    attention_mask.detach().cpu(),
                    position_ids.detach().cpu(),
                )
            )
            logits = torch.zeros(*input_ids.shape, 3)
            logits[..., 0] = input_ids
            logits[..., 1] = position_ids
            return type("Output", (), {"logits": logits})()

    def documents(self):
        return (
            BoundDocument("cal", "doc-a", "0" * 64, "cluster-a", "text-a"),
            BoundDocument("cal", "doc-b", "1" * 64, "cluster-b", "text-b"),
        )

    def test_forward_uses_prefix_only_and_emits_one_target_per_document(self):
        tokens = {"text-a": list(range(17)), "text-b": list(range(10, 31))}
        tokenizer = self.Tokenizer(tokens)
        model = self.Model()

        batches = list(
            batch_document_position_logits(
                model,
                tokenizer,
                self.documents(),
                {"batch_size": 2, "max_length": 64},
                torch.device("cpu"),
                position_salt="cal-salt",
            )
        )

        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(len(batch.selections), 2)
        self.assertEqual({item.doc_id for item in batch.selections}, {"doc-a", "doc-b"})
        self.assertEqual(batch.logits.shape, (2, 3))
        for row, (selection, target) in enumerate(
            zip(batch.selections, batch.targets.tolist())
        ):
            source_tokens = tokens[f"text-{selection.doc_id[-1]}"]
            self.assertEqual(target, source_tokens[selection.target_index])
            self.assertEqual(
                batch.logits[row, 0].item(),
                source_tokens[selection.target_index - 1],
            )
        input_ids, attention_mask, position_ids = model.calls[0]
        self.assertTrue(torch.equal(input_ids[:, -1], batch.logits[:, 0].long()))
        self.assertTrue((attention_mask[:, -1] == 1).all())
        self.assertTrue((position_ids[:, -1] == attention_mask.sum(dim=1) - 1).all())
        self.assertFalse(tokenizer.kwargs["truncation"])
        self.assertFalse(tokenizer.kwargs["add_special_tokens"])

    def test_long_document_uses_tail_context_window_without_target_leakage(self):
        documents = (self.documents()[0],)
        tokens = {"text-a": list(range(100))}
        model = self.Model()

        batch = next(
            batch_document_position_logits(
                model,
                self.Tokenizer(tokens),
                documents,
                {"batch_size": 1, "max_length": 8},
                torch.device("cpu"),
                position_salt="cal-salt",
            )
        )

        selection = batch.selections[0]
        input_ids, _, _ = model.calls[0]
        self.assertEqual(input_ids.shape[1], 8)
        self.assertEqual(input_ids[0, -1].item(), tokens["text-a"][selection.target_index - 1])
        self.assertEqual(batch.targets[0].item(), tokens["text-a"][selection.target_index])
        self.assertNotIn(tokens["text-a"][selection.target_index], input_ids[0].tolist())

    def test_mixed_roles_fail_before_tokenization(self):
        first, second = self.documents()
        mixed = (
            first,
            BoundDocument(
                "test",
                second.doc_id,
                second.content_sha256,
                second.cluster_id,
                second.text,
            ),
        )
        tokenizer = self.Tokenizer({"text-a": list(range(17)), "text-b": list(range(21))})

        with self.assertRaisesRegex(ValueError, "single manifest role"):
            list(
                batch_document_position_logits(
                    self.Model(),
                    tokenizer,
                    mixed,
                    {"batch_size": 2, "max_length": 64},
                    torch.device("cpu"),
                    position_salt="salt",
                )
            )
        self.assertFalse(hasattr(tokenizer, "kwargs"))

    def test_calibration_and_test_manifests_use_independent_forward_calls(self):
        cal_documents = self.documents()
        test_documents = (
            BoundDocument("test", "doc-c", "2" * 64, "cluster-c", "text-c"),
        )
        tokens = {
            "text-a": list(range(17)),
            "text-b": list(range(10, 31)),
            "text-c": list(range(20, 45)),
        }
        model = self.Model()
        tokenizer = self.Tokenizer(tokens)

        calibration = list(
            batch_document_position_logits(
                model,
                tokenizer,
                cal_documents,
                {"batch_size": 2, "max_length": 16},
                torch.device("cpu"),
                position_salt="calibration-salt",
            )
        )
        test = list(
            batch_document_position_logits(
                model,
                tokenizer,
                test_documents,
                {"batch_size": 2, "max_length": 16},
                torch.device("cpu"),
                position_salt="test-salt",
            )
        )

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(
            {item.doc_id for batch in calibration for item in batch.selections},
            {"doc-a", "doc-b"},
        )
        self.assertEqual(
            {item.doc_id for batch in test for item in batch.selections},
            {"doc-c"},
        )

    def test_configured_position_count_is_an_assertion_not_a_truncation(self):
        tokens = {"text-a": list(range(17)), "text-b": list(range(21))}

        with self.assertRaisesRegex(ValueError, "n_calibration.*manifest count"):
            list(
                batch_document_position_logits(
                    self.Model(),
                    self.Tokenizer(tokens),
                    self.documents(),
                    {"batch_size": 2, "max_length": 16, "n_calibration": 1},
                    torch.device("cpu"),
                    position_salt="cal-salt",
                )
            )


if __name__ == "__main__":
    unittest.main()
