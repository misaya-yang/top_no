import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
    from samplers import apply_truncation, batch_generate
except ModuleNotFoundError:
    torch = None
    apply_truncation = None
    batch_generate = None


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class SamplerTests(unittest.TestCase):
    def test_top_p_keeps_crossing_token(self):
        probs = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32)
        logits = probs.log()

        truncated = apply_truncation(logits, "top_p", p=0.8)

        self.assertTrue(torch.isfinite(truncated[0, 0]))
        self.assertTrue(torch.isfinite(truncated[0, 1]))
        self.assertFalse(torch.isfinite(truncated[0, 2]))

    def test_top_p_always_keeps_top_token(self):
        logits = torch.tensor([[10.0, 9.0, 8.0]], dtype=torch.float32)

        truncated = apply_truncation(logits, "top_p", p=0.01)

        self.assertTrue(torch.isfinite(truncated[0, 0]))
        self.assertFalse(torch.isfinite(truncated[0, 1]))
        self.assertFalse(torch.isfinite(truncated[0, 2]))

    def test_nu_margin_is_larger_for_lower_frequency_tokens(self):
        logits = torch.tensor([[10.0, 2.0, 2.0]], dtype=torch.float32)
        freqs = torch.tensor([100.0, 0.0, 99.0], dtype=torch.float32)

        truncated = apply_truncation(
            logits,
            "nu",
            token_freq_table=freqs,
            kappa=10.0,
            m0=3.0,
        )

        low_freq_margin = 3.0 + 10.0 / math.sqrt(0.0 + 1.0)
        high_freq_margin = 3.0 + 10.0 / math.sqrt(99.0 + 1.0)
        self.assertGreater(low_freq_margin, high_freq_margin)
        self.assertTrue(torch.isfinite(truncated[0, 1]))
        self.assertFalse(torch.isfinite(truncated[0, 2]))

    def test_fixed_margin_matches_min_p_threshold(self):
        logits = torch.tensor([[5.0, 2.0, 1.0]], dtype=torch.float32)

        min_p = apply_truncation(logits, "min_p", p_min=math.exp(-3.0))
        fixed = apply_truncation(logits, "fixed_margin", margin=3.0)

        self.assertEqual(torch.isfinite(min_p).tolist(), torch.isfinite(fixed).tolist())

    def test_nu_kappa_zero_matches_fixed_margin(self):
        logits = torch.tensor([[5.0, 2.0, 1.0]], dtype=torch.float32)
        freqs = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        nu = apply_truncation(logits, "nu", token_freq_table=freqs, kappa=0.0, m0=3.0)
        fixed = apply_truncation(logits, "fixed_margin", margin=3.0)

        self.assertEqual(torch.isfinite(nu).tolist(), torch.isfinite(fixed).tolist())

    def test_top_k_keeps_requested_number_of_tokens(self):
        logits = torch.tensor([[5.0, 2.0, 3.0, 1.0]], dtype=torch.float32)

        truncated = apply_truncation(logits, "top_k", k=2)

        self.assertEqual(torch.isfinite(truncated).sum().item(), 2)
        self.assertTrue(torch.isfinite(truncated[0, 0]))
        self.assertTrue(torch.isfinite(truncated[0, 2]))

    def test_conformal_nu_uses_nonconformity_threshold(self):
        logits = torch.tensor([[5.0, 2.0, 1.0]], dtype=torch.float32)
        freqs = torch.tensor([100.0, 0.0, 10.0], dtype=torch.float32)

        truncated = apply_truncation(
            logits,
            "conformal_nu",
            token_freq_table=freqs,
            kappa=1.0,
            q_hat=2.0,
            alpha=1.0,
        )

        self.assertEqual(torch.isfinite(truncated).tolist(), [[True, True, False]])

    def test_legacy_nu_variants_require_explicit_flag(self):
        logits = torch.tensor([[5.0, 2.0, 1.0]], dtype=torch.float32)
        freqs = torch.tensor([100.0, 1.0, 1.0], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "deprecated legacy strategy"):
            apply_truncation(logits, "nu_entropy", token_freq_table=freqs)

        truncated = apply_truncation(
            logits,
            "nu_entropy",
            token_freq_table=freqs,
            legacy=True,
        )
        self.assertTrue(torch.isfinite(truncated[0, 0]))

    def test_legacy_mathboost_requires_explicit_flag(self):
        logits = torch.tensor([[5.0, 2.0, 1.0]], dtype=torch.float32)
        freqs = torch.tensor([100.0, 1.0, 1.0], dtype=torch.float32)
        math_freqs = torch.tensor([100.0, 50.0, 50.0], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "deprecated legacy strategy"):
            apply_truncation(
                logits,
                "nu_mathboost",
                token_freq_table=freqs,
                math_freq_table=math_freqs,
            )

        truncated = apply_truncation(
            logits,
            "nu_mathboost",
            token_freq_table=freqs,
            math_freq_table=math_freqs,
            legacy=True,
        )
        self.assertTrue(torch.isfinite(truncated[0, 0]))

    def test_batch_generate_left_padding_position_ids_and_eos_slicing(self):
        class Encoding(dict):
            def to(self, device):
                return Encoding({key: value.to(device) for key, value in self.items()})

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2
            eos_token = "<eos>"
            pad_token = "<pad>"
            padding_side = "right"

            def __call__(self, prompts, **_kwargs):
                rows = [[5, 6], [7, 8, 9]]
                max_len = max(len(row) for row in rows)
                padded, masks = [], []
                for row in rows:
                    pad = [self.pad_token_id] * (max_len - len(row))
                    padded.append(pad + row)
                    masks.append([0] * len(pad) + [1] * len(row))
                return Encoding({
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                })

            def decode(self, ids, skip_special_tokens=True):
                return " ".join(str(i) for i in ids)

        class Output:
            def __init__(self, logits):
                self.logits = logits
                self.past_key_values = "cache"

        class FakeModel:
            def __init__(self):
                self.param = torch.nn.Parameter(torch.zeros(1))
                self.calls = 0
                self.position_ids = []

            def parameters(self):
                yield self.param

            def __call__(self, input_ids, attention_mask, position_ids, **_kwargs):
                self.position_ids.append(position_ids.detach().cpu())
                logits = torch.full((input_ids.shape[0], input_ids.shape[1], 10), -100.0)
                token = 4 if self.calls == 0 else 2
                logits[:, -1, token] = 100.0
                self.calls += 1
                return Output(logits)

        model = FakeModel()
        tokenizer = FakeTokenizer()

        generated = batch_generate(
            model,
            tokenizer,
            ["short", "long"],
            max_new_tokens=3,
            batch_size=2,
            strategy="greedy",
            strategy_kwargs={},
        )

        self.assertEqual([item["tokens"] for item in generated], [[4], [4]])
        self.assertEqual([item["stopped_eos"] for item in generated], [True, True])
        self.assertEqual(model.position_ids[0].tolist(), [[0, 0, 1], [0, 1, 2]])
        self.assertEqual(model.position_ids[1].tolist(), [[2], [3]])


if __name__ == "__main__":
    unittest.main()
