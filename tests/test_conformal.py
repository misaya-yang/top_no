import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
    from conformal import (
        aps_scores,
        conformal_nu_keep_mask,
        conformal_quantile,
        descending_order,
        nu_nonconformity,
        raps_nonconformity,
        raps_scores,
    )
except ModuleNotFoundError:
    torch = None
    aps_scores = None
    conformal_quantile = None
    descending_order = None
    nu_nonconformity = None
    conformal_nu_keep_mask = None
    raps_nonconformity = None
    raps_scores = None


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class ConformalTests(unittest.TestCase):
    def test_quantile_uses_conformal_rank(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(conformal_quantile(scores, delta=0.25), 4.0)

    def test_nu_nonconformity_uses_target_frequency(self):
        logits = torch.tensor([[5.0, 2.0, 1.0]])
        target_ids = torch.tensor([1])
        freqs = torch.tensor([100.0, 0.0, 10.0])

        score = nu_nonconformity(logits, target_ids, freqs, kappa=1.0)

        self.assertAlmostEqual(float(score[0]), 2.0)

    def test_conformal_mask_keeps_tokens_below_threshold(self):
        logits = torch.tensor([[5.0, 2.0, 1.0]])
        freqs = torch.tensor([100.0, 0.0, 10.0])

        keep = conformal_nu_keep_mask(logits, freqs, kappa=1.0, q_hat=2.0)

        self.assertEqual(keep.tolist(), [[True, True, False]])

    def test_raps_adds_one_based_rank_penalty_after_k_reg(self):
        logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64))
        order = descending_order(logits)
        uniforms = torch.zeros_like(logits)
        aps = aps_scores(logits, order=order, uniforms=uniforms)

        scores = raps_scores(
            logits,
            order=order,
            uniforms=uniforms,
            lambda_reg=0.2,
            k_reg=2,
        )

        expected_penalty = torch.tensor([[0.0, 0.0, 0.2, 0.4]], dtype=torch.float64)
        self.assertTrue(torch.allclose(scores, aps + expected_penalty))

    def test_aps_promotes_large_vocabulary_low_precision_before_scoring(self):
        vocab_size = 70_000
        logits = torch.zeros((1, vocab_size), dtype=torch.float16)
        order = torch.arange(vocab_size).unsqueeze(0)
        zero_uniforms = torch.zeros_like(logits)
        half_uniforms = torch.full_like(logits, 0.5)

        zero_scores = aps_scores(logits, order=order, uniforms=zero_uniforms)
        half_scores = aps_scores(logits, order=order, uniforms=half_uniforms)

        expected_last_prefix = (vocab_size - 1) / vocab_size
        self.assertEqual(zero_scores.dtype, torch.float32)
        self.assertAlmostEqual(
            float(zero_scores[0, -1]), expected_last_prefix, places=6
        )
        self.assertTrue(torch.all(half_scores > zero_scores))

    def test_raps_lambda_zero_is_exactly_aps_after_dtype_alignment(self):
        torch.manual_seed(41)
        logits = torch.randn(3, 17, dtype=torch.float16)
        order = descending_order(logits)
        uniforms = torch.rand_like(logits)
        aps = aps_scores(logits, order=order, uniforms=uniforms)

        raps = raps_scores(
            logits,
            order=order,
            uniforms=uniforms,
            lambda_reg=0.0,
            k_reg=5,
        )

        self.assertEqual(raps.dtype, torch.float32)
        self.assertTrue(torch.equal(raps, aps.float()))

    def test_raps_large_vocabulary_rank_penalty_does_not_collapse_in_fp16(self):
        vocab_size = 70_000
        logits = torch.zeros((1, vocab_size), dtype=torch.float16)
        order = torch.arange(vocab_size).unsqueeze(0)
        uniforms = torch.zeros_like(logits)

        scores = raps_scores(
            logits,
            order=order,
            uniforms=uniforms,
            lambda_reg=1.0,
            k_reg=1,
        )

        self.assertEqual(scores.dtype, torch.float32)
        self.assertEqual(float(scores[0, -1] - scores[0, -2]), 1.0)

    def test_raps_target_scores_are_gathered_from_candidate_scores(self):
        logits = torch.tensor([[3.0, 1.0, 2.0], [0.0, 2.0, 1.0]])
        targets = torch.tensor([2, 0])
        order = descending_order(logits)
        uniforms = torch.full_like(logits, 0.25)
        full = raps_scores(
            logits,
            order=order,
            uniforms=uniforms,
            lambda_reg=0.1,
            k_reg=1,
        )

        target = raps_nonconformity(
            logits,
            targets,
            order=order,
            uniforms=uniforms,
            lambda_reg=0.1,
            k_reg=1,
        )

        expected = full.gather(-1, targets[:, None]).squeeze(-1)
        self.assertTrue(torch.equal(target, expected))

    def test_raps_rejects_invalid_regularization_parameters(self):
        logits = torch.zeros((1, 3))
        order = descending_order(logits)
        uniforms = torch.zeros_like(logits)
        invalid = (
            {"lambda_reg": -0.1, "k_reg": 1},
            {"lambda_reg": float("nan"), "k_reg": 1},
            {"lambda_reg": True, "k_reg": 1},
            {"lambda_reg": 0.1, "k_reg": 0},
            {"lambda_reg": 0.1, "k_reg": 1.5},
            {"lambda_reg": 0.1, "k_reg": True},
        )
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ValueError):
                raps_scores(
                    logits,
                    order=order,
                    uniforms=uniforms,
                    **params,
                )


if __name__ == "__main__":
    unittest.main()
