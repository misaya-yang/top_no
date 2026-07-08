import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
    from conformal import (
        conformal_nu_keep_mask,
        conformal_quantile,
        nu_nonconformity,
    )
except ModuleNotFoundError:
    torch = None
    conformal_quantile = None
    nu_nonconformity = None
    conformal_nu_keep_mask = None


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


if __name__ == "__main__":
    unittest.main()
