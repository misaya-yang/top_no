import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    from eval_reasoning_self_consistency import answers_match, extract_answer, extract_number
except ModuleNotFoundError:
    answers_match = None
    extract_answer = None
    extract_number = None


@unittest.skipIf(extract_number is None, "evaluation dependencies are not installed")
class EvalHelperTests(unittest.TestCase):
    def test_extract_number_handles_latex_fraction(self):
        self.assertAlmostEqual(extract_number(r"The answer is \frac{1}{2}."), 0.5)

    def test_extract_answer_prefers_boxed_answer(self):
        self.assertEqual(extract_answer(r"Therefore \boxed{42}."), "42")

    def test_answers_match_numeric_fraction(self):
        self.assertTrue(answers_match("0.25", r"\frac{1}{4}"))


if __name__ == "__main__":
    unittest.main()
