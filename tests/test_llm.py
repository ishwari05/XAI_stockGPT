import unittest
from src.llm.evaluator import LLMExplanationEvaluator


class TestLLMModule(unittest.TestCase):

    def test_evaluator_catches_hallucinations(self):
        evaluator = LLMExplanationEvaluator()
        text_hallucinated = "The model suggests BUY because CEO announced positive earnings and revenue beat."
        res = evaluator.evaluate_explanation("BUY", text_hallucinated, ["rsi_14"])
        self.assertTrue(res["hallucination_detected"])
        self.assertIn("ceo", res["hallucinations"])
        self.assertIn("earnings", res["hallucinations"])

    def test_evaluator_passes_clean_explanation(self):
        evaluator = LLMExplanationEvaluator()
        text_clean = "The model recommends BUY due to rsi_14 indicating strong upward momentum without overbought readings."
        res = evaluator.evaluate_explanation("BUY", text_clean, ["rsi_14"])
        self.assertFalse(res["hallucination_detected"])
        self.assertTrue(res["signal_consistent"])
        self.assertEqual(res["shap_grounding_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
