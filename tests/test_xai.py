import unittest
import numpy as np
from src.explainability.feature_importance import extract_top_features


class TestXAIModule(unittest.TestCase):

    def test_extract_top_features(self):
        feat_names = ["rsi_14", "macd", "bb_pct_b", "volume_ratio"]
        shap_vals = np.array([0.15, -0.45, 0.05, -0.02])
        raw_vals = np.array([65.0, 1.2, 0.8, 1.5])

        top = extract_top_features(feat_names, shap_vals, raw_vals, top_k=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0]["feature"], "macd") # highest abs shap (0.45)
        self.assertEqual(top[1]["feature"], "rsi_14") # second highest abs shap (0.15)


if __name__ == "__main__":
    unittest.main()
