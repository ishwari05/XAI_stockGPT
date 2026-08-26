import unittest
import numpy as np
import pandas as pd
from src.models.registry import ModelRegistry
from src.models.predictors import (
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    NeuralNetBaselineModel
)


class TestModelsModule(unittest.TestCase):

    def test_model_registry(self):
        models = ModelRegistry.list_models()
        self.assertIn("xgboost", models)
        self.assertIn("lightgbm", models)
        self.assertIn("random_forest", models)
        self.assertIn("neural_baseline", models)

    def test_model_instantiation_from_registry(self):
        xgb = ModelRegistry.create("xgboost", {"hyperparameters": {"max_depth": 3}})
        self.assertIsInstance(xgb, XGBoostModel)
        nn = ModelRegistry.create("neural_baseline", {"hyperparameters": {"hidden_layer_sizes": [32]}})
        self.assertIsInstance(nn, NeuralNetBaselineModel)

    def test_model_fit_predict_adapters(self):
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(np.random.choice([0, 1, 2], size=50))
        
        # Test Random Forest adapter fit/predict
        rf = ModelRegistry.create("random_forest", {"hyperparameters": {"n_estimators": 5, "random_state": 42}})
        rf.fit(X, y)
        preds = rf.predict(X)
        self.assertEqual(len(preds), 50)
        
        # Test Neural Baseline adapter fit/predict
        nn = ModelRegistry.create("neural_baseline", {"hyperparameters": {"hidden_layer_sizes": [10], "max_iter": 10, "random_state": 42}})
        nn.fit(X, y)
        nn_preds = nn.predict(X)
        self.assertEqual(len(nn_preds), 50)


if __name__ == "__main__":
    unittest.main()

