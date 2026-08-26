import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from src.models.evaluator import ModelEvaluator
from src.models.trainer import ModelTrainer


class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluator_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 2, 2, 0])
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)

    def test_evaluator_per_ticker(self):
        y_true = pd.Series([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        tickers = pd.Series(["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"])
        
        evaluator = ModelEvaluator()
        per_ticker = evaluator.evaluate_per_ticker(y_true, y_pred, tickers)
        
        self.assertIn("AAPL", per_ticker)
        self.assertIn("MSFT", per_ticker)
        self.assertIn("macro_f1", per_ticker["AAPL"])

    def test_trainer_fit_and_save(self):
        X_train = pd.DataFrame(np.random.randn(40, 4), columns=[f"f{i}" for i in range(4)])
        y_train = pd.Series(np.random.choice([0, 1, 2], size=40))
        X_val = pd.DataFrame(np.random.randn(10, 4), columns=[f"f{i}" for i in range(4)])
        y_val = pd.Series(np.random.choice([0, 1, 2], size=10))

        config = {
            "model": {"name": "random_forest"},
            "hyperparameters": {"n_estimators": 5, "random_state": 42}
        }
        trainer = ModelTrainer("random_forest", config, self.output_dir)
        model, summary = trainer.train(X_train, y_train, X_val, y_val)

        self.assertIn("validation", summary)
        self.assertTrue((self.output_dir / "random_forest" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
