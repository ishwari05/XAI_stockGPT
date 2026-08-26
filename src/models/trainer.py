from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import pandas as pd

from .base import BasePredictionModel
from .registry import ModelRegistry
from .evaluator import ModelEvaluator


class ModelTrainer:
    """
    Generic, reusable model training coordinator.
    Loads configurations, features, splits, executes training, evaluates validation metrics,
    and saves model checkpoints and experiment metadata.
    """

    def __init__(self, model_name: str, config: Dict[str, Any], output_dir: Path):
        self.model_name = model_name
        self.config = config
        self.output_dir = output_dir / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ModelEvaluator()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None,
              tickers_test: Optional[pd.Series] = None) -> Tuple[BasePredictionModel, Dict[str, Any]]:
        # Instantiate registered model
        model = ModelRegistry.create(self.model_name, self.config)
        
        # Fit model
        model.fit(X_train, y_train, X_val, y_val)

        # Evaluate on validation split
        val_preds = model.predict(X_val)
        val_proba = model.predict_proba(X_val)
        val_metrics = self.evaluator.evaluate(y_val, val_preds, val_proba)

        summary_metrics = {
            "validation": val_metrics
        }

        # Evaluate on test split if provided
        if X_test is not None and y_test is not None:
            test_preds = model.predict(X_test)
            test_proba = model.predict_proba(X_test)
            test_metrics = self.evaluator.evaluate(y_test, test_preds, test_proba)
            summary_metrics["test"] = test_metrics

            if tickers_test is not None:
                per_ticker_metrics = self.evaluator.evaluate_per_ticker(y_test, test_preds, tickers_test)
                summary_metrics["test_per_ticker"] = per_ticker_metrics

        # Save artifacts
        model.save(self.output_dir)
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(summary_metrics, f, indent=2)

        return model, summary_metrics

