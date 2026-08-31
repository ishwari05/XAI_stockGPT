import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from ..config.loader import ConfigLoader
from ..models.trainer import ModelTrainer


class TrainingPipeline:
    """
    Generic, configuration-driven training pipeline.
    Runs for ANY registered prediction model (XGBoost, LightGBM, Random Forest, Neural Baseline, etc.)
    without writing model-specific training scripts.
    """

    def __init__(self, model_name: str, config_loader: Optional[ConfigLoader] = None):
        self.model_name = model_name
        self.config_loader = config_loader or ConfigLoader()
        self.model_cfg = self.config_loader.get_model_config(model_name)
        self.features_cfg = self.config_loader.load_yaml("data/features.yaml")
        self.output_dir = self.config_loader.resolve_path("models/prediction")

    def run(self) -> Dict[str, Any]:
        features_dir = self.config_loader.resolve_path("data/features")
        
        train_path = features_dir / "train.parquet"
        val_path = features_dir / "validation.parquet"
        test_path = features_dir / "test.parquet"

        if not (train_path.exists() and val_path.exists() and test_path.exists()):
            raise FileNotFoundError(
                f"Feature parquet files not found in {features_dir}. Make sure Phase 2 feature pipeline ran."
            )

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        target_col = "target_cls_h1"
        if target_col not in train_df.columns:
            # Fallback to any target_ column if target_cls_h1 missing
            target_cols = [c for c in train_df.columns if c.startswith("target_")]
            if not target_cols:
                raise ValueError("No target column found in feature datasets.")
            target_col = target_cols[0]

        non_feature_cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
        target_cols = [c for c in train_df.columns if c.startswith("target_")]
        drop_cols = list(set(non_feature_cols + target_cols))

        feature_cols = [c for c in train_df.columns if c not in drop_cols]

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_val, y_val = val_df[feature_cols], val_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]
        tickers_test = test_df["Ticker"] if "Ticker" in test_df.columns else None

        trainer = ModelTrainer(self.model_name, self.model_cfg, self.output_dir)
        model, metrics = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            tickers_test=tickers_test
        )

        return {
            "status": "success",
            "model": self.model_name,
            "metrics": metrics
        }

