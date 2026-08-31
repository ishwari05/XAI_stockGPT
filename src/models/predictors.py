import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .base import BasePredictionModel
from .registry import ModelRegistry


@ModelRegistry.register("xgboost")
class XGBoostModel(BasePredictionModel):
    """XGBoost Classifier with tree-shap compatibility."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> "XGBoostModel":
        import xgboost as xgb
        from sklearn.utils.class_weight import compute_sample_weight

        self.feature_names = list(X_train.columns)
        params = self.config.get("hyperparameters", {}).copy()
        training_cfg = self.config.get("training", {})

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        fit_kwargs: Dict[str, Any] = {"eval_set": eval_set, "verbose": False}
        if training_cfg.get("use_class_weights", False):
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / "model.json"))
        with open(path / "features.json", "w") as f:
            json.dump(self.feature_names, f)

    def load(self, path: Path) -> "XGBoostModel":
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path / "model.json"))
        with open(path / "features.json", "r") as f:
            self.feature_names = json.load(f)
        return self


@ModelRegistry.register("lightgbm")
class LightGBMModel(BasePredictionModel):
    """LightGBM Classifier wrapper."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> "LightGBMModel":
        import lightgbm as lgb
        from sklearn.utils.class_weight import compute_sample_weight

        self.feature_names = list(X_train.columns)
        params = self.config.get("hyperparameters", {}).copy()
        training_cfg = self.config.get("training", {})

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if training_cfg.get("use_class_weights", False):
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        with open(path / "features.json", "w") as f:
            json.dump(self.feature_names, f)

    def load(self, path: Path) -> "LightGBMModel":
        import joblib
        self.model = joblib.load(path / "model.joblib")
        with open(path / "features.json", "r") as f:
            self.feature_names = json.load(f)
        return self


@ModelRegistry.register("random_forest")
class RandomForestModel(BasePredictionModel):
    """Scikit-Learn Random Forest Classifier wrapper."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> "RandomForestModel":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils.class_weight import compute_sample_weight

        self.feature_names = list(X_train.columns)
        params = self.config.get("hyperparameters", {}).copy()
        training_cfg = self.config.get("training", {})

        fit_kwargs: Dict[str, Any] = {}
        if training_cfg.get("use_class_weights", False):
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)

        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        with open(path / "features.json", "w") as f:
            json.dump(self.feature_names, f)

    def load(self, path: Path) -> "RandomForestModel":
        import joblib
        self.model = joblib.load(path / "model.joblib")
        with open(path / "features.json", "r") as f:
            self.feature_names = json.load(f)
        return self


@ModelRegistry.register("neural_baseline")
class NeuralNetBaselineModel(BasePredictionModel):
    """Neural Network Baseline (MLP / PyTorch capable) Classifier wrapper."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> "NeuralNetBaselineModel":
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        self.feature_names = list(X_train.columns)
        params = self.config.get("hyperparameters", {}).copy()

        if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

        self.model = make_pipeline(StandardScaler(), MLPClassifier(**params))
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        with open(path / "features.json", "w") as f:
            json.dump(self.feature_names, f)

    def load(self, path: Path) -> "NeuralNetBaselineModel":
        import joblib
        self.model = joblib.load(path / "model.joblib")
        with open(path / "features.json", "r") as f:
            self.feature_names = json.load(f)
        return self

