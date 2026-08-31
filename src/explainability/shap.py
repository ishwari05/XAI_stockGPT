from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import shap


class BaseExplainer(ABC):
    """Abstract interface for model explanations."""

    @abstractmethod
    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """Compute attribution values (e.g. SHAP values)."""
        pass


class ShapExplainer(BaseExplainer):
    """
    Reusable SHAP explainer supporting any tree ensemble model (XGBoost, LightGBM, Random Forest)
    or kernel-based model without model-specific scripts.
    """

    @staticmethod
    def normalize_explainer_type(explainer_type: str) -> str:
        mapping = {
            "tree": "tree",
            "tree_shap": "tree",
            "kernel": "kernel",
            "kernel_shap": "kernel",
        }
        return mapping.get(explainer_type, explainer_type)

    def __init__(self, model: Any, explainer_type: str = "tree"):
        self.explainer_type = self.normalize_explainer_type(explainer_type)
        self.model = model
        if self.explainer_type == "tree":
            # For tree ensembles (XGBoost, LightGBM, RF)
            if hasattr(model, "model"):
                raw_model = model.model
            else:
                raw_model = model
            self.explainer = shap.TreeExplainer(raw_model)

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        if self.explainer_type == "tree":
            shap_values = self.explainer.shap_values(X)
        else:
            model_func = self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict
            # Use KernelExplainer with a k-means summary of X as the background to speed it up
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = shap.KernelExplainer(model_func, shap.kmeans(X, 10))
                shap_values = explainer.shap_values(X)
                
        return np.array(shap_values)
