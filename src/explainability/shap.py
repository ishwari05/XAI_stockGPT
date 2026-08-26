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

    def __init__(self, model: Any, explainer_type: str = "tree"):
        self.explainer_type = explainer_type
        if explainer_type == "tree":
            # For tree ensembles (XGBoost, LightGBM, RF)
            if hasattr(model, "model"):
                raw_model = model.model
            else:
                raw_model = model
            self.explainer = shap.TreeExplainer(raw_model)
        else:
            self.explainer = shap.Explainer(model)

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        shap_values = self.explainer.shap_values(X)
        return np.array(shap_values)
