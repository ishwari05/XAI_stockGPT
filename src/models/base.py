from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


class BasePredictionModel(ABC):
    """
    Unified abstract interface for all predictive models in XAI-StockGPT v2.
    Ensures any model (XGBoost, LightGBM, Random Forest, Neural, etc.) implements
    standardized fit, predict, predict_proba, save, and load interfaces.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> "BasePredictionModel":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> "BasePredictionModel":
        pass
