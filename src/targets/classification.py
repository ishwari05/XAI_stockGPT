from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd


class BaseTargetGenerator(ABC):
    """Abstract interface for target generation."""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        pass


class ClassificationTargetGenerator(BaseTargetGenerator):
    """
    Symmetric or asymmetric forward-return threshold target generator:
    future_return = Close[t+horizon] / Close[t] - 1.0

    BUY (2)  : future_return >= buy_threshold
    SELL (0) : future_return <= sell_threshold
    HOLD (1) : otherwise

    Processes each symbol independently to prevent cross-ticker leakage.
    Explicitly drops trailing unobservable forward rows.
    """

    def __init__(
        self,
        horizon: int = 1,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
        class_labels: Optional[Dict[int, str]] = None,
        close_col: str = "Close",
        symbol_col: str = "Symbol",
        date_col: str = "Date"
    ):
        self.horizon = int(horizon)
        self.buy_threshold = float(buy_threshold)
        self.sell_threshold = float(sell_threshold)
        self.class_labels = class_labels or {0: "SELL", 1: "HOLD", 2: "BUY"}
        self.close_col = close_col
        self.symbol_col = symbol_col
        self.date_col = date_col

    def _generate_for_symbol(self, df_symbol: pd.DataFrame) -> pd.DataFrame:
        out = df_symbol.copy()
        out = out.sort_values(self.date_col).reset_index(drop=True)

        close = out[self.close_col]
        # Shift -horizon looks forward in time ONLY for target calculation
        future_return = close.shift(-self.horizon) / close - 1.0
        
        conditions = [
            future_return >= self.buy_threshold,
            future_return <= self.sell_threshold,
        ]
        choices = [2, 0]  # BUY=2, SELL=0
        out["target_classification"] = np.select(conditions, choices, default=1)  # HOLD=1
        out["target_label"] = out["target_classification"].map(self.class_labels)
        
        # Mark trailing unobservable horizon rows as NaN
        out.loc[future_return.isna(), ["target_classification", "target_label"]] = np.nan
        return out

    def generate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Generates classification targets across single or multiple tickers.
        Drops trailing rows per ticker where the forward return cannot be calculated.
        """
        if df.empty:
            return df, ["target_classification"], {"distribution": {}}

        if self.symbol_col in df.columns:
            symbols = df[self.symbol_col].unique()
            dfs = []
            for sym in symbols:
                sub = df[df[self.symbol_col] == sym].copy()
                sub_target = self._generate_for_symbol(sub)
                dfs.append(sub_target)
            combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            combined = self._generate_for_symbol(df)

        # Drop rows where target is NaN (trailing horizon)
        combined = combined.dropna(subset=["target_classification"]).reset_index(drop=True)
        combined["target_classification"] = combined["target_classification"].astype(int)

        dist = combined["target_label"].value_counts().to_dict()
        meta = {
            "target_column": "target_classification",
            "target_label_column": "target_label",
            "horizon_days": self.horizon,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "distribution": dist,
            "total_labeled_rows": len(combined)
        }

        return combined, ["target_classification", "target_label"], meta
