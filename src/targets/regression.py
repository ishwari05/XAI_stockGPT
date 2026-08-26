from typing import Dict, Any, List, Tuple, Sequence
import pandas as pd
from .classification import BaseTargetGenerator


class ContinuousReturnTargetGenerator(BaseTargetGenerator):
    """
    Generates continuous forward return target(s) for regression models:
    target_return_{h}d = (Close[t+h] - Close[t]) / Close[t]
    
    Processes each symbol independently to prevent cross-ticker boundary leakage.
    """

    def __init__(
        self,
        horizons: Sequence[int] = (1, 5, 10),
        close_col: str = "Close",
        symbol_col: str = "Symbol",
        date_col: str = "Date"
    ):
        self.horizons = list(horizons)
        self.close_col = close_col
        self.symbol_col = symbol_col
        self.date_col = date_col

    def _generate_for_symbol(self, df_symbol: pd.DataFrame) -> pd.DataFrame:
        out = df_symbol.copy()
        out = out.sort_values(self.date_col).reset_index(drop=True)
        close = out[self.close_col]

        for h in self.horizons:
            col_name = f"target_return_{h}d"
            out[col_name] = close.shift(-h) / close - 1.0
        return out

    def generate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Appends regression targets across single or multiple tickers.
        """
        if df.empty:
            target_cols = [f"target_return_{h}d" for h in self.horizons]
            return df, target_cols, {}

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

        target_cols = [f"target_return_{h}d" for h in self.horizons]
        # Drop rows where primary regression target is NaN
        primary_col = target_cols[0]
        combined = combined.dropna(subset=[primary_col]).reset_index(drop=True)

        meta = {
            "regression_target_columns": target_cols,
            "horizons": self.horizons,
            "total_labeled_rows": len(combined)
        }

        return combined, target_cols, meta
