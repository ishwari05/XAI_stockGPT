from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


class BaseDataCleaner(ABC):
    """Abstract interface for data cleaning."""

    @abstractmethod
    def clean(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        pass


class MarketDataCleaner(BaseDataCleaner):
    """
    Standardized market data cleaner.
    Performs:
    1. Column name standardization (Date, Open, High, Low, Close, Volume, Symbol)
    2. Duplicate removal
    3. Chronological sorting
    4. Invalid row detection (NaNs, nulls)
    5. Configurable missing-value handling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cleaning_cfg = self.config.get("cleaning", {})
        self.remove_dups = bool(self.cleaning_cfg.get("remove_duplicates", True))
        self.fill_missing = bool(self.cleaning_cfg.get("fill_missing", False))
        self.drop_invalid = bool(self.cleaning_cfg.get("drop_invalid_rows", True))
        self.max_consecutive_missing = int(self.cleaning_cfg.get("max_consecutive_missing", 5))

    def standardize_columns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardizes column names to Title Case: Date, Open, High, Low, Close, Volume, Symbol."""
        out = df.copy()
        
        # Mapping variations
        col_map = {}
        for c in out.columns:
            cl = str(c).lower().strip()
            if cl in ["date", "datetime", "timestamp"]:
                col_map[c] = "Date"
            elif cl == "open":
                col_map[c] = "Open"
            elif cl == "high":
                col_map[c] = "High"
            elif cl == "low":
                col_map[c] = "Low"
            elif cl in ["close", "adj close", "adj_close"]:
                if "Close" not in col_map.values():
                    col_map[c] = "Close"
            elif cl == "volume":
                col_map[c] = "Volume"

        out = out.rename(columns=col_map)
        
        # Ensure Symbol column exists
        if "Symbol" not in out.columns:
            out["Symbol"] = symbol

        # Ensure Date is parsed as datetime
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)

        # Standard OHLCV column selection
        std_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
        present_cols = [c for c in std_cols if c in out.columns]
        return out[present_cols]

    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Removes duplicate rows based on Date."""
        if "Date" not in df.columns:
            return df, 0
        before = len(df)
        df_dedup = df.drop_duplicates(subset=["Date"], keep="last")
        removed = before - len(df_dedup)
        return df_dedup, removed

    def sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sorts chronologically ascending by Date."""
        if "Date" in df.columns:
            return df.sort_values("Date").reset_index(drop=True)
        return df.reset_index(drop=True)

    def detect_invalid_rows(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Detects and optionally drops rows with invalid OHLCV values (e.g. non-numeric, all NaNs)."""
        ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        
        # Coerce to numeric
        df_num = df.copy()
        for col in ohlcv_cols:
            df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

        # Find rows with any NaN in OHLCV
        invalid_mask = df_num[ohlcv_cols].isna().any(axis=1)
        invalid_count = int(invalid_mask.sum())

        if self.drop_invalid:
            df_valid = df_num[~invalid_mask].reset_index(drop=True)
            return df_valid, invalid_count
        return df_num, invalid_count

    def fill_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Optionally forward-fills missing price data within the configured consecutive threshold."""
        if not self.fill_missing:
            return df, 0
        
        price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        before_nulls = df[price_cols].isna().sum().sum()
        
        df_filled = df.copy()
        df_filled[price_cols] = df_filled[price_cols].ffill(limit=self.max_consecutive_missing)
        if "Volume" in df_filled.columns:
            df_filled["Volume"] = df_filled["Volume"].fillna(0)

        after_nulls = df_filled[price_cols].isna().sum().sum()
        filled_count = int(before_nulls - after_nulls)
        return df_filled, filled_count

    def clean(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes full standardized cleaning pipeline on input DataFrame.
        """
        raw_count = len(df)
        
        # 1. Standardize column names
        df_std = self.standardize_columns(df, symbol)
        
        # 2. Remove duplicates
        if self.remove_dups:
            df_dedup, dups_removed = self.remove_duplicates(df_std)
        else:
            df_dedup, dups_removed = df_std, 0

        # 3. Sort by date
        df_sorted = self.sort_by_date(df_dedup)

        # 4. Fill missing values (if enabled)
        df_filled, filled_count = self.fill_missing_values(df_sorted)

        # 5. Detect and drop invalid rows
        df_clean, invalid_dropped = self.detect_invalid_rows(df_filled)

        metrics = {
            "raw_rows": raw_count,
            "cleaned_rows": len(df_clean),
            "duplicates_removed": dups_removed,
            "missing_filled": filled_count,
            "invalid_rows_dropped": invalid_dropped
        }

        return df_clean, metrics
