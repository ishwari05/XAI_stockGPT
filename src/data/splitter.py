from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd


class BaseSplitter(ABC):
    """Abstract interface for dataset splitting."""

    @abstractmethod
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
        pass


class ChronologicalSplitter(BaseSplitter):
    """
    Strict chronological dataset splitter.
    Ensures zero temporal leakage across train, validation, and locked test partitions.
    """

    def __init__(self, train_end_date: str, validation_end_date: str, 
                 test_start_date: Optional[str] = None, date_col: str = "Date"):
        self.train_end = pd.to_datetime(train_end_date)
        self.val_end = pd.to_datetime(validation_end_date)
        self.test_start = pd.to_datetime(test_start_date) if test_start_date else self.val_end
        self.date_col = date_col

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits DataFrame chronologically based on strict date thresholds.
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_sorted = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[self.date_col]):
            df_sorted[self.date_col] = pd.to_datetime(df_sorted[self.date_col])

        df_sorted = df_sorted.sort_values([self.date_col, "Symbol"] if "Symbol" in df_sorted.columns else [self.date_col]).reset_index(drop=True)

        train_df = df_sorted[df_sorted[self.date_col] <= self.train_end].copy().reset_index(drop=True)
        val_df = df_sorted[
            (df_sorted[self.date_col] > self.train_end) & 
            (df_sorted[self.date_col] <= self.val_end)
        ].copy().reset_index(drop=True)
        test_df = df_sorted[df_sorted[self.date_col] >= self.test_start].copy().reset_index(drop=True)

        # Verification assertion: No overlapping timestamps
        if not train_df.empty and not val_df.empty:
            assert train_df[self.date_col].max() < val_df[self.date_col].min(), "Leakage: Train overlaps with Validation!"
        if not val_df.empty and not test_df.empty:
            assert val_df[self.date_col].max() < test_df[self.date_col].min(), "Leakage: Validation overlaps with Test!"

        return train_df, val_df, test_df

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                    test_df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
        """Saves partitioned splits to Parquet files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.parquet"
        val_path = output_dir / "validation.parquet"
        test_path = output_dir / "test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        return {
            "train": str(train_path),
            "validation": str(val_path),
            "test": str(test_path),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        }
