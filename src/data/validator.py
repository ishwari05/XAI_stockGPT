import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseDataValidator(ABC):
    """Abstract interface for market data validation."""

    @abstractmethod
    def validate(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, Dict[str, Any], List[str]]:
        pass

    @abstractmethod
    def generate_validation_report(self, validation_results: List[Dict[str, Any]], output_path: Path) -> pd.DataFrame:
        pass


class MarketDataValidator(BaseDataValidator):
    """
    Comprehensive financial market data validator.
    Ensures research-grade data integrity:
    1. Required columns & Date format
    2. Duplicate dates detection
    3. Missing values & null ratios
    4. OHLC logical consistency (High >= Open, High >= Close, Low <= Open, Low <= Close)
    5. Non-negative prices and volume
    6. Minimum sample length requirement
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.val_cfg = self.config.get("validation", {})
        self.min_rows = int(self.val_cfg.get("min_rows", 100))
        self.missing_thresh = float(self.val_cfg.get("missing_threshold", 0.05))
        self.check_ohlc = bool(self.val_cfg.get("check_ohlc_consistency", True))
        self.check_non_neg = bool(self.val_cfg.get("check_non_negative", True))
        self.check_dups = bool(self.val_cfg.get("check_duplicate_dates", True))

    def validate(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, Dict[str, Any], List[str]]:
        issues: List[str] = []
        n_rows = len(df)

        result: Dict[str, Any] = {
            "symbol": symbol,
            "rows": n_rows,
            "passed": True,
            "start_date": None,
            "end_date": None,
            "duplicate_dates": 0,
            "missing_values_total": 0,
            "missing_pct": 0.0,
            "ohlc_violations": 0,
            "negative_prices": 0,
            "negative_volume": 0,
            "issues": []
        }

        # 1. Check empty & minimum rows
        if n_rows == 0:
            issues.append("DataFrame is empty.")
            result["passed"] = False
            result["issues"] = issues
            return False, result, issues

        if n_rows < self.min_rows:
            issues.append(f"Insufficient history: {n_rows} rows < minimum {self.min_rows}.")
            result["passed"] = False

        # 2. Check required columns
        req_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            result["passed"] = False
            result["issues"] = issues
            return False, result, issues

        # 3. Check Date column
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            try:
                pd.to_datetime(df["Date"])
            except Exception as e:
                issues.append(f"Invalid Date format: {e}")
                result["passed"] = False

        result["start_date"] = str(df["Date"].min().date()) if not df["Date"].isna().all() else "N/A"
        result["end_date"] = str(df["Date"].max().date()) if not df["Date"].isna().all() else "N/A"

        # 4. Check duplicate dates
        if self.check_dups:
            dup_count = int(df.duplicated(subset=["Date"]).sum())
            result["duplicate_dates"] = dup_count
            if dup_count > 0:
                issues.append(f"Found {dup_count} duplicate dates.")
                result["passed"] = False

        # 5. Missing values
        total_missing = int(df[req_cols].isna().sum().sum())
        missing_pct = total_missing / (n_rows * len(req_cols))
        result["missing_values_total"] = total_missing
        result["missing_pct"] = round(missing_pct, 4)
        if missing_pct > self.missing_thresh:
            issues.append(f"Missing value ratio ({missing_pct:.2%}) exceeds threshold ({self.missing_thresh:.2%}).")
            result["passed"] = False

        # 6. OHLC Logical Consistency
        if self.check_ohlc:
            # High >= Open and High >= Close
            # Low <= Open and Low <= Close
            # High >= Low
            high_invalid = (df["High"] < df["Open"]) | (df["High"] < df["Close"]) | (df["High"] < df["Low"])
            low_invalid = (df["Low"] > df["Open"]) | (df["Low"] > df["Close"])
            ohlc_violations = int((high_invalid | low_invalid).sum())
            result["ohlc_violations"] = ohlc_violations
            if ohlc_violations > 0:
                issues.append(f"Found {ohlc_violations} rows with OHLC logical inconsistencies (e.g. High < Low or High < Close).")
                result["passed"] = False

        # 7. Non-negative prices & volume
        if self.check_non_neg:
            neg_prices = int(((df[["Open", "High", "Low", "Close"]] < 0).any(axis=1)).sum())
            neg_vol = int((df["Volume"] < 0).sum())
            result["negative_prices"] = neg_prices
            result["negative_volume"] = neg_vol

            if neg_prices > 0:
                issues.append(f"Found {neg_prices} rows with negative prices.")
                result["passed"] = False
            if neg_vol > 0:
                issues.append(f"Found {neg_vol} rows with negative volume.")
                result["passed"] = False

        result["issues"] = issues
        return result["passed"], result, issues

    def generate_validation_report(self, validation_results: List[Dict[str, Any]], output_path: Path) -> pd.DataFrame:
        """Saves a summary CSV validation report to reports/data_validation/."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df = pd.DataFrame(validation_results)
        
        # Flatten list of issues into string for CSV export
        if "issues" in report_df.columns:
            report_df["issues_summary"] = report_df["issues"].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
            report_df = report_df.drop(columns=["issues"])

        report_df.to_csv(output_path, index=False)
        return report_df
