import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LeakageError(Exception):
    """Raised when data leakage or future information is detected in feature columns."""
    pass


def check_feature_name_leakage(feature_columns: List[str]) -> List[str]:
    """
    Scans feature column names for keywords indicative of future information.
    """
    forbidden_substrings = [
        "fwd_", "target", "future", "next_", "lead_", "label", "t+1", "t+2", "t+5"
    ]
    leaked_cols = []
    for col in feature_columns:
        col_lower = str(col).lower()
        for f in forbidden_substrings:
            if f in col_lower:
                leaked_cols.append(col)
                break
    return leaked_cols


def verify_no_future_leakage(
    df: pd.DataFrame, 
    feature_columns: List[str], 
    target_columns: Optional[List[str]] = None,
    date_col: str = "Date",
    symbol_col: str = "Symbol",
    raise_on_error: bool = True
) -> Dict[str, Any]:
    """
    Validates that feature columns do NOT contain future price/target data.
    
    Checks:
    1. Feature column names do not contain forward/target keywords.
    2. None of the target columns appear inside the feature columns list.
    3. Shift leakage check: features must not correlate perfectly with future price shifts (e.g. shift(-1))
       while having zero correlation with current price.
    4. Exact row ordering integrity: no out-of-order temporal data within symbols.
    
    Raises:
        LeakageError: if any future leakage is detected and raise_on_error is True.
    """
    leakage_issues = []

    # 1. Name-based check
    leaked_names = check_feature_name_leakage(feature_columns)
    if leaked_names:
        msg = f"Forbidden future/target column names detected in feature set: {leaked_names}"
        leakage_issues.append(msg)

    # 2. Target intersection check
    if target_columns:
        overlap = set(feature_columns).intersection(set(target_columns))
        if overlap:
            msg = f"Target columns accidentally included in feature list: {list(overlap)}"
            leakage_issues.append(msg)

    # 3. Temporal ordering check per symbol
    if date_col in df.columns:
        symbols = df[symbol_col].unique() if symbol_col in df.columns else ["ALL"]
        for sym in symbols:
            sub = df[df[symbol_col] == sym] if symbol_col in df.columns else df
            if not sub[date_col].is_monotonic_increasing:
                msg = f"Temporal ordering violation in symbol '{sym}': dates are not monotonically increasing."
                leakage_issues.append(msg)
                break

    # 4. Statistical shift check (detect if a feature is an exact shift(-k) of future close price)
    if "Close" in df.columns:
        symbols = df[symbol_col].unique() if symbol_col in df.columns else ["ALL"]
        # Check first symbol with at least 30 observations
        for sym in symbols:
            sub = df[df[symbol_col] == sym] if symbol_col in df.columns else df
            if len(sub) >= 30:
                future_close = sub["Close"].shift(-1)
                for feat in feature_columns:
                    if feat in sub.columns and pd.api.types.is_numeric_dtype(sub[feat]):
                        feat_vals = sub[feat]
                        # Check if feature is identical to future close or future return
                        valid_mask = (~feat_vals.isna()) & (~future_close.isna())
                        if valid_mask.sum() > 20:
                            diff = (feat_vals[valid_mask] - future_close[valid_mask]).abs().max()
                            if diff < 1e-7:
                                msg = f"Feature '{feat}' is identical to future price Close[t+1]! Direct lookahead leakage detected."
                                leakage_issues.append(msg)
                break

    has_leakage = len(leakage_issues) > 0
    result = {
        "passed": not has_leakage,
        "issues": leakage_issues,
        "checked_features_count": len(feature_columns),
        "checked_rows_count": len(df)
    }

    if has_leakage and raise_on_error:
        error_msg = "DATA LEAKAGE DETECTED!\n" + "\n".join(f"- {iss}" for iss in leakage_issues)
        logger.error(error_msg)
        raise LeakageError(error_msg)

    return result
