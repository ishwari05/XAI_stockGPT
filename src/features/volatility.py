from typing import Tuple, Sequence
import numpy as np
import pandas as pd


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR).
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def compute_atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Normalized ATR: ATR / Close.
    Allows volatility comparison across stocks with different nominal price levels.
    """
    atr = compute_atr(high, low, close, window)
    return atr / close.replace(0, np.nan)


def compute_bollinger_bands(
    close: pd.Series, 
    window: int = 20, 
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns:
        upper_band, middle_band, lower_band, bandwidth, pct_b
    """
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, bandwidth, pct_b


def compute_rolling_volatility(
    close: pd.Series, 
    windows: Sequence[int] = (10, 20)
) -> pd.DataFrame:
    """
    Computes rolling standard deviation of 1-day percentage returns:
    volatility_{w}d = std(return_1d, window=w) * sqrt(252) (annualized or raw std)
    """
    ret1d = close.pct_change(1)
    res = pd.DataFrame(index=close.index)
    for w in windows:
        res[f"volatility_{w}d"] = ret1d.rolling(window=w, min_periods=w).std()
    return res
