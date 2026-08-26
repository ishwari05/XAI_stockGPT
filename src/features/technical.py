from typing import Tuple
import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Computes Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Computes Exponential Moving Average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_price_vs_sma(close: pd.Series, window: int) -> pd.Series:
    """Computes relative distance of Close price to its SMA: (Close - SMA) / SMA."""
    sma = compute_sma(close, window)
    return (close - sma) / sma.replace(0, np.nan)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing).
    Output bounded in [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi


def compute_macd(
    close: pd.Series, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD).
    Returns:
        macd_line, signal_line, histogram
    """
    ema_fast = compute_ema(close, fast_period)
    ema_slow = compute_ema(close, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal_period)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_stochastic(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    k_window: int = 14, 
    d_window: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K, %D).
    Returns:
        percent_k, percent_d
    """
    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    percent_k = 100.0 * (close - lowest_low) / denom
    percent_d = percent_k.rolling(window=d_window, min_periods=d_window).mean()
    return percent_k, percent_d


def compute_volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Computes Simple Moving Average of Volume."""
    return volume.rolling(window=window, min_periods=window).mean()


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Computes Volume relative to its SMA: Volume / Volume_SMA."""
    vol_sma = compute_volume_sma(volume, window)
    return volume / vol_sma.replace(0, np.nan)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV)."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()
