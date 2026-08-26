from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from .technical import (
    compute_sma, 
    compute_ema, 
    compute_price_vs_sma,
    compute_rsi, 
    compute_macd, 
    compute_stochastic,
    compute_volume_sma,
    compute_volume_ratio,
    compute_obv
)
from .returns import compute_lagged_returns
from .volatility import (
    compute_atr, 
    compute_atr_ratio, 
    compute_bollinger_bands, 
    compute_rolling_volatility
)


class BaseFeatureBuilder(ABC):
    """Abstract interface for feature engineering."""

    @abstractmethod
    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        pass


class FeatureBuilder(BaseFeatureBuilder):
    """
    Configuration-driven feature builder.
    Computes configured price, trend, momentum, volatility, volume, and return features.
    Maintains clean separation between features and targets, tracks warm-up rows, and
    records metadata for every generated feature.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feat_cfg = self.config.get("features", {})
        self.warmup_cfg = self.config.get("warmup", {})
        self.drop_warmup = bool(self.warmup_cfg.get("drop_warmup_rows", True))

    def build_features_for_symbol(self, df_symbol: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Builds all configured technical features for a single stock DataFrame.
        Assumes df_symbol is sorted chronologically by Date.
        """
        out = df_symbol.copy()
        out = out.sort_values("Date").reset_index(drop=True)

        initial_row_count = len(out)
        close = out["Close"]
        high = out["High"]
        low = out["Low"]
        volume = out["Volume"]

        generated_features: List[str] = []
        feature_descriptions: Dict[str, str] = {}

        # 1. Trend Features
        trend_cfg = self.feat_cfg.get("trend", {})
        if trend_cfg.get("enabled", True):
            for w in trend_cfg.get("sma_periods", [20, 50]):
                col = f"sma_{w}"
                out[col] = compute_sma(close, w)
                generated_features.append(col)
                feature_descriptions[col] = f"{w}-day Simple Moving Average of Close price"

            for s in trend_cfg.get("ema_periods", [20, 50]):
                col = f"ema_{s}"
                out[col] = compute_ema(close, s)
                generated_features.append(col)
                feature_descriptions[col] = f"{s}-day Exponential Moving Average of Close price"

            for w in trend_cfg.get("price_vs_sma_periods", [20, 50]):
                col = f"price_vs_sma{w}"
                out[col] = compute_price_vs_sma(close, w)
                generated_features.append(col)
                feature_descriptions[col] = f"Percentage distance of Close price to {w}-day SMA"

            macd_cfg = trend_cfg.get("macd", {})
            if macd_cfg.get("enabled", True):
                fp = macd_cfg.get("fast_period", 12)
                sp = macd_cfg.get("slow_period", 26)
                sig = macd_cfg.get("signal_period", 9)
                m, s, h = compute_macd(close, fast_period=fp, slow_period=sp, signal_period=sig)
                out["macd"] = m
                out["macd_signal"] = s
                out["macd_hist"] = h
                generated_features.extend(["macd", "macd_signal", "macd_hist"])
                feature_descriptions["macd"] = f"MACD line ({fp}, {sp})"
                feature_descriptions["macd_signal"] = f"MACD signal line ({sig})"
                feature_descriptions["macd_hist"] = f"MACD histogram (MACD - Signal)"

        # 2. Momentum Features
        mom_cfg = self.feat_cfg.get("momentum", {})
        if mom_cfg.get("enabled", True):
            rsi_p = mom_cfg.get("rsi_period", 14)
            rsi_col = f"rsi_{rsi_p}"
            out[rsi_col] = compute_rsi(close, rsi_p)
            generated_features.append(rsi_col)
            feature_descriptions[rsi_col] = f"Relative Strength Index ({rsi_p} periods)"

            stoch_cfg = mom_cfg.get("stochastic", {})
            if stoch_cfg.get("enabled", True):
                kw = stoch_cfg.get("k_window", 14)
                dw = stoch_cfg.get("d_window", 3)
                pk, pd_ = compute_stochastic(high, low, close, k_window=kw, d_window=dw)
                out["stoch_k"] = pk
                out["stoch_d"] = pd_
                generated_features.extend(["stoch_k", "stoch_d"])
                feature_descriptions["stoch_k"] = f"Stochastic %K ({kw} periods)"
                feature_descriptions["stoch_d"] = f"Stochastic %D ({dw} periods)"

        # 3. Volatility Features
        vol_cfg = self.feat_cfg.get("volatility", {})
        if vol_cfg.get("enabled", True):
            atr_p = vol_cfg.get("atr_period", 14)
            atr_col = f"atr_{atr_p}"
            out[atr_col] = compute_atr(high, low, close, atr_p)
            out["atr_ratio"] = compute_atr_ratio(high, low, close, atr_p)
            generated_features.extend([atr_col, "atr_ratio"])
            feature_descriptions[atr_col] = f"Average True Range ({atr_p} periods)"
            feature_descriptions["atr_ratio"] = f"Normalized ATR ratio (ATR_{atr_p} / Close)"

            bb_cfg = vol_cfg.get("bollinger", {})
            if bb_cfg.get("enabled", True):
                bb_p = bb_cfg.get("period", 20)
                num_std = bb_cfg.get("num_std", 2.0)
                up, mid, low_bb, bw, pct_b = compute_bollinger_bands(close, window=bb_p, num_std=num_std)
                out["bb_upper"] = up
                out["bb_middle"] = mid
                out["bb_lower"] = low_bb
                out["bb_bandwidth"] = bw
                out["bb_pct_b"] = pct_b
                bb_cols = ["bb_upper", "bb_middle", "bb_lower", "bb_bandwidth", "bb_pct_b"]
                generated_features.extend(bb_cols)
                feature_descriptions["bb_upper"] = f"Bollinger Upper Band ({bb_p}, {num_std} std)"
                feature_descriptions["bb_middle"] = f"Bollinger Middle Band ({bb_p} SMA)"
                feature_descriptions["bb_lower"] = f"Bollinger Lower Band ({bb_p}, {num_std} std)"
                feature_descriptions["bb_bandwidth"] = f"Bollinger Bandwidth: (Upper - Lower) / Middle"
                feature_descriptions["bb_pct_b"] = f"Bollinger %B: (Close - Lower) / (Upper - Lower)"

            vol_windows = vol_cfg.get("rolling_volatility_windows", [10, 20])
            if vol_windows:
                rvol_df = compute_rolling_volatility(close, windows=vol_windows)
                for c in rvol_df.columns:
                    out[c] = rvol_df[c]
                    generated_features.append(c)
                    feature_descriptions[c] = f"Rolling standard deviation of 1-day returns over {c.split('_')[1]} window"

        # 4. Volume Features
        vol_feat_cfg = self.feat_cfg.get("volume", {})
        if vol_feat_cfg.get("enabled", True):
            vol_p = vol_feat_cfg.get("volume_sma_period", 20)
            if vol_p:
                out[f"volume_sma_{vol_p}"] = compute_volume_sma(volume, vol_p)
                generated_features.append(f"volume_sma_{vol_p}")
                feature_descriptions[f"volume_sma_{vol_p}"] = f"{vol_p}-day SMA of Volume"

            if vol_feat_cfg.get("volume_ratio", True):
                out["volume_ratio"] = compute_volume_ratio(volume, vol_p)
                generated_features.append("volume_ratio")
                feature_descriptions["volume_ratio"] = f"Volume relative to {vol_p}-day Volume SMA"

            if vol_feat_cfg.get("obv", True):
                out["obv"] = compute_obv(close, volume)
                generated_features.append("obv")
                feature_descriptions["obv"] = "On-Balance Volume"

        # 5. Historical Returns Features
        ret_cfg = self.feat_cfg.get("returns", {})
        if ret_cfg.get("enabled", True):
            horizons = ret_cfg.get("horizons", [1, 5, 10, 20])
            log_ret = ret_cfg.get("log_returns", False)
            ret_df = compute_lagged_returns(close, horizons=horizons, log_returns=log_ret)
            for c in ret_df.columns:
                out[c] = ret_df[c]
                generated_features.append(c)
                feature_descriptions[c] = f"{c.split('_')[1]} trailing historical return (t vs t-h)"

        # 6. Warm-up Period Handling
        warmup_rows_dropped = 0
        if self.drop_warmup and generated_features:
            # Find rows with any NaN in the generated feature subset
            feature_nan_mask = out[generated_features].isna().any(axis=1)
            warmup_rows_dropped = int(feature_nan_mask.sum())
            out = out[~feature_nan_mask].reset_index(drop=True)

        metadata = {
            "initial_rows": initial_row_count,
            "final_rows": len(out),
            "warmup_rows_dropped": warmup_rows_dropped,
            "feature_count": len(generated_features),
            "features": generated_features,
            "descriptions": feature_descriptions
        }

        return out, generated_features, metadata

    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Builds features across a DataFrame containing one or multiple tickers.
        Processes each ticker independently to avoid cross-ticker boundary bleeding.
        """
        if "Symbol" in df.columns:
            symbols = df["Symbol"].unique()
            dfs = []
            total_warmup = 0
            feature_cols = []
            feature_descs = {}

            for sym in symbols:
                sub = df[df["Symbol"] == sym].copy()
                sub_feats, feats, meta = self.build_features_for_symbol(sub)
                dfs.append(sub_feats)
                total_warmup += meta["warmup_rows_dropped"]
                if not feature_cols:
                    feature_cols = feats
                    feature_descs = meta["descriptions"]

            combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            overall_meta = {
                "initial_rows": len(df),
                "final_rows": len(combined),
                "warmup_rows_dropped": total_warmup,
                "feature_count": len(feature_cols),
                "features": feature_cols,
                "descriptions": feature_descs
            }
            return combined, feature_cols, overall_meta
        else:
            return self.build_features_for_symbol(df)
