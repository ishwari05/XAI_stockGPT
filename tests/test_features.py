import unittest
import pandas as pd
import numpy as np

from src.features.technical import (
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
from src.features.returns import compute_lagged_returns
from src.features.volatility import (
    compute_atr, 
    compute_atr_ratio, 
    compute_bollinger_bands, 
    compute_rolling_volatility
)
from src.features.builder import FeatureBuilder
from src.targets.classification import ClassificationTargetGenerator
from src.targets.regression import ContinuousReturnTargetGenerator
from src.utils.leakage import verify_no_future_leakage, LeakageError


class TestFeaturesAndTargets(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=150, freq="D")
        
        # Plausible OHLCV series
        price = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        high = price * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        low = price * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        open_ = low + (high - low) * np.random.uniform(0.2, 0.8, len(dates))
        volume = np.random.uniform(10000, 50000, len(dates))

        self.df = pd.DataFrame({
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": price,
            "Volume": volume,
            "Symbol": "TEST"
        })

    def test_technical_indicators(self):
        close = self.df["Close"]
        high = self.df["High"]
        low = self.df["Low"]
        volume = self.df["Volume"]

        # SMA & EMA
        sma20 = compute_sma(close, 20)
        self.assertEqual(len(sma20), 150)
        self.assertTrue(sma20.iloc[:19].isna().all())
        self.assertFalse(sma20.iloc[19:].isna().any())

        ema20 = compute_ema(close, 20)
        self.assertEqual(len(ema20), 150)

        # Price vs SMA
        p_sma = compute_price_vs_sma(close, 20)
        self.assertEqual(len(p_sma), 150)

        # RSI
        rsi = compute_rsi(close, 14)
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all())

        # MACD
        m, s, h = compute_macd(close)
        self.assertEqual(len(m), 150)
        self.assertEqual(len(s), 150)
        self.assertEqual(len(h), 150)

        # Stochastic
        pk, pd_ = compute_stochastic(high, low, close)
        valid_pk = pk.dropna()
        self.assertTrue((valid_pk >= 0.0).all() and (valid_pk <= 100.0).all())

        # Volume
        v_sma = compute_volume_sma(volume, 20)
        v_ratio = compute_volume_ratio(volume, 20)
        obv = compute_obv(close, volume)
        self.assertEqual(len(v_ratio), 150)
        self.assertEqual(len(obv), 150)

    def test_volatility_indicators(self):
        high = self.df["High"]
        low = self.df["Low"]
        close = self.df["Close"]

        # ATR & ATR Ratio
        atr = compute_atr(high, low, close, 14)
        atr_r = compute_atr_ratio(high, low, close, 14)
        self.assertTrue((atr.dropna() > 0).all())
        self.assertTrue((atr_r.dropna() > 0).all())

        # Bollinger Bands
        up, mid, low_bb, bw, pct_b = compute_bollinger_bands(close, window=20, num_std=2.0)
        valid_idx = ~(up.isna() | mid.isna() | low_bb.isna())
        self.assertTrue((up[valid_idx] >= mid[valid_idx]).all())
        self.assertTrue((mid[valid_idx] >= low_bb[valid_idx]).all())
        self.assertTrue((bw[valid_idx] >= 0).all())

        # Rolling Volatility
        rvol = compute_rolling_volatility(close, windows=[10, 20])
        self.assertIn("volatility_10d", rvol.columns)
        self.assertIn("volatility_20d", rvol.columns)

    def test_return_features(self):
        close = self.df["Close"]
        rets = compute_lagged_returns(close, horizons=[1, 5, 10, 20])
        self.assertIn("return_1d", rets.columns)
        self.assertIn("return_5d", rets.columns)
        self.assertIn("return_10d", rets.columns)
        self.assertIn("return_20d", rets.columns)
        
        # Verify backward-looking calculation
        expected_ret1 = (close.iloc[10] - close.iloc[9]) / close.iloc[9]
        self.assertAlmostEqual(rets["return_1d"].iloc[10], expected_ret1)

    def test_feature_builder_and_warmup_dropping(self):
        builder = FeatureBuilder({
            "features": {
                "trend": {"enabled": True, "sma_periods": [20, 50]},
                "momentum": {"enabled": True, "rsi_period": 14},
                "volatility": {"enabled": True, "atr_period": 14, "bollinger": {"enabled": True, "period": 20}},
                "returns": {"enabled": True, "horizons": [1, 5]}
            },
            "warmup": {"drop_warmup_rows": True}
        })
        out_df, feature_cols, meta = builder.build_features(self.df)

        self.assertGreater(len(feature_cols), 10)
        self.assertIn("sma_50", feature_cols)
        self.assertIn("rsi_14", feature_cols)
        # Should have dropped warm-up rows (initial 49 rows for a 50-period rolling window)
        self.assertEqual(meta["warmup_rows_dropped"], 49)
        self.assertEqual(len(out_df), 101)
        self.assertFalse(out_df[feature_cols].isna().any().any())

    def test_target_generation_classification(self):
        clf_gen = ClassificationTargetGenerator(horizon=1, buy_threshold=0.01, sell_threshold=-0.01)
        out_df, cols, meta = clf_gen.generate(self.df)

        self.assertIn("target_classification", out_df.columns)
        self.assertIn("target_label", out_df.columns)
        self.assertTrue(set(out_df["target_classification"].unique()).issubset({0, 1, 2}))
        # Last row should be dropped because forward return is unobservable
        self.assertEqual(len(out_df), len(self.df) - 1)
        self.assertIn("BUY", meta["distribution"])

    def test_target_generation_regression(self):
        reg_gen = ContinuousReturnTargetGenerator(horizons=[1, 5, 10])
        out_df, cols, meta = reg_gen.generate(self.df)

        self.assertIn("target_return_1d", out_df.columns)
        self.assertIn("target_return_5d", out_df.columns)
        self.assertIn("target_return_10d", out_df.columns)
        # Trailing rows dropped based on primary target
        self.assertEqual(len(out_df), len(self.df) - 1)

    def test_leakage_check_passes_on_clean_data(self):
        builder = FeatureBuilder({})
        df_feat, feats, _ = builder.build_features(self.df)
        clf_gen = ClassificationTargetGenerator(horizon=1)
        df_target, target_cols, _ = clf_gen.generate(df_feat)

        # Verification must pass without error
        res = verify_no_future_leakage(
            df=df_target, 
            feature_columns=feats, 
            target_columns=target_cols,
            raise_on_error=True
        )
        self.assertTrue(res["passed"])

    def test_leakage_check_fails_on_deliberately_injected_future_information(self):
        builder = FeatureBuilder({})
        df_feat, feats, _ = builder.build_features(self.df)
        
        # Case A: Inject future column name
        leaked_feats_a = feats + ["fwd_return_5d"]
        with self.assertRaises(LeakageError):
            verify_no_future_leakage(df=df_feat, feature_columns=leaked_feats_a, raise_on_error=True)

        # Case B: Include target in feature list
        leaked_feats_b = feats + ["target_classification"]
        with self.assertRaises(LeakageError):
            verify_no_future_leakage(
                df=df_feat, 
                feature_columns=leaked_feats_b, 
                target_columns=["target_classification"],
                raise_on_error=True
            )

        # Case C: Inject future price shift Close[t+1] directly into feature values
        df_leaked = df_feat.copy()
        df_leaked["sneaky_feature"] = df_leaked["Close"].shift(-1)
        with self.assertRaises(LeakageError):
            verify_no_future_leakage(
                df=df_leaked, 
                feature_columns=feats + ["sneaky_feature"], 
                raise_on_error=True
            )


if __name__ == "__main__":
    unittest.main()
