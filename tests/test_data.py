import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.collector import MarketDataCollector
from src.data.cleaner import MarketDataCleaner
from src.data.validator import MarketDataValidator
from src.data.splitter import ChronologicalSplitter


class TestMarketDataCollector(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            "data_source": {"provider": "yfinance", "retry_attempts": 2, "request_delay_seconds": 0.0},
            "tickers": {"source": "list", "symbols": ["RELIANCE.NS", "TCS.NS"]},
            "date_range": {"start": "2023-01-01", "end": "2023-01-10"},
            "storage": {"raw_path": str(self.temp_dir / "raw"), "logs_path": str(self.temp_dir / "logs")}
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_tickers_from_list(self):
        collector = MarketDataCollector(self.config, project_root=self.temp_dir)
        tickers = collector.load_tickers()
        self.assertEqual(tickers, ["RELIANCE.NS", "TCS.NS"])

    def test_load_tickers_from_file(self):
        file_path = self.temp_dir / "tickers.csv"
        pd.DataFrame({"Symbol": ["INFY.NS", "HDFCBANK.NS", "SBIN.NS"]}).to_csv(file_path, index=False)
        
        cfg = self.config.copy()
        cfg["tickers"] = {"source": "file", "path": str(file_path), "symbol_column": "Symbol"}
        
        collector = MarketDataCollector(cfg, project_root=self.temp_dir)
        tickers = collector.load_tickers()
        self.assertEqual(tickers, ["HDFCBANK.NS", "INFY.NS", "SBIN.NS"])

    def test_save_raw_data(self):
        collector = MarketDataCollector(self.config, project_root=self.temp_dir)
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=5),
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [95, 96, 97, 98, 99],
            "Close": [102, 103, 104, 105, 106],
            "Volume": [1000, 1100, 1200, 1300, 1400],
            "Symbol": ["TEST", "TEST", "TEST", "TEST", "TEST"]
        })
        res = collector.save_raw_data({"TEST": df})
        self.assertEqual(res["saved_csv_count"], 1)
        self.assertTrue((self.temp_dir / "raw" / "TEST.csv").exists())
        self.assertTrue((self.temp_dir / "raw" / "combined_raw.parquet").exists())


class TestMarketDataCleaner(unittest.TestCase):

    def setUp(self):
        self.cleaner = MarketDataCleaner({
            "cleaning": {
                "remove_duplicates": True,
                "fill_missing": True,
                "drop_invalid_rows": True,
                "max_consecutive_missing": 2
            }
        })

    def test_standardize_columns(self):
        df = pd.DataFrame({
            "datetime": ["2023-01-01", "2023-01-02"],
            "open": [100.0, 101.0],
            "HIGH": [105.0, 106.0],
            "Low": [95.0, 96.0],
            "adj close": [102.0, 103.0],
            "Volume": [1000, 2000]
        })
        res = self.cleaner.standardize_columns(df, "TEST")
        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
        for col in expected_cols:
            self.assertIn(col, res.columns)
        self.assertEqual(res["Symbol"].iloc[0], "TEST")

    def test_remove_duplicates_and_sort(self):
        df = pd.DataFrame({
            "Date": ["2023-01-03", "2023-01-01", "2023-01-01"],
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [102, 103, 104],
            "Volume": [1000, 1100, 1200],
            "Symbol": ["TEST", "TEST", "TEST"]
        })
        clean_df, metrics = self.cleaner.clean(df, "TEST")
        self.assertEqual(len(clean_df), 2)
        self.assertEqual(metrics["duplicates_removed"], 1)
        self.assertTrue(clean_df["Date"].is_monotonic_increasing)

    def test_detect_and_drop_invalid_rows(self):
        df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "Open": [100.0, np.nan],
            "High": [105.0, 106.0],
            "Low": [95.0, 96.0],
            "Close": [102.0, 103.0],
            "Volume": [1000, 2000],
            "Symbol": ["TEST", "TEST"]
        })
        cleaner_no_fill = MarketDataCleaner({"cleaning": {"fill_missing": False, "drop_invalid_rows": True}})
        clean_df, metrics = cleaner_no_fill.clean(df, "TEST")
        self.assertEqual(len(clean_df), 1)
        self.assertEqual(metrics["invalid_rows_dropped"], 1)


class TestMarketDataValidator(unittest.TestCase):

    def setUp(self):
        self.validator = MarketDataValidator({
            "validation": {
                "min_rows": 3,
                "missing_threshold": 0.05,
                "check_ohlc_consistency": True,
                "check_non_negative": True,
                "check_duplicate_dates": True
            }
        })

    def test_valid_dataset_passes(self):
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [95.0, 96.0, 97.0],
            "Close": [102.0, 103.0, 104.0],
            "Volume": [1000, 2000, 3000],
            "Symbol": ["TEST", "TEST", "TEST"]
        })
        passed, result, issues = self.validator.validate(df, "TEST")
        self.assertTrue(passed)
        self.assertEqual(len(issues), 0)
        self.assertEqual(result["ohlc_violations"], 0)

    def test_ohlc_violations_fail(self):
        # High is lower than Open
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Open": [110.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0], # High < Open for first row
            "Low": [95.0, 96.0, 97.0],
            "Close": [102.0, 103.0, 104.0],
            "Volume": [1000, 2000, 3000],
            "Symbol": ["TEST", "TEST", "TEST"]
        })
        passed, result, issues = self.validator.validate(df, "TEST")
        self.assertFalse(passed)
        self.assertGreater(result["ohlc_violations"], 0)

    def test_negative_values_fail(self):
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [-5.0, 96.0, 97.0], # Negative Low
            "Close": [102.0, 103.0, 104.0],
            "Volume": [-100, 2000, 3000], # Negative Volume
            "Symbol": ["TEST", "TEST", "TEST"]
        })
        passed, result, issues = self.validator.validate(df, "TEST")
        self.assertFalse(passed)
        self.assertGreater(result["negative_prices"], 0)
        self.assertGreater(result["negative_volume"], 0)


class TestChronologicalSplitter(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
        self.df = pd.DataFrame({
            "Date": self.dates,
            "Open": 100.0,
            "High": 105.0,
            "Low": 95.0,
            "Close": 102.0,
            "Volume": 1000,
            "Symbol": "TEST"
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chronological_split_strict_boundaries(self):
        splitter = ChronologicalSplitter(
            train_end_date="2023-12-31", 
            validation_end_date="2024-12-31",
            test_start_date="2025-01-01"
        )
        train_df, val_df, test_df = splitter.split(self.df)

        self.assertTrue((train_df["Date"] <= "2023-12-31").all())
        self.assertTrue(((val_df["Date"] > "2023-12-31") & (val_df["Date"] <= "2024-12-31")).all())
        self.assertTrue((test_df["Date"] >= "2025-01-01").all())
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(self.df))

        # Check save functionality
        res = splitter.save_splits(train_df, val_df, test_df, self.temp_dir)
        self.assertTrue((self.temp_dir / "train.parquet").exists())
        self.assertTrue((self.temp_dir / "validation.parquet").exists())
        self.assertTrue((self.temp_dir / "test.parquet").exists())


if __name__ == "__main__":
    unittest.main()
