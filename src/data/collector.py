import os
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class BaseDataCollector(ABC):
    """Abstract interface for market data collection."""

    @abstractmethod
    def load_tickers(self) -> List[str]:
        pass

    @abstractmethod
    def download_ticker(self, symbol: str) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def download_all(self, symbols: Optional[List[str]] = None) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]], List[str]]:
        pass

    @abstractmethod
    def save_raw_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        pass


class MarketDataCollector(BaseDataCollector):
    """
    Configuration-driven market data collector using Yahoo Finance (or other pluggable providers).
    Handles retries with exponential backoff, rate limiting, batch tracking,
    and saves per-ticker raw files and combined datasets.
    """

    def __init__(self, config: Dict[str, Any], project_root: Optional[Path] = None):
        self.config = config
        self.project_root = project_root or Path.cwd()
        
        # Data source config
        self.ds_cfg = self.config.get("data_source", {})
        self.provider = self.ds_cfg.get("provider", "yfinance")
        self.retry_attempts = int(self.ds_cfg.get("retry_attempts", 3))
        self.backoff_factor = float(self.ds_cfg.get("backoff_factor", 2.0))
        self.delay = float(self.ds_cfg.get("request_delay_seconds", 0.5))
        self.batch_size = int(self.ds_cfg.get("batch_size", 10))

        # Tickers config
        self.tickers_cfg = self.config.get("tickers", {})
        
        # Date range config
        self.date_cfg = self.config.get("date_range", {})
        self.start_date = str(self.date_cfg.get("start", "2015-01-01"))
        self.end_date = str(self.date_cfg.get("end", "2026-01-01"))

        # Download parameters
        self.dl_cfg = self.config.get("download", {})
        self.interval = str(self.dl_cfg.get("interval", "1d"))
        self.auto_adjust = bool(self.dl_cfg.get("auto_adjust", False))

        # Storage paths
        self.storage_cfg = self.config.get("storage", {})
        self.raw_dir = self.project_root / self.storage_cfg.get("raw_path", "data/raw")
        self.logs_dir = self.project_root / self.storage_cfg.get("logs_path", "logs")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def load_tickers(self) -> List[str]:
        """Loads ticker symbols from inline list or file based on config."""
        source_type = self.tickers_cfg.get("source", "list")
        if source_type == "file":
            file_path = self.project_root / self.tickers_cfg.get("path", "")
            if not file_path.exists():
                raise FileNotFoundError(f"Tickers file not found at: {file_path}")
            col = self.tickers_cfg.get("symbol_column", "Symbol")
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
                if col in df.columns:
                    symbols = df[col].dropna().astype(str).tolist()
                else:
                    symbols = df.iloc[:, 0].dropna().astype(str).tolist()
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    symbols = [line.strip() for line in f if line.strip()]
            return sorted(list(set(symbols)))
        
        # Fallback to inline list
        symbols = self.tickers_cfg.get("symbols", [])
        return sorted(list(set(symbols)))

    def download_ticker(self, symbol: str) -> Optional[pd.DataFrame]:
        """Downloads historical OHLCV data for a single symbol with exponential backoff."""
        if self.provider.lower() == "yfinance":
            import yfinance as yf
            
            attempt = 0
            curr_delay = self.delay

            while attempt < self.retry_attempts:
                attempt += 1
                try:
                    ticker_obj = yf.Ticker(symbol)
                    df = ticker_obj.history(
                        start=self.start_date,
                        end=self.end_date,
                        interval=self.interval,
                        auto_adjust=self.auto_adjust,
                    )
                    
                    if df is not None and not df.empty:
                        df = df.reset_index()
                        # Standardize date column name
                        if "Date" in df.columns:
                            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                        elif "Datetime" in df.columns:
                            df["Date"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
                            df = df.drop(columns=["Datetime"])
                        
                        df["Symbol"] = symbol
                        return df
                    else:
                        logger.warning(f"[{symbol}] Attempt {attempt}: Empty DataFrame returned.")
                except Exception as e:
                    logger.warning(f"[{symbol}] Attempt {attempt} failed: {e}")

                if attempt < self.retry_attempts:
                    time.sleep(curr_delay)
                    curr_delay *= self.backoff_factor

            logger.error(f"[{symbol}] All {self.retry_attempts} download attempts failed.")
            return None
        else:
            raise NotImplementedError(f"Unsupported data provider: {self.provider}")

    def download_all(self, symbols: Optional[List[str]] = None) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]], List[str]]:
        """
        Downloads all tickers with batch progress tracking and error logging.
        Returns:
            data: Mapping from symbol to raw DataFrame
            logs: Download execution logs for all symbols
            failed: List of failed ticker symbols
        """
        if symbols is None:
            symbols = self.load_tickers()

        collected_data: Dict[str, pd.DataFrame] = {}
        download_logs: List[Dict[str, Any]] = []
        failed_tickers: List[str] = []

        total = len(symbols)
        logger.info(f"Starting data collection for {total} symbols...")

        for i, sym in enumerate(symbols, 1):
            t0 = time.time()
            df = self.download_ticker(sym)
            duration = time.time() - t0

            if df is not None and not df.empty:
                collected_data[sym] = df
                download_logs.append({
                    "symbol": sym,
                    "status": "SUCCESS",
                    "rows": len(df),
                    "start_date": str(df["Date"].min().date()) if "Date" in df.columns else "N/A",
                    "end_date": str(df["Date"].max().date()) if "Date" in df.columns else "N/A",
                    "duration_sec": round(duration, 2),
                    "timestamp": pd.Timestamp.now().isoformat()
                })
            else:
                failed_tickers.append(sym)
                download_logs.append({
                    "symbol": sym,
                    "status": "FAILED",
                    "rows": 0,
                    "start_date": "N/A",
                    "end_date": "N/A",
                    "duration_sec": round(duration, 2),
                    "timestamp": pd.Timestamp.now().isoformat()
                })

            if i % self.batch_size == 0 or i == total:
                logger.info(f"Progress: [{i}/{total}] symbols processed ({len(collected_data)} success, {len(failed_tickers)} failed)")

            # Polite request throttling
            time.sleep(self.delay)

        # Write download logs
        log_df = pd.DataFrame(download_logs)
        log_df.to_csv(self.logs_dir / "download_log.csv", index=False)

        if failed_tickers:
            pd.DataFrame({"symbol": failed_tickers}).to_csv(self.logs_dir / "failed_tickers.csv", index=False)

        return collected_data, download_logs, failed_tickers

    def save_raw_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Saves raw per-symbol CSV files and optionally a combined Parquet dataset."""
        save_csv = bool(self.storage_cfg.get("save_individual_csv", True))
        save_parquet = bool(self.storage_cfg.get("save_combined_parquet", True))

        saved_files = []
        combined_dfs = []

        for symbol, df in data.items():
            clean_sym = symbol.replace(".NS", "").replace(".BO", "").replace("^", "")
            if save_csv:
                csv_path = self.raw_dir / f"{clean_sym}.csv"
                df.to_csv(csv_path, index=False)
                saved_files.append(str(csv_path))
            if save_parquet:
                combined_dfs.append(df)

        combined_path = None
        total_rows = 0
        if save_parquet and combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            combined_path = self.raw_dir / "combined_raw.parquet"
            combined_df.to_parquet(combined_path, index=False)
            total_rows = len(combined_df)

        return {
            "saved_csv_count": len(saved_files),
            "combined_parquet_path": str(combined_path) if combined_path else None,
            "total_rows": total_rows
        }
