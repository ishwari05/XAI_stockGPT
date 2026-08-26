import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from ..config.loader import ConfigLoader
from ..data.collector import MarketDataCollector
from ..data.cleaner import MarketDataCleaner
from ..data.validator import MarketDataValidator
from ..data.splitter import ChronologicalSplitter

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Production-grade, configuration-driven market data pipeline:
    Market Source -> Raw Data -> Cleaning -> Validation -> Deduplication -> Metadata Generation -> Splits.
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None, config_path: str = "data/market.yaml"):
        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load_yaml(config_path)
        self.project_root = self.config_loader.root_dir

        # Initialize pipeline stages from config
        self.collector = MarketDataCollector(self.config, project_root=self.project_root)
        self.cleaner = MarketDataCleaner(self.config)
        self.validator = MarketDataValidator(self.config)

        # Storage paths
        storage_cfg = self.config.get("storage", {})
        self.raw_dir = self.project_root / storage_cfg.get("raw_path", "data/raw")
        self.processed_dir = self.project_root / storage_cfg.get("processed_path", "data/processed")
        self.metadata_dir = self.project_root / storage_cfg.get("metadata_path", "data/metadata")
        self.splits_dir = self.project_root / storage_cfg.get("splits_path", "data/splits")
        self.reports_dir = self.project_root / storage_cfg.get("reports_path", "reports/data_validation")
        self.logs_dir = self.project_root / storage_cfg.get("logs_path", "logs")

        for d in [self.raw_dir, self.processed_dir, self.metadata_dir, self.splits_dir, self.reports_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Executes the full end-to-end data collection, cleaning, validation, and splitting flow."""
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("  STARTING PHASE 1: DATA INGESTION & VALIDATION PIPELINE")
        logger.info("=" * 60)

        # 1. Collection
        raw_data, download_logs, failed_tickers = self.collector.download_all(symbols=symbols)
        raw_storage_res = self.collector.save_raw_data(raw_data)

        # 2. Cleaning & 3. Validation
        cleaned_datasets: Dict[str, pd.DataFrame] = {}
        validation_results: List[Dict[str, Any]] = []
        total_dups_removed = 0
        total_invalid_dropped = 0
        total_missing_filled = 0

        for symbol, df_raw in raw_data.items():
            # Clean
            df_clean, clean_metrics = self.cleaner.clean(df_raw, symbol)
            total_dups_removed += clean_metrics["duplicates_removed"]
            total_invalid_dropped += clean_metrics["invalid_rows_dropped"]
            total_missing_filled += clean_metrics["missing_filled"]

            # Validate
            passed, val_result, issues = self.validator.validate(df_clean, symbol)
            validation_results.append(val_result)

            if passed or not self.config.get("validation", {}).get("drop_failed_validation", False):
                cleaned_datasets[symbol] = df_clean
                # Save per-stock processed CSV
                clean_sym = symbol.replace(".NS", "").replace(".BO", "").replace("^", "")
                df_clean.to_csv(self.processed_dir / f"{clean_sym}_cleaned.csv", index=False)

        # 4. Save Validation Report & Logs
        val_report_path = self.reports_dir / "validation_report.csv"
        self.validator.generate_validation_report(validation_results, val_report_path)
        
        val_log_df = pd.DataFrame(validation_results)
        if "issues" in val_log_df.columns:
            val_log_df["issues"] = val_log_df["issues"].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
        val_log_df.to_csv(self.logs_dir / "validation_log.csv", index=False)

        # 5. Chronological Splitting (if enabled)
        split_res = {}
        split_cfg = self.config.get("splitting", {})
        if split_cfg.get("enabled", True) and cleaned_datasets:
            combined_cleaned = pd.concat(list(cleaned_datasets.values()), ignore_index=True)
            combined_cleaned.to_parquet(self.processed_dir / "combined_processed.parquet", index=False)

            splitter = ChronologicalSplitter(
                train_end_date=split_cfg.get("train_end", "2023-12-31"),
                validation_end_date=split_cfg.get("validation_end", "2024-12-31"),
                test_start_date=split_cfg.get("test_start", "2025-01-01"),
                date_col="Date"
            )
            train_df, val_df, test_df = splitter.split(combined_cleaned)
            split_res = splitter.save_splits(train_df, val_df, test_df, self.splits_dir)

        # 6. Metadata Generation
        all_dates = []
        total_rows = 0
        total_missing = sum(r["missing_values_total"] for r in validation_results)
        
        for df in cleaned_datasets.values():
            total_rows += len(df)
            if "Date" in df.columns and not df["Date"].empty:
                all_dates.extend([df["Date"].min(), df["Date"].max()])

        start_date_str = str(min(all_dates).date()) if all_dates else self.config.get("date_range", {}).get("start", "N/A")
        end_date_str = str(max(all_dates).date()) if all_dates else self.config.get("date_range", {}).get("end", "N/A")

        metadata = {
            "tickers_requested": len(self.collector.load_tickers() if symbols is None else symbols),
            "tickers_collected": len(raw_data),
            "tickers_validated_success": sum(1 for r in validation_results if r["passed"]),
            "failed_downloads_count": len(failed_tickers),
            "failed_downloads": failed_tickers,
            "total_raw_rows": raw_storage_res["total_rows"],
            "total_cleaned_rows": total_rows,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "duplicates_removed": total_dups_removed,
            "missing_values_detected": total_missing,
            "missing_values_filled": total_missing_filled,
            "invalid_rows_dropped": total_invalid_dropped,
            "split_info": split_res,
            "timestamp": pd.Timestamp.now().isoformat(),
            "pipeline_duration_sec": round(time.time() - t0, 2)
        }

        with open(self.metadata_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Phase 1 complete in {metadata['pipeline_duration_sec']}s. Metadata saved to {self.metadata_dir / 'metadata.json'}")
        return metadata
