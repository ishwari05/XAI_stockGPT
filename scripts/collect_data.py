import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import ConfigLoader
from src.pipeline.data_pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("collect_data")


def main():
    parser = argparse.ArgumentParser(description="Collect, Clean, Validate & Split Market Data (Phase 1).")
    parser.add_argument(
        "--config", 
        type=str, 
        default="data/market.yaml", 
        help="Path to market configuration YAML (relative to configs/ directory)"
    )
    parser.add_argument(
        "--symbols", 
        nargs="+", 
        default=None, 
        help="Optional list of specific ticker symbols to override config"
    )
    args = parser.parse_args()

    loader = ConfigLoader(PROJECT_ROOT)
    logger.info(f"Loading configuration from configs/{args.config}...")
    pipeline = DataPipeline(config_loader=loader, config_path=args.config)

    metadata = pipeline.run(symbols=args.symbols)
    print("\n" + "=" * 60)
    print("      DATA COLLECTION & VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tickers Requested    : {metadata['tickers_requested']}")
    print(f"Tickers Collected    : {metadata['tickers_collected']}")
    print(f"Validated Passed     : {metadata['tickers_validated_success']}")
    print(f"Failed Downloads     : {metadata['failed_downloads_count']}")
    print(f"Total Cleaned Rows   : {metadata['total_cleaned_rows']:,}")
    print(f"Date Range           : {metadata['start_date']} -> {metadata['end_date']}")
    print(f"Duplicates Removed   : {metadata['duplicates_removed']}")
    print(f"Missing Detected     : {metadata['missing_values_detected']}")
    print(f"Execution Duration   : {metadata['pipeline_duration_sec']} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
