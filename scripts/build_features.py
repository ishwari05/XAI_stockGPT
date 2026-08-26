import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import ConfigLoader
from src.pipeline.feature_pipeline import FeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("build_features")


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering & Target Generation (Phase 2).")
    parser.add_argument(
        "--config", 
        type=str, 
        default="data/features.yaml", 
        help="Path to features configuration YAML (relative to configs/ directory)"
    )
    args = parser.parse_args()

    loader = ConfigLoader(PROJECT_ROOT)
    logger.info(f"Loading configuration from configs/{args.config}...")
    pipeline = FeaturePipeline(config_loader=loader, config_path=args.config)

    meta = pipeline.run()
    splits = meta.get("splits", {})

    train_info = splits.get("train", {})
    val_info = splits.get("validation", {})
    test_info = splits.get("test", {})

    train_dist = train_info.get("classification_distribution", {})

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Train rows        : {train_info.get('final_rows', 0):,}")
    print(f"Validation rows   : {val_info.get('final_rows', 0):,}")
    print(f"Test rows         : {test_info.get('final_rows', 0):,}")
    print()
    print(f"Features generated: {meta.get('feature_count', 0)}")
    print(f"Features list     : {', '.join(meta.get('feature_names', []))}")
    print(f"NaN/Warmup dropped: Train={train_info.get('warmup_rows_dropped', 0)}, Val={val_info.get('warmup_rows_dropped', 0)}, Test={test_info.get('warmup_rows_dropped', 0)}")
    print()
    print("Classification distribution (Train):")
    print(f"  BUY  : {train_dist.get('BUY', 0):,} ({train_dist.get('BUY', 0)/max(1, train_info.get('final_rows', 1)):.1%})")
    print(f"  HOLD : {train_dist.get('HOLD', 0):,} ({train_dist.get('HOLD', 0)/max(1, train_info.get('final_rows', 1)):.1%})")
    print(f"  SELL : {train_dist.get('SELL', 0):,} ({train_dist.get('SELL', 0)/max(1, train_info.get('final_rows', 1)):.1%})")
    print()
    print("Leakage checks:")
    leakage_status = "PASSED" if all(s.get("leakage_check") == "PASSED" for s in splits.values()) else "FAILED"
    print(f"  Status: {leakage_status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
