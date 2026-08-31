#!/usr/bin/env python3
"""
Convert v1 CSV splits in data/final/ to v2 parquet layout for the training pipeline.

Reads:
  data/final/train.csv, validation.csv, test.csv
  data/final/llm_{train,validation,test_locked}.jsonl  (optional)
  data/final/test_predictions.csv                        (optional)

Writes:
  data/features/{train,validation,test}.parquet
  data/features/feature_metadata.json
  data/llm/{train,validation,test}.jsonl                 (optional copy)
  data/predictions/test_predictions.parquet              (optional copy)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

LEAKY_COLUMNS = ["target", "next_day_return"]
METADATA_COLUMNS = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
TARGET_RENAME = {"target_encoded": "target_cls_h1"}
SPLITS = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate v1 data/final CSV splits to v2 parquet layout."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "final",
        help="Directory containing train/validation/test CSV files.",
    )
    parser.add_argument(
        "--output-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "features",
        help="Output directory for parquet feature files.",
    )
    parser.add_argument(
        "--output-llm",
        type=Path,
        default=PROJECT_ROOT / "data" / "llm",
        help="Output directory for LLM jsonl files.",
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=PROJECT_ROOT / "data" / "predictions",
        help="Output directory for prediction artifacts.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip copying LLM jsonl files.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Skip copying test prediction files.",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.0075,
        help="Forward return threshold for BUY label (default: 0.0075 = +0.75%%).",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=-0.0075,
        help="Forward return threshold for SELL label (default: -0.0075 = -0.75%%).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned actions without writing files.",
    )
    return parser.parse_args()


def validate_source(source: Path) -> None:
    missing = [source / f"{split}.csv" for split in SPLITS if not (source / f"{split}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required CSV splits:\n  " + "\n  ".join(str(p) for p in missing)
        )


def feature_column_names(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in METADATA_COLUMNS and not c.startswith("target_")
    ]


def prepare_dataframe(
    df: pd.DataFrame,
    split: str,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    if "next_day_return" in df.columns:
        forward_return = df["next_day_return"]
        df["target_cls_h1"] = np.select(
            [forward_return >= buy_threshold, forward_return <= sell_threshold],
            [2, 0],
            default=1,
        )
    elif "target_encoded" in df.columns:
        df = df.rename(columns=TARGET_RENAME)
    else:
        raise ValueError(
            f"{split}: need 'next_day_return' or 'target_encoded' to build targets."
        )

    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    if "Ticker" not in df.columns and "Symbol" in df.columns:
        df = df.rename(columns={"Symbol": "Ticker"})

    if not feature_column_names(df):
        raise ValueError(f"{split}: no feature columns remain after preprocessing.")

    return df


def write_feature_metadata(
    output_dir: Path,
    split_stats: dict[str, dict],
    feature_names: list[str],
    buy_threshold: float,
    sell_threshold: float,
) -> None:
    metadata = {
        "pipeline_stage": "v1_to_v2_migration",
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "source_format": "data/final/*.csv",
        "target_column": "target_cls_h1",
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "splits": split_stats,
    }
    metadata_path = output_dir / "feature_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def migrate_split_csv(
    source: Path,
    output_dir: Path,
    split: str,
    dry_run: bool,
    buy_threshold: float,
    sell_threshold: float,
) -> tuple[dict, list[str]]:
    csv_path = source / f"{split}.csv"
    parquet_path = output_dir / f"{split}.parquet"

    print(f"  [{split}] reading {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    raw_rows = len(df)

    df = prepare_dataframe(df, split, buy_threshold, sell_threshold)
    feature_cols = feature_column_names(df)

    target_dist = df["target_cls_h1"].value_counts().to_dict()

    stats = {
        "source_csv": str(csv_path),
        "output_parquet": str(parquet_path),
        "rows": len(df),
        "columns": len(df.columns),
        "feature_count": len(feature_cols),
        "tickers": int(df["Ticker"].nunique()) if "Ticker" in df.columns else None,
        "date_start": str(df["Date"].min()) if "Date" in df.columns else None,
        "date_end": str(df["Date"].max()) if "Date" in df.columns else None,
        "dropped_leaky_columns": [c for c in LEAKY_COLUMNS if c in pd.read_csv(csv_path, nrows=0).columns],
        "raw_rows": raw_rows,
        "target_distribution": {str(k): int(v) for k, v in target_dist.items()},
    }

    if dry_run:
        print(f"    would write {parquet_path} ({stats['rows']:,} rows, {stats['columns']} cols)")
        return stats, feature_cols

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    size_mb = parquet_path.stat().st_size / 1024 / 1024
    print(f"    wrote {parquet_path.name} ({stats['rows']:,} rows, {size_mb:.1f} MB)")
    return stats, feature_cols


def copy_llm_files(source: Path, output_dir: Path, dry_run: bool) -> None:
    mapping = {
        "llm_train.jsonl": "train.jsonl",
        "llm_validation.jsonl": "validation.jsonl",
        "llm_test_locked.jsonl": "test.jsonl",
    }

    print("Copying LLM datasets ...")
    for src_name, dst_name in mapping.items():
        src = source / src_name
        dst = output_dir / dst_name
        if not src.exists():
            print(f"  skip {src_name} (not found)")
            continue
        if dry_run:
            print(f"  would copy {src_name} -> {dst}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  copied {dst_name} ({size_mb:.1f} MB)")


def copy_predictions(source: Path, output_dir: Path, dry_run: bool) -> None:
    candidates = ["test_predictions.csv", "test_predictions_phase8.csv"]
    src = next((source / name for name in candidates if (source / name).exists()), None)
    if src is None:
        print("No test prediction CSV found; skipping.")
        return

    dst = output_dir / "test_predictions.parquet"
    print(f"Copying predictions from {src.name} ...")
    if dry_run:
        print(f"  would write {dst}")
        return

    df = pd.read_csv(src)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  wrote {dst.name} ({len(df):,} rows, {size_mb:.1f} MB)")


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output_features = args.output_features.resolve()

    print("=== XAI-StockGPT v1 → v2 Data Migration ===")
    print(f"Source:           {source}")
    print(f"Features output:  {output_features}")
    print(f"LLM output:       {args.output_llm.resolve()}")
    print(f"Predictions output: {args.output_predictions.resolve()}")
    print(f"Target thresholds: BUY >= {args.buy_threshold}, SELL <= {args.sell_threshold}")
    if args.dry_run:
        print("Mode: DRY RUN (no files will be written)")
    print()

    try:
        validate_source(source)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    split_stats: dict[str, dict] = {}
    feature_names: list[str] = []

    print("Migrating feature splits ...")
    for split in SPLITS:
        stats, cols = migrate_split_csv(
            source, output_features, split, args.dry_run,
            args.buy_threshold, args.sell_threshold,
        )
        split_stats[split] = stats
        if not feature_names:
            feature_names = cols

    if not args.dry_run:
        write_feature_metadata(
            output_features, split_stats, feature_names,
            args.buy_threshold, args.sell_threshold,
        )
        print(f"  wrote feature_metadata.json ({len(feature_names)} features)")

    if not args.skip_llm:
        copy_llm_files(source, args.output_llm.resolve(), args.dry_run)

    if not args.skip_predictions:
        copy_predictions(source, args.output_predictions.resolve(), args.dry_run)

    print()
    print("Migration complete.")
    print("Next steps:")
    print("  python scripts/train.py --model xgboost")
    print("  python scripts/explain.py --model xgboost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
