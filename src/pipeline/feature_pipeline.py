import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from ..config.loader import ConfigLoader
from ..features.builder import FeatureBuilder
from ..targets.classification import ClassificationTargetGenerator
from ..targets.regression import ContinuousReturnTargetGenerator
from ..utils.leakage import verify_no_future_leakage, LeakageError

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Orchestrator for Phase 2: Feature Engineering & Target Generation.
    
    Pipeline Steps:
    1. Load Phase 1 Chronological Splits (train, validation, test)
    2. Build Technical, Volatility, Momentum, Volume, and Return Features
    3. Warm-up row tracking and removal
    4. Feature validation (NaN, Inf, ranges, data types)
    5. Generate Classification & Regression Targets
    6. Comprehensive Leakage Prevention Verification
    7. Save Final Feature Parquet datasets & Feature Metadata JSON
    8. Generate Validation CSV Report
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None, config_path: str = "data/features.yaml"):
        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load_yaml(config_path)
        self.project_root = self.config_loader.root_dir

        # Initialize builders
        self.builder = FeatureBuilder(self.config)
        
        # Target generator configs
        self.targets_cfg = self.config.get("targets", {})
        self.clf_cfg = self.targets_cfg.get("classification", {})
        self.reg_cfg = self.targets_cfg.get("regression", {})

        self.classification_generator = ClassificationTargetGenerator(
            horizon=self.clf_cfg.get("horizon", 1),
            buy_threshold=self.clf_cfg.get("buy_threshold", 0.01),
            sell_threshold=self.clf_cfg.get("sell_threshold", -0.01),
            class_labels=self.clf_cfg.get("class_labels", {0: "SELL", 1: "HOLD", 2: "BUY"})
        ) if self.clf_cfg.get("enabled", True) else None

        self.regression_generator = ContinuousReturnTargetGenerator(
            horizons=self.reg_cfg.get("horizons", [1, 5, 10])
        ) if self.reg_cfg.get("enabled", True) else None

        # Storage paths
        storage_cfg = self.config.get("storage", {})
        self.input_splits_dir = self.project_root / storage_cfg.get("input_splits_path", "data/splits")
        self.output_features_dir = self.project_root / storage_cfg.get("output_features_path", "data/features")
        self.metadata_file = self.project_root / storage_cfg.get("metadata_path", "data/features/feature_metadata.json")
        self.reports_dir = self.project_root / storage_cfg.get("reports_path", "reports/features")
        self.logs_dir = self.project_root / storage_cfg.get("logs_path", "logs")

        for d in [self.output_features_dir, self.reports_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _validate_features_dataframe(self, df: pd.DataFrame, feature_cols: List[str], split_name: str) -> List[Dict[str, Any]]:
        """Validates feature columns for NaNs, Infs, dtypes, and ranges."""
        validation_records = []
        for feat in feature_cols:
            if feat not in df.columns:
                validation_records.append({
                    "split": split_name,
                    "feature": feat,
                    "dtype": "MISSING",
                    "nan_count": len(df),
                    "inf_count": 0,
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "status": "FAIL_MISSING"
                })
                continue

            s = df[feat]
            nan_cnt = int(s.isna().sum())
            inf_cnt = int(np.isinf(s).sum()) if pd.api.types.is_numeric_dtype(s) else 0
            
            min_val = float(s.min()) if nan_cnt < len(s) and pd.api.types.is_numeric_dtype(s) else np.nan
            max_val = float(s.max()) if nan_cnt < len(s) and pd.api.types.is_numeric_dtype(s) else np.nan
            mean_val = float(s.mean()) if nan_cnt < len(s) and pd.api.types.is_numeric_dtype(s) else np.nan

            status = "PASS"
            if nan_cnt > 0 or inf_cnt > 0:
                status = "FAIL_NAN_OR_INF"

            validation_records.append({
                "split": split_name,
                "feature": feat,
                "dtype": str(s.dtype),
                "nan_count": nan_cnt,
                "inf_count": inf_cnt,
                "min": round(min_val, 4) if not np.isnan(min_val) else np.nan,
                "max": round(max_val, 4) if not np.isnan(max_val) else np.nan,
                "mean": round(mean_val, 4) if not np.isnan(mean_val) else np.nan,
                "status": status
            })
        return validation_records

    def process_split(self, split_name: str, split_path: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Processes a single chronological split file:
        loads data -> builds features -> creates targets -> runs leakage checks -> validates.
        """
        if not split_path.exists():
            raise FileNotFoundError(f"Input split file not found: {split_path}")

        logger.info(f"Processing split [{split_name}] from {split_path}...")
        df_raw = pd.read_parquet(split_path)
        initial_rows = len(df_raw)

        # 1. Feature Engineering & Warm-up handling
        df_features, feature_cols, feat_meta = self.builder.build_features(df_raw)

        # 2. Target Generation
        all_target_cols = []
        clf_dist = {}

        if self.classification_generator:
            df_features, clf_cols, clf_meta = self.classification_generator.generate(df_features)
            all_target_cols.extend(clf_cols)
            clf_dist = clf_meta.get("distribution", {})

        if self.regression_generator:
            df_features, reg_cols, reg_meta = self.regression_generator.generate(df_features)
            all_target_cols.extend(reg_cols)

        # 3. Leakage Checks
        leakage_res = verify_no_future_leakage(
            df=df_features,
            feature_columns=feature_cols,
            target_columns=all_target_cols,
            date_col="Date",
            symbol_col="Symbol",
            raise_on_error=True
        )

        # 4. Feature Validation
        val_records = self._validate_features_dataframe(df_features, feature_cols, split_name)

        split_summary = {
            "split": split_name,
            "initial_rows": initial_rows,
            "final_rows": len(df_features),
            "warmup_rows_dropped": feat_meta["warmup_rows_dropped"],
            "feature_count": len(feature_cols),
            "classification_distribution": clf_dist,
            "leakage_check": "PASSED" if leakage_res["passed"] else "FAILED",
            "start_date": str(df_features["Date"].min().date()) if not df_features.empty else "N/A",
            "end_date": str(df_features["Date"].max().date()) if not df_features.empty else "N/A",
        }

        return df_features, feature_cols, split_summary, val_records

    def run(self) -> Dict[str, Any]:
        """
        Executes the entire Phase 2 Feature Pipeline across all splits.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("  STARTING PHASE 2: FEATURE ENGINEERING & TARGET PIPELINE")
        logger.info("=" * 60)

        split_files = {
            "train": self.input_splits_dir / "train.parquet",
            "validation": self.input_splits_dir / "validation.parquet",
            "test": self.input_splits_dir / "test.parquet"
        }

        all_splits_summary = {}
        all_val_records = []
        feature_names = []
        feature_descriptions = {}

        for split_name, split_path in split_files.items():
            df_feat, feat_cols, summary, val_recs = self.process_split(split_name, split_path)
            all_splits_summary[split_name] = summary
            all_val_records.extend(val_recs)

            if not feature_names:
                feature_names = feat_cols
                feature_descriptions = self.builder.config.get("features", {})

            # Save processed feature split
            out_split_path = self.output_features_dir / f"{split_name}_features.parquet"
            df_feat.to_parquet(out_split_path, index=False)
            # Canonical name expected by TrainingPipeline
            canonical_path = self.output_features_dir / f"{split_name}.parquet"
            df_feat.to_parquet(canonical_path, index=False)
            all_splits_summary[split_name]["saved_path"] = str(canonical_path)

        # Save Feature Validation Report CSV
        val_df = pd.DataFrame(all_val_records)
        val_report_path = self.reports_dir / "feature_validation.csv"
        val_df.to_csv(val_report_path, index=False)

        # Generate Configuration Hash for Reproducibility
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:12]

        metadata = {
            "pipeline_stage": "Phase 2 - Feature Engineering & Target Generation",
            "config_version_hash": config_hash,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "feature_categories": {
                "trend": [f for f in feature_names if any(k in f for k in ["sma", "ema", "macd"])],
                "momentum": [f for f in feature_names if any(k in f for k in ["rsi", "stoch"])],
                "volatility": [f for f in feature_names if any(k in f for k in ["atr", "bb_", "volatility"])],
                "volume": [f for f in feature_names if any(k in f for k in ["volume", "obv"])],
                "returns": [f for f in feature_names if f.startswith("return_")]
            },
            "target_definitions": {
                "classification": {
                    "enabled": bool(self.clf_cfg.get("enabled", True)),
                    "target_column": "target_classification",
                    "target_label_column": "target_label",
                    "horizon": self.clf_cfg.get("horizon", 1),
                    "buy_threshold": self.clf_cfg.get("buy_threshold", 0.01),
                    "sell_threshold": self.clf_cfg.get("sell_threshold", -0.01),
                    "classes": self.clf_cfg.get("class_labels", {0: "SELL", 1: "HOLD", 2: "BUY"})
                },
                "regression": {
                    "enabled": bool(self.reg_cfg.get("enabled", True)),
                    "horizons": self.reg_cfg.get("horizons", [1, 5, 10])
                }
            },
            "splits": all_splits_summary,
            "validation_report_file": str(val_report_path),
            "timestamp": pd.Timestamp.now().isoformat(),
            "duration_seconds": round(time.time() - t0, 2)
        }

        # Save Feature Metadata JSON
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Write execution log
        log_file = self.logs_dir / "feature_generation.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{metadata['timestamp']}] Phase 2 completed successfully in {metadata['duration_seconds']}s | Features: {len(feature_names)} | Config Hash: {config_hash}\n")

        logger.info(f"Phase 2 complete in {metadata['duration_seconds']}s. Metadata saved to {self.metadata_file}")
        return metadata
