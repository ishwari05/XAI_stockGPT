import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from ..config.loader import ConfigLoader
from ..models.registry import ModelRegistry
from ..explainability.shap import ShapExplainer
from ..explainability.feature_importance import extract_top_features
from ..explainability.faithfulness import evaluate_feature_masking_faithfulness


class ExplanationPipeline:
    """
    Generic SHAP/XAI pipeline that works with any trained prediction model.
    """

    def __init__(self, model_name: str, config_loader: Optional[ConfigLoader] = None):
        self.model_name = model_name
        self.config_loader = config_loader or ConfigLoader()
        self.output_dir = self.config_loader.resolve_path("data/explanations") / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.config_loader.resolve_path("reports_tables") / "xai"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        exp_cfg = self.config_loader.get_experiment_config()
        self.default_sample_size = exp_cfg.get("pipeline", {}).get("xai", {}).get("sample_size", 5000)

    def run(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        if sample_size is None:
            sample_size = self.default_sample_size
        # 1. Load trained model
        models_dir = self.config_loader.resolve_path("models/prediction") / self.model_name
        model_cls = ModelRegistry.get(self.model_name)
        model_cfg = self.config_loader.get_model_config(self.model_name)
        model_inst = model_cls(model_cfg)
        
        if (models_dir / "model.json").exists() or (models_dir / "model.joblib").exists():
            model_inst.load(models_dir)

        # 2. Load feature data
        features_dir = self.config_loader.resolve_path("data/features")
        test_path = features_dir / "test.parquet"
        if not test_path.exists():
            raise FileNotFoundError(f"Feature dataset not found at {test_path}")
            
        test_df = pd.read_parquet(test_path)
        if sample_size and len(test_df) > sample_size:
            test_df = test_df.iloc[:sample_size].copy()

        non_feature_cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
        target_cols = [c for c in test_df.columns if c.startswith("target_")]
        drop_cols = list(set(non_feature_cols + target_cols))

        feature_cols = [c for c in test_df.columns if c not in drop_cols]
        X = test_df[feature_cols]

        # 3. Compute SHAP explanations
        explainer = ShapExplainer(
            model_inst,
            explainer_type=model_cfg.get("model", {}).get("explainer_type", "tree"),
        )
        shap_values = explainer.explain(X)

        # Handle multiclass or single-class SHAP dimensions
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # 4. Global Feature Importance
        global_importance = sorted(
            [{"feature": f, "importance": float(imp)} for f, imp in zip(feature_cols, mean_abs_shap)],
            key=lambda x: x["importance"],
            reverse=True
        )

        # 5. Faithfulness test on top feature vs control feature
        faithfulness_results = {}
        if len(global_importance) >= 2 and hasattr(model_inst, "predict_proba"):
            top_feat = global_importance[0]["feature"]
            ctrl_feat = global_importance[-1]["feature"]
            faithfulness_results = evaluate_feature_masking_faithfulness(
                model_inst, X.iloc[[0]], top_feat, ctrl_feat
            )

        # Save artifacts
        with open(self.output_dir / "global_feature_importance.json", "w") as f:
            json.dump(global_importance, f, indent=2)

        with open(self.output_dir / "faithfulness_metrics.json", "w") as f:
            json.dump(faithfulness_results, f, indent=2)

        return {
            "status": "success",
            "model": self.model_name,
            "top_features": global_importance[:5],
            "faithfulness": faithfulness_results
        }

