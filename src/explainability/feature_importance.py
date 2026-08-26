from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd


def extract_top_features(feature_names: List[str], class_shap: np.ndarray, 
                         raw_values: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Extracts top-k features ranked by absolute SHAP contribution for a single sample.
    """
    items = []
    for f, s, v in zip(feature_names, class_shap, raw_values):
        items.append({
            "feature": f,
            "shap_value": float(s),
            "raw_value": float(v) if not pd.isna(v) else 0.0,
            "abs_shap": float(abs(s))
        })
    items.sort(key=lambda x: x["abs_shap"], reverse=True)
    return items[:top_k]
