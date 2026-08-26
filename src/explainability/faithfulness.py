from typing import Dict, Any, List
import numpy as np
import pandas as pd


def evaluate_feature_masking_faithfulness(model: Any, X_row: pd.DataFrame, 
                                         top_feature: str, control_feature: str) -> Dict[str, Any]:
    """
    Evaluates attribution faithfulness via feature masking perturbation tests.
    Neutralizes the top-attributed feature vs a random/control non-top feature,
    measuring the drop in predicted confidence.
    """
    base_proba = model.predict_proba(X_row)[0]
    pred_class = int(np.argmax(base_proba))
    base_conf = float(base_proba[pred_class])

    # 1. Perturb top feature
    X_pert_top = X_row.copy()
    X_pert_top[top_feature] = 0.0
    top_proba = model.predict_proba(X_pert_top)[0]
    top_conf = float(top_proba[pred_class])

    # 2. Perturb control feature
    X_pert_ctrl = X_row.copy()
    X_pert_ctrl[control_feature] = 0.0
    ctrl_proba = model.predict_proba(X_pert_ctrl)[0]
    ctrl_conf = float(ctrl_proba[pred_class])

    return {
        "baseline_confidence": base_conf,
        "top_feature_perturbed": top_feature,
        "top_perturbed_confidence": top_conf,
        "top_confidence_drop": base_conf - top_conf,
        "control_feature_perturbed": control_feature,
        "control_perturbed_confidence": ctrl_conf,
        "control_confidence_drop": base_conf - ctrl_conf,
        "faithfulness_ratio": (base_conf - top_conf) / max(1e-5, (base_conf - ctrl_conf))
    }
