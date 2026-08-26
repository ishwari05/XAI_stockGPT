from typing import List, Dict, Any
from pathlib import Path
import pandas as pd


class PredictionBenchmark:
    """
    Standardized benchmark across all predictive ML/DL models.
    Produces prediction_model_comparison.csv matching research-paper standards.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_results(self, model_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for model_name, m in model_metrics.items():
            rep = m.get("classification_report", {})
            rows.append({
                "Model": model_name,
                "Accuracy": m.get("accuracy", 0.0),
                "Balanced_Accuracy": m.get("balanced_accuracy", 0.0),
                "Macro_F1": m.get("macro_f1", 0.0),
                "MCC": m.get("mcc", 0.0),
                "BUY_F1": rep.get("2", rep.get("BUY", {})).get("f1-score", 0.0),
                "SELL_F1": rep.get("0", rep.get("SELL", {})).get("f1-score", 0.0),
                "HOLD_F1": rep.get("1", rep.get("HOLD", {})).get("f1-score", 0.0),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "prediction_model_comparison.csv", index=False)
        return df
