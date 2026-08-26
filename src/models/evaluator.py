from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


class ModelEvaluator:
    """Standardized classification evaluator across all prediction models."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "mcc": mcc,
            "confusion_matrix": cm,
            "classification_report": report
        }

    def evaluate_per_ticker(self, y_true: pd.Series, y_pred: np.ndarray, 
                            tickers: pd.Series) -> Dict[str, Dict[str, Any]]:
        ticker_metrics = {}
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "ticker": tickers})
        
        for ticker, group in df.groupby("ticker"):
            if len(group) == 0:
                continue
            metrics = self.evaluate(group["y_true"].values, group["y_pred"].values)
            ticker_metrics[str(ticker)] = metrics
            
        return ticker_metrics

