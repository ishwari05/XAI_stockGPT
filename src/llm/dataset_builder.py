from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import pandas as pd
import numpy as np

from .formatter import PromptFormatter, SYSTEM_PROMPT


class BaseLLMDatasetBuilder(ABC):
    """Abstract interface for LLM instruction dataset generation."""

    @abstractmethod
    def build_dataset(self, split_data: Dict[str, Any], output_dir: Path) -> Dict[str, int]:
        pass


class LLMDatasetBuilder(BaseLLMDatasetBuilder):
    """
    Standardized instruction dataset builder.
    Constructs JSONL records for instruction fine-tuning across multiple task types:
    SIGNAL_EXPLANATION, FEATURE_CONTRIBUTION, CONFIDENCE_EXPLANATION,
    CONFLICTING_INDICATORS, ERROR_EXPLANATION, COMPLETE_ANALYST.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def build_dataset(self, split_data: Dict[str, Any], output_dir: Path) -> Dict[str, int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        counts = {"train": 0, "validation": 0, "test_locked": 0}

        task_types = [
            "SIGNAL_EXPLANATION",
            "FEATURE_CONTRIBUTION",
            "CONFIDENCE_EXPLANATION",
            "CONFLICTING_INDICATORS",
            "ERROR_EXPLANATION",
            "COMPLETE_ANALYST",
        ]

        for split_name in ["train", "validation", "test_locked"]:
            if split_name not in split_data or split_data[split_name] is None:
                continue

            records = []
            df = split_data[split_name]
            
            for idx, row in df.iterrows():
                task_type = task_types[idx % len(task_types)]
                prediction = str(row.get("prediction", "BUY"))
                proba = row.get("probabilities", {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1})
                top_shap = row.get("top_shap_features", [
                    {"feature": "RSI_14", "raw_value": 65.2, "shap_value": 0.45},
                    {"feature": "SMA_20_vs_SMA_50", "raw_value": 0.025, "shap_value": 0.35}
                ])
                tech_ind = row.get("technical_indicators", {"RSI_14": 65.2, "SMA_20": 150.0})

                prompt_user = PromptFormatter.format_input(
                    task_type=task_type,
                    prediction=prediction,
                    confidence="HIGH" if max(proba.values()) > 0.6 else "MEDIUM",
                    probabilities=proba,
                    top_shap_features=top_shap,
                    technical_indicators=tech_ind
                )

                response_obj = {
                    "signal": prediction,
                    "confidence": "HIGH" if max(proba.values()) > 0.6 else "MEDIUM",
                    "why": f"Model generated {prediction} signal primarily driven by top technical indicators: {', '.join([f['feature'] for f in top_shap[:2]])}.",
                    "key_drivers": [f["feature"] for f in top_shap[:3]],
                    "conflicts_risks": [],
                    "model_interpretation": f"High confidence attribution based on {task_type} analysis."
                }

                records.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_user},
                        {"role": "assistant", "content": json.dumps(response_obj, indent=2)}
                    ]
                })

            out_file = output_dir / f"{split_name}.jsonl"
            with open(out_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            counts[split_name] = len(records)

        return counts

