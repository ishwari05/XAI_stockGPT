import json
from typing import Dict, Any, Optional
from pathlib import Path
from ..config.loader import ConfigLoader
from ..benchmark.prediction import PredictionBenchmark
from ..benchmark.llm import LLMBenchmark
from ..benchmark.explanation import ExplanationBenchmark
from ..benchmark.reports import ResearchReportGenerator


class EvaluationPipeline:
    """Consolidated benchmarking pipeline for prediction models, XAI, and LLMs."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.config_loader = config_loader or ConfigLoader()
        self.tables_dir = self.config_loader.resolve_path("reports_tables")
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.final_reports_dir = self.config_loader.resolve_path("reports_final")
        self.final_reports_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        pred_bench = PredictionBenchmark(self.tables_dir)
        llm_bench = LLMBenchmark(self.tables_dir)
        xai_bench = ExplanationBenchmark(self.tables_dir)
        reporter = ResearchReportGenerator(self.final_reports_dir)

        # 1. Aggregate Prediction Models Metrics
        models_dir = self.config_loader.resolve_path("models/prediction")
        collected_pred_metrics = {}

        if models_dir.exists():
            for m_dir in models_dir.iterdir():
                if m_dir.is_dir() and (m_dir / "metrics.json").exists():
                    try:
                        with open(m_dir / "metrics.json", "r") as f:
                            data = json.load(f)
                            # Get test metrics or validation metrics
                            m = data.get("test", data.get("validation", data))
                            collected_pred_metrics[m_dir.name] = m
                    except Exception:
                        pass

        if collected_pred_metrics:
            pred_bench.aggregate_results(collected_pred_metrics)

        # 2. Aggregate XAI Faithfulness Metrics
        explanations_dir = self.config_loader.resolve_path("data/explanations")
        if explanations_dir.exists():
            for exp_dir in explanations_dir.iterdir():
                if exp_dir.is_dir() and (exp_dir / "faithfulness_metrics.json").exists():
                    try:
                        with open(exp_dir / "faithfulness_metrics.json", "r") as f:
                            faith_data = json.load(f)
                            faith_data["model"] = exp_dir.name
                            xai_bench.generate_report(faith_data)
                    except Exception:
                        pass

        # 3. Generate Final Summary Report
        summary_path = reporter.generate_final_summary({
            "status": "COMPLETED",
            "evaluated_models": list(collected_pred_metrics.keys()),
            "output_tables_directory": str(self.tables_dir)
        })

        return {
            "status": "success",
            "models_evaluated": list(collected_pred_metrics.keys()),
            "summary_report": str(summary_path)
        }

