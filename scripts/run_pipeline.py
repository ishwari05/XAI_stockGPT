import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import ConfigLoader
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.explanation_pipeline import ExplanationPipeline
from src.pipeline.llm_pipeline import LLMPipeline
from src.pipeline.evaluation_pipeline import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator for XAI-StockGPT v2")
    parser.add_argument(
        "--stage", 
        type=str, 
        default="all",
        choices=["data", "features", "targets", "train", "explain", "llm_dataset", "llm_train", "benchmark", "all"],
        help="Pipeline stage to execute"
    )
    parser.add_argument("--model", type=str, default="xgboost", help="Prediction model (for train/explain)")
    parser.add_argument("--llm_model", type=str, default="qwen", help="LLM model (for llm_train)")
    parser.add_argument("--config", type=str, default="experiments/default.yaml", help="Experiment config")
    args = parser.parse_args()

    loader = ConfigLoader(PROJECT_ROOT)
    print(f"=== Running XAI-StockGPT v2 Pipeline: Stage [{args.stage}] ===")

    if args.stage in ["data", "features", "targets", "all"]:
        print("[1] Executing Data, Feature & Target Pipeline...")
        data_pipe = DataPipeline(config_loader=loader)
        res = data_pipe.run()
        print(f"    Data pipeline: {res}")

    if args.stage in ["train", "all"]:
        print(f"[2] Training Prediction Model: {args.model}...")
        train_pipe = TrainingPipeline(model_name=args.model, config_loader=loader)
        res = train_pipe.run()
        print(f"    Training pipeline: {res}")

    if args.stage in ["explain", "all"]:
        print(f"[3] Generating Explanations for Model: {args.model}...")
        exp_pipe = ExplanationPipeline(model_name=args.model, config_loader=loader)
        res = exp_pipe.run()
        print(f"    Explanation pipeline: {res}")

    if args.stage in ["llm_train", "all"]:
        print(f"[4] Fine-Tuning Finance LLM: {args.llm_model}...")
        llm_pipe = LLMPipeline(llm_name=args.llm_model, config_loader=loader)
        res = llm_pipe.run()
        print(f"    LLM pipeline: {res}")

    if args.stage in ["benchmark", "all"]:
        print("[5] Running Consolidated Benchmarks...")
        eval_pipe = EvaluationPipeline(config_loader=loader)
        res = eval_pipe.run()
        print(f"    Evaluation pipeline: {res}")

    print("=== Pipeline Stage Complete ===")


if __name__ == "__main__":
    main()
