import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import ConfigLoader
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.explanation_pipeline import ExplanationPipeline
from src.pipeline.llm_pipeline import LLMPipeline
from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.llm.dataset_builder import LLMDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator for XAI-StockGPT v2")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "data", "features", "targets", "train", "explain",
            "llm_dataset", "llm_train", "benchmark", "all",
        ],
        help="Pipeline stage to execute",
    )
    parser.add_argument("--model", type=str, default="xgboost", help="Prediction model (for train/explain)")
    parser.add_argument("--llm_model", type=str, default="qwen", help="LLM model (for llm_train)")
    parser.add_argument("--config", type=str, default="experiments/default.yaml", help="Experiment config")
    args = parser.parse_args()

    loader = ConfigLoader(PROJECT_ROOT)
    print(f"=== Running XAI-StockGPT v2 Pipeline: Stage [{args.stage}] ===")

    if args.stage == "data":
        print("[1] Executing Data Collection Pipeline...")
        data_pipe = DataPipeline(config_loader=loader)
        print(f"    Data pipeline: {data_pipe.run()}")

    if args.stage in ["features", "targets", "all"]:
        splits_dir = loader.resolve_path("data/splits")
        if (splits_dir / "train.parquet").exists():
            print("[2] Executing Feature & Target Pipeline...")
            feature_pipe = FeaturePipeline(config_loader=loader)
            print(f"    Feature pipeline: {feature_pipe.run()}")
        elif args.stage in ["features", "targets"]:
            raise FileNotFoundError(
                f"No splits found in {splits_dir}. Run data collection first or use migrate_data.py."
            )
        else:
            print("[2] Skipping feature pipeline (using existing data/features/*.parquet).")

    if args.stage in ["train", "all"]:
        print(f"[3] Training Prediction Model: {args.model}...")
        train_pipe = TrainingPipeline(model_name=args.model, config_loader=loader)
        print(f"    Training pipeline: {train_pipe.run()}")

    if args.stage in ["explain", "all"]:
        print(f"[4] Generating Explanations for Model: {args.model}...")
        exp_pipe = ExplanationPipeline(model_name=args.model, config_loader=loader)
        print(f"    Explanation pipeline: {exp_pipe.run()}")

    if args.stage in ["llm_dataset", "all"]:
        print("[5] Building LLM Instruction Dataset...")
        cfg = loader.get_experiment_config()
        builder = LLMDatasetBuilder(cfg)
        
        # Load prediction test/validation outputs or raw features to generate dataset records
        features_dir = loader.resolve_path("data/features")
        split_data = {}
        for split in ["train", "validation"]:
            path = features_dir / f"{split}.parquet"
            if path.exists():
                # We can load a subset to generate instructions efficiently
                import pandas as pd
                df = pd.read_parquet(path)
                # Sample train/val count from configuration
                limit = cfg.get("pipeline", {}).get("llm_dataset", {}).get(f"{split}_samples", 5000)
                split_data[split] = df.head(min(limit, len(df)))
                
        # Handle test_locked
        test_path = features_dir / "test.parquet"
        if test_path.exists():
            import pandas as pd
            df = pd.read_parquet(test_path)
            limit = cfg.get("pipeline", {}).get("llm_dataset", {}).get("test_samples", 5000)
            split_data["test_locked"] = df.head(min(limit, len(df)))

        print(f"    LLM dataset pipeline: {builder.build_dataset(split_data, loader.resolve_path('data_llm'))}")

    if args.stage in ["llm_train", "all"]:
        print(f"[6] Fine-Tuning Finance LLM: {args.llm_model}...")
        llm_pipe = LLMPipeline(llm_name=args.llm_model, config_loader=loader)
        print(f"    LLM pipeline: {llm_pipe.run()}")

    if args.stage in ["benchmark", "all"]:
        print("[7] Running Consolidated Benchmarks...")
        eval_pipe = EvaluationPipeline(config_loader=loader)
        print(f"    Evaluation pipeline: {eval_pipe.run()}")

    print("=== Pipeline Stage Complete ===")


if __name__ == "__main__":
    main()
