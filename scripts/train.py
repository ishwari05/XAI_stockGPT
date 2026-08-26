import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import ConfigLoader
from src.pipeline.training_pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Train predictive ML model.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. xgboost, lightgbm, random_forest)")
    args = parser.parse_args()

    loader = ConfigLoader(PROJECT_ROOT)
    pipeline = TrainingPipeline(model_name=args.model, config_loader=loader)
    res = pipeline.run()
    print(f"Training status for {args.model}: {res}")


if __name__ == "__main__":
    main()
