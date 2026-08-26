import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import ConfigLoader
from src.pipeline.explanation_pipeline import ExplanationPipeline


def main():
    parser = argparse.ArgumentParser(description="Run SHAP and XAI explanations.")
    parser.add_argument("--model", type=str, required=True, help="Model name to explain (e.g. xgboost, lightgbm)")
    args = parser.parse_args()

    loader = ConfigLoader(PROJECT_ROOT)
    pipeline = ExplanationPipeline(model_name=args.model, config_loader=loader)
    res = pipeline.run()
    print(f"Explanation status for {args.model}: {res}")


if __name__ == "__main__":
    main()
