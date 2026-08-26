import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import ConfigLoader
from src.pipeline.evaluation_pipeline import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(description="Run consolidated benchmark reports.")
    parser.add_argument("--config", type=str, default="experiments/default.yaml", help="Experiment config")
    args = parser.parse_args()

    loader = ConfigLoader(PROJECT_ROOT)
    pipeline = EvaluationPipeline(config_loader=loader)
    res = pipeline.run()
    print(f"Benchmark Evaluation status: {res}")


if __name__ == "__main__":
    main()
