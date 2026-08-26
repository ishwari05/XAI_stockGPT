import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import ConfigLoader
from src.llm.dataset_builder import LLMDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Build structured LLM instruction dataset.")
    parser.add_argument("--config", type=str, default="experiments/default.yaml", help="Experiment config")
    args = parser.parse_args()

    loader = ConfigLoader(PROJECT_ROOT)
    cfg = loader.get_experiment_config()
    builder = LLMDatasetBuilder(cfg)
    res = builder.build_dataset({}, loader.resolve_path("data/llm"))
    print(f"LLM Dataset Building status: {res}")


if __name__ == "__main__":
    main()
