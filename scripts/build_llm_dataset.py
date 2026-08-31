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
    
    features_dir = loader.resolve_path("data/features")
    split_data = {}
    for split in ["train", "validation"]:
        path = features_dir / f"{split}.parquet"
        if path.exists():
            import pandas as pd
            df = pd.read_parquet(path)
            limit = cfg.get("pipeline", {}).get("llm_dataset", {}).get(f"{split}_samples", 5000)
            split_data[split] = df.head(min(limit, len(df)))
            
    test_path = features_dir / "test.parquet"
    if test_path.exists():
        import pandas as pd
        df = pd.read_parquet(test_path)
        limit = cfg.get("pipeline", {}).get("llm_dataset", {}).get("test_samples", 5000)
        split_data["test_locked"] = df.head(min(limit, len(df)))

    res = builder.build_dataset(split_data, loader.resolve_path("data_llm"))
    print(f"LLM Dataset Building status: {res}")


if __name__ == "__main__":
    main()
