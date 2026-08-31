import pandas as pd
import shutil
from pathlib import Path

v1_dir = Path("/Users/ishwaripandit05/Documents/xai_stockgpt/data/final")
v2_features_dir = Path("/Users/ishwaripandit05/Documents/xai-stockgpt-v2/data/features")
v2_llm_dir = Path("/Users/ishwaripandit05/Documents/xai-stockgpt-v2/data/llm")

v2_features_dir.mkdir(parents=True, exist_ok=True)
v2_llm_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "validation", "test"]:
    print(f"Migrating {split} data...")
    df = pd.read_csv(v1_dir / f"{split}.csv")
    
    # Rename target_encoded to target_cls_h1
    if "target_encoded" in df.columns:
        df = df.rename(columns={"target_encoded": "target_cls_h1"})
        
    # Drop leaky columns if they exist
    cols_to_drop = ["target", "next_day_return"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Save as parquet
    df.to_parquet(v2_features_dir / f"{split}.parquet", index=False)

for split in ["train", "validation"]:
    print(f"Migrating LLM {split} data...")
    shutil.copy2(v1_dir / f"llm_{split}.jsonl", v2_llm_dir / f"{split}.jsonl")

print("Migration complete!")
