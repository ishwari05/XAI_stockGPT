from typing import Dict, Any, List
from pathlib import Path
import pandas as pd


class LLMBenchmark:
    """
    Standardized benchmark across all fine-tuned Finance LLMs.
    Produces llm_comparison.csv matching research-paper tables.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare_models(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "llm_comparison.csv", index=False)
        return df
