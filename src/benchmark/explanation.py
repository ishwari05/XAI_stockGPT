from typing import Dict, Any
from pathlib import Path
import pandas as pd


class ExplanationBenchmark:
    """Benchmark for evaluating XAI attribution quality and faithfulness."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, faithfulness_results: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([faithfulness_results])
        df.to_csv(self.output_dir / "xai_faithfulness_benchmark.csv", index=False)
        return df
