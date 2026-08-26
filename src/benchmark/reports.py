from pathlib import Path
from typing import Dict, Any


class ResearchReportGenerator:
    """Compiles consolidated research-paper tables and summary text reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_final_summary(self, summary_data: Dict[str, Any]) -> Path:
        out_file = self.output_dir / "research_summary_report.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("      XAI-STOCKGPT v2: RESEARCH PAPER SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            for k, v in summary_data.items():
                f.write(f"{k}: {v}\n")
        return out_file
