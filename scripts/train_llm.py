import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import ConfigLoader
from src.pipeline.llm_pipeline import LLMPipeline


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Finance LLM with LoRA.")
    parser.add_argument("--model", type=str, required=True, help="LLM model name (e.g. qwen, fingpt)")
    args = parser.parse_args()

    loader = ConfigLoader(PROJECT_ROOT)
    pipeline = LLMPipeline(llm_name=args.model, config_loader=loader)
    res = pipeline.run()
    print(f"LLM Training status for {args.model}: {res}")


if __name__ == "__main__":
    main()
