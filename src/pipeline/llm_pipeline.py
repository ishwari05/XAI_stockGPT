from typing import Dict, Any, Optional
from pathlib import Path
from ..config.loader import ConfigLoader
from ..llm.trainer import FinanceLLMTrainer


class LLMPipeline:
    """
    Generic LLM Fine-Tuning & Evaluation Pipeline.
    Runs for ANY configured Finance LLM (Qwen, FinGPT, etc.) without duplicate training scripts.
    """

    def __init__(self, llm_name: str, config_loader: Optional[ConfigLoader] = None):
        self.llm_name = llm_name
        self.config_loader = config_loader or ConfigLoader()
        self.llm_cfg = self.config_loader.get_llm_config(llm_name)
        self.output_dir = self.config_loader.resolve_path("models_llm") / llm_name

    def run(self) -> Dict[str, Any]:
        llm_dataset_dir = self.config_loader.resolve_path("data_llm")
        train_path = llm_dataset_dir / "train.jsonl"
        val_path = llm_dataset_dir / "validation.jsonl"

        trainer = FinanceLLMTrainer(self.llm_cfg)
        res = trainer.train(train_path, val_path, self.output_dir)

        return {
            "status": "success",
            "llm_model": self.llm_name,
            "device": trainer.device,
            "training_summary": res
        }

