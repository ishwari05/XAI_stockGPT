from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from ..utils.hardware import detect_compute_device


class BaseLLMTrainer(ABC):
    """Abstract interface for LLM Fine-Tuning."""

    @abstractmethod
    def train(self, train_path: Path, val_path: Path, output_dir: Path) -> Dict[str, Any]:
        pass


class FinanceLLMTrainer(BaseLLMTrainer):
    """
    Hardware-Agnostic Finance LLM PEFT/LoRA Fine-Tuning Trainer.
    Automatically resolves compute accelerator (CUDA > Accelerator > CPU) without hardcoding assumptions.
    """

    def __init__(self, llm_config: Dict[str, Any]):
        self.config = llm_config
        req_dev = self.config.get("training", {}).get("device", "auto")
        self.device, self.device_info = detect_compute_device(req_dev)

    def train(self, train_path: Path, val_path: Path, output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Training orchestration placeholder
        return {
            "model_name": self.config.get("model", {}).get("name"),
            "device": self.device,
            "status": "ready_for_training",
            "epochs": self.config.get("training", {}).get("num_train_epochs", 3)
        }
