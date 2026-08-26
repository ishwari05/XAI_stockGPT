import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigLoader:
    """
    Central configuration loader for XAI-StockGPT v2.
    Loads and merges project, data, model, LLM, and experiment configs.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            self.root_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.root_dir = Path(root_dir)
        self.config_dir = self.root_dir / "configs"

    def load_yaml(self, rel_path: str) -> Dict[str, Any]:
        """Loads a YAML file given a path relative to configs/ directory."""
        path = self.config_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_project_config(self) -> Dict[str, Any]:
        return self.load_yaml("project.yaml")

    def get_market_config(self) -> Dict[str, Any]:
        return self.load_yaml("data/market.yaml")

    def get_features_config(self) -> Dict[str, Any]:
        return self.load_yaml("data/features.yaml")

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        return self.load_yaml(f"models/{model_name}.yaml")

    def get_llm_config(self, llm_name: str) -> Dict[str, Any]:
        return self.load_yaml(f"llms/{llm_name}.yaml")

    def get_experiment_config(self, exp_name: str = "default") -> Dict[str, Any]:
        return self.load_yaml(f"experiments/{exp_name}.yaml")

    def resolve_path(self, rel_path: str) -> Path:
        """Resolves relative project paths against project root."""
        return self.root_dir / rel_path
