import unittest
from pathlib import Path
import tempfile
from src.config.loader import ConfigLoader
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.explanation_pipeline import ExplanationPipeline
from src.pipeline.llm_pipeline import LLMPipeline
from src.pipeline.evaluation_pipeline import EvaluationPipeline


class TestPipelines(unittest.TestCase):

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.loader = ConfigLoader(self.project_root)

    def test_config_loader(self):
        proj_cfg = self.loader.get_project_config()
        self.assertEqual(proj_cfg["project"]["name"], "XAI-StockGPT")

    def test_llm_pipeline_init(self):
        pipe = LLMPipeline("qwen", self.loader)
        res = pipe.run()
        self.assertEqual(res["status"], "success")

    def test_evaluation_pipeline_init(self):
        pipe = EvaluationPipeline(self.loader)
        res = pipe.run()
        self.assertEqual(res["status"], "success")


if __name__ == "__main__":
    unittest.main()
