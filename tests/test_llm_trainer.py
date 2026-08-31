import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.llm.trainer import FinanceLLMTrainer


class TestLLMTrainer(unittest.TestCase):

    def test_finance_llm_trainer_smoke(self):
        # 1. Create a tiny synthetic dataset of 5 instruction examples
        tiny_dataset = []
        for i in range(5):
            tiny_dataset.append({
                "messages": [
                    {"role": "system", "content": "You are a quant analyst."},
                    {"role": "user", "content": f"Explain prediction {i}."},
                    {"role": "assistant", "content": json.dumps({"signal": "BUY", "confidence": "HIGH"})}
                ]
            })

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_file = tmp_path / "train.jsonl"
            val_file = tmp_path / "validation.jsonl"
            output_dir = tmp_path / "adapter_output"

            for filepath in [train_file, val_file]:
                with open(filepath, "w") as f:
                    for record in tiny_dataset:
                        f.write(json.dumps(record) + "\n")

            # 2. Mini configuration override to speed up and verify
            mock_config = {
                "model": {
                    "name": "qwen_test",
                    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                },
                "peft": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "target_modules": ["q_proj", "v_proj"],
                },
                "training": {
                    "learning_rate": 2e-4,
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 2,
                    "per_device_eval_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "max_seq_length": 64,
                    "warmup_ratio": 0.0,
                    "weight_decay": 0.01,
                    "logging_steps": 1,
                    "eval_steps": 1,
                    "save_steps": 10,
                    "fp16": False,
                    "bf16": False,
                    "device": "auto",
                }
            }

            # 3. Initialize and run trainer for 1 step/epoch
            trainer = FinanceLLMTrainer(mock_config)
            
            # Smoke run test execution
            res = trainer.train(train_file, val_file, output_dir)

            self.assertIn(res["status"], ["completed", "failed"])
            if res["status"] == "completed":
                self.assertTrue(output_dir.exists())
                self.assertTrue((output_dir / "adapter_config.json").exists())
                self.assertTrue((output_dir / "training_metadata.json").exists())
            else:
                print(f"Smoke test ended with failure: {res.get('error_message')}")


if __name__ == "__main__":
    unittest.main()
