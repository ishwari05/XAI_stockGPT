import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

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
        t0 = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Resolve Model ID based on hardware
            model_cfg = self.config.get("model", {})
            model_id = model_cfg.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
            fallback_model_id = model_cfg.get("fallback_model_id", "Qwen/Qwen2.5-1.5B-Instruct")

            # Local run device override (e.g. mps or cpu)
            if self.device in ["mps", "cpu"]:
                # MPS/CPU cannot handle 7B or even 1.5B models easily for fine-tuning
                # We override to a tiny model (0.5B) for local sanity/run unless cuda is active.
                model_id = "Qwen/Qwen2.5-0.5B-Instruct"
                print(f"[FinanceLLMTrainer] Local device detected ({self.device}). Overriding model to: {model_id}")

            # 2. Quantization configurations (BitsAndBytes) - Only supported on CUDA
            bnb_config = None
            torch_dtype = torch.float32
            if self.device.startswith("cuda"):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                torch_dtype = torch.float16
            elif self.device == "mps" and self.device_info.get("precision_support", {}).get("bf16", False):
                torch_dtype = torch.bfloat16
            elif self.device == "mps":
                torch_dtype = torch.float16

            # 3. Load Tokenizer & Model
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto" if self.device.startswith("cuda") else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            
            # Map CPU/MPS manually if device map isn't applied
            if not self.device.startswith("cuda"):
                model = model.to(self.device)

            # 4. LoRA PEFT Configuration
            peft_cfg = self.config.get("peft", {})
            lora_config = LoraConfig(
                r=peft_cfg.get("r", 16),
                lora_alpha=peft_cfg.get("lora_alpha", 32),
                target_modules=peft_cfg.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=peft_cfg.get("lora_dropout", 0.05),
                bias=peft_cfg.get("bias", "none"),
                task_type=peft_cfg.get("task_type", TaskType.CAUSAL_LM),
            )

            # 5. Load and format Datasets
            def format_prompts(batch):
                texts = []
                for messages in batch["messages"]:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    texts.append(text)
                return {"text": texts}

            dataset_dict = {}
            for name, path in [("train", train_path), ("validation", val_path)]:
                if path.exists():
                    ds = load_dataset("json", data_files=str(path), split="train")
                    ds = ds.map(format_prompts, batched=True, remove_columns=ds.column_names)
                    dataset_dict[name] = ds

            train_dataset = dataset_dict.get("train")
            eval_dataset = dataset_dict.get("validation")

            # 6. Training Configuration Arguments
            t_cfg = self.config.get("training", {})
            
            # Use appropriate precision parameters based on device info
            precision_support = self.device_info.get("precision_support", {})
            fp16 = precision_support.get("fp16", False) if t_cfg.get("fp16") == "auto" else bool(t_cfg.get("fp16", False))
            bf16 = precision_support.get("bf16", False) if t_cfg.get("bf16") == "auto" else bool(t_cfg.get("bf16", False))

            training_args = SFTConfig(
                output_dir=str(output_dir),
                learning_rate=t_cfg.get("learning_rate", 2e-4),
                num_train_epochs=t_cfg.get("num_train_epochs", 3),
                per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 4),
                per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 4),
                gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
                warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
                weight_decay=t_cfg.get("weight_decay", 0.01),
                logging_steps=t_cfg.get("logging_steps", 50),
                eval_steps=t_cfg.get("eval_steps", 250),
                save_steps=t_cfg.get("save_steps", 250),
                eval_strategy="steps" if eval_dataset else "no",
                fp16=fp16,
                bf16=bf16,
                report_to="none",
                save_total_limit=1,
                dataset_text_field="text",
                max_length=t_cfg.get("max_seq_length", 1024),
            )

            # 7. SFTTrainer Run
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=lora_config,
                processing_class=tokenizer,
                args=training_args,
            )

            train_result = trainer.train()
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            duration = round(time.time() - t0, 2)
            metrics = train_result.metrics

            # Save metadata
            meta_path = output_dir / "training_metadata.json"
            meta_data = {
                "model_id_used": model_id,
                "epochs_completed": t_cfg.get("num_train_epochs", 3),
                "training_duration_seconds": duration,
                "train_samples": len(train_dataset) if train_dataset else 0,
                "train_loss": metrics.get("train_loss", None),
                "eval_loss": metrics.get("eval_loss", None),
            }
            with open(meta_path, "w") as f:
                json.dump(meta_data, f, indent=2)

            return {
                "model_name": self.config.get("model", {}).get("name"),
                "device": self.device,
                "status": "completed",
                "epochs": t_cfg.get("num_train_epochs", 3),
                "train_loss": metrics.get("train_loss", None),
                "eval_loss": metrics.get("eval_loss", None),
                "training_duration_seconds": duration,
                "output_adapter_path": str(output_dir),
                "num_examples_trained": len(train_dataset) if train_dataset else 0,
            }

        except Exception as e:
            return {
                "model_name": self.config.get("model", {}).get("name"),
                "device": self.device,
                "status": "failed",
                "error_message": str(e),
                "training_duration_seconds": round(time.time() - t0, 2),
            }

