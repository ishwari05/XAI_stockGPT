# XAI-StockGPT v2 — Architecture & Research System

> A configuration-driven, hardware-agnostic research framework combining Classical Predictive ML (XGBoost, LightGBM, Random Forest, Neural Baselines), Explainable AI (SHAP), and Fine-Tuned Finance LLMs (Qwen, FinGPT) for faithful natural-language trading signal rationale generation.

---

## 1. Architectural Overview

```
MARKET DATA (NSE / Yahoo Finance)
      ↓
DATA PIPELINE (Cleaning, Validation, Deduplication)
      ↓
FEATURE ENGINEERING (Trend, Momentum, Volatility, Volume, Returns)
      ↓
TARGET GENERATION (Forward Return Horizon, Configurable Thresholds)
      ↓
CHRONOLOGICAL SPLITTING (Train / Validation / Locked Test)
      ↓
PREDICTION MODEL PIPELINE (XGBoost / LightGBM / Random Forest / Neural)
      ↓
PREDICTIONS + PROBABILITIES
      ↓
SHAP EXPLANATIONS (TreeExplainer / Exact Attributions / Faithfulness Tests)
      ↓
STRUCTURED EXPLANATION DATASET (JSONL Multi-Task Instruction Tuning)
      ↓
FINANCE LLM FINE-TUNING (LoRA / PEFT across Qwen, FinGPT, etc.)
      ↓
STRUCTURED NATURAL-LANGUAGE EXPLANATION GENERATION
      ↓
STANDARDIZED RESEARCH BENCHMARKS & PAPER REPORTS
```

---

## 2. Core Design Principles

1. **Strictly Configuration-Driven**: Adding a new predictive model requires only a YAML file under `configs/models/` without writing duplicate training/evaluation code.
2. **Hardware-Agnostic**: Detects available compute accelerators automatically (CUDA GPU prioritized, followed by other accelerators, with CPU fallback). No hardcoded assumptions or platform lock-in.
3. **Strict Temporal Integrity**: Chronological train/validation/test splitting guarantees zero lookahead leakage. The test split is **locked** and strictly isolated from model selection and tuning.
4. **Faithful by Construction**: Grounded NLG over SHAP feature attributions ensures every generated rationale clause is mathematically traceable to model internals, strictly evaluated against hallucination and causal-claim filters.

---

## 3. Directory Layout

```
xai-stockgpt-v2/
├── configs/                   # Configuration files
│   ├── project.yaml           # Global project settings and paths
│   ├── data/
│   │   ├── market.yaml        # Market data sources, tickers, validation rules
│   │   └── features.yaml      # Technical indicator & target parameters
│   ├── models/                # Prediction model configurations
│   │   ├── xgboost.yaml
│   │   ├── lightgbm.yaml
│   │   ├── random_forest.yaml
│   │   └── neural_baseline.yaml
│   ├── llms/                  # Finance LLM configurations
│   │   ├── qwen.yaml
│   │   ├── fingpt.yaml
│   │   └── other_finance_llm.yaml
│   └── experiments/
│       └── default.yaml       # Master benchmark orchestration settings
│
├── data/                      # Artifact directories (partitioned)
│   ├── raw/                   # Raw downloaded OHLCV files
│   ├── interim/               # Intermediate cleaned stages
│   ├── processed/             # Processed datasets (train/val/locked test)
│   ├── features/              # Feature matrix stores (Parquet)
│   ├── predictions/           # Model inference output files
│   ├── explanations/          # SHAP attribution arrays and metadata
│   └── llm/                   # Instruction-tuning datasets (train/val/test JSONL)
│
├── src/                       # Modular source code
│   ├── config/loader.py       # Configuration parser & path resolver
│   ├── data/                  # Collector, cleaner, validator, splitter
│   ├── features/              # Technical, momentum, volatility, return builders
│   ├── targets/               # Classification & regression target generators
│   ├── models/                # Registry, base classes, predictors, trainer, evaluator
│   ├── explainability/        # SHAP explainers, feature importance, faithfulness
│   ├── llm/                   # Dataset builder, formatter, trainer, evaluator, inference
│   ├── benchmark/             # Prediction, XAI, and LLM benchmarking tables
│   ├── pipeline/              # Orchestrator pipelines per stage
│   └── utils/hardware.py      # Hardware detection and accelerator selector
│
├── experiments/               # Experiment tracking logs and configs
├── models/                    # Serialized checkpoints
│   ├── prediction/            # Saved classical/neural models
│   └── llm/                   # Saved LoRA adapter weights
├── reports/                   # Research-paper outputs
│   ├── tables/                # Benchmark comparison CSV tables
│   ├── figures/               # High-res diagnostic & XAI plots
│   └── final/                 # Final consolidated summary reports
├── scripts/                   # CLI entry points
│   ├── collect_data.py
│   ├── build_features.py
│   ├── train.py
│   ├── explain.py
│   ├── build_llm_dataset.py
│   ├── train_llm.py
│   ├── benchmark.py
│   └── run_pipeline.py        # Master CLI runner
├── tests/                     # Automated unit test suite
├── requirements.txt           # Production dependencies
└── README.md
```

---

## 4. How to Add a New Prediction Model

Adding a new prediction model requires **zero duplicate pipeline code**:

1. **Create Configuration**: Add `configs/models/<model_name>.yaml`:
   ```yaml
   model:
     name: "catboost"
     family: "tree_ensemble"
     type: "classifier"
     explainer_type: "tree_shap"
   hyperparameters:
     iterations: 300
     learning_rate: 0.05
     depth: 4
   ```

2. **Register Model (if new architecture)** in `src/models/predictors.py`:
   ```python
   @ModelRegistry.register("catboost")
   class CatBoostModel(BasePredictionModel):
       def fit(self, X_train, y_train, X_val=None, y_val=None):
           ...
       def predict(self, X):
           ...
       def predict_proba(self, X):
           ...
   ```

3. **Train & Evaluate**:
   ```bash
   python scripts/train.py --model catboost
   ```

---

## 5. How to Add a New Finance LLM

1. **Create Configuration**: Add `configs/llms/<llm_name>.yaml`:
   ```yaml
   model:
     name: "llama3_finance"
     model_id: "meta-llama/Llama-3.2-3B-Instruct"
     family: "instruction"
   peft:
     method: "lora"
     r: 16
     lora_alpha: 32
     target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
   training:
     learning_rate: 0.0002
     num_train_epochs: 3
     device: "auto"
   ```

2. **Fine-tune & Benchmark**:
   ```bash
   python scripts/train_llm.py --model llama3_finance
   ```

---

## 6. Pipeline Execution Commands

```bash
# 1. Market Data Collection
python scripts/collect_data.py --config data/market.yaml

# 2. Feature & Target Generation
python scripts/build_features.py --config data/features.yaml

# 3. Train Prediction Models
python scripts/train.py --model xgboost
python scripts/train.py --model lightgbm
python scripts/train.py --model random_forest

# 4. Generate SHAP Explanations & Faithfulness Tests
python scripts/explain.py --model xgboost

# 5. Build Standardized LLM Instruction Dataset
python scripts/build_llm_dataset.py --config experiments/default.yaml

# 6. Fine-Tune Finance LLMs
python scripts/train_llm.py --model qwen
python scripts/train_llm.py --model fingpt

# 7. Run Research Benchmarks & Compile Tables
python scripts/benchmark.py --config experiments/default.yaml

# OR Run End-to-End via Master Orchestrator:
python scripts/run_pipeline.py --stage all --model xgboost --llm_model qwen
```
