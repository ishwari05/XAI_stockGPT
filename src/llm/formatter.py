from typing import Dict, Any, List


SYSTEM_PROMPT = """You are an expert quantitative financial analyst explaining algorithmic trading signals.
Your job is to faithfully explain WHY the predictive model generated a BUY, HOLD, or SELL signal based EXCLUSIVELY on provided technical indicators, class probabilities, and SHAP feature attributions.

Rules:
1. NEVER invent or hallucinate news, macroeconomic rumors, earnings, or ungrounded external data.
2. NEVER claim technical indicators guarantee or cause future price moves.
3. Faithfully align the explanation with the model's actual prediction and SHAP rankings.
"""


class PromptFormatter:
    """Standardized prompt and instruction formatter for Finance LLM training and inference."""

    @staticmethod
    def format_input(task_type: str, prediction: str, confidence: str, 
                     probabilities: Dict[str, float], top_shap_features: List[Dict[str, Any]], 
                     technical_indicators: Dict[str, float]) -> str:
        prompt_lines = [
            f"TASK: {task_type}",
            f"MODEL_PREDICTION: {prediction}",
            f"CONFIDENCE_LEVEL: {confidence}",
            f"PROBABILITIES: {probabilities}",
            "TOP_SHAP_FEATURES:",
        ]
        for f in top_shap_features:
            prompt_lines.append(f"  - {f['feature']}: value={f['raw_value']:.4f}, contribution={f['shap_value']:+.4f}")

        prompt_lines.append("KEY_TECHNICAL_INDICATORS:")
        for k, v in list(technical_indicators.items())[:8]:
            prompt_lines.append(f"  - {k}: {v:.4f}")

        return "\n".join(prompt_lines)
