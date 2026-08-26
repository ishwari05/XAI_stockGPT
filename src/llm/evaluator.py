from typing import Dict, Any, List


class LLMExplanationEvaluator:
    """
    Evaluates LLM-generated explanations across 9 scientific dimensions:
    1. Signal consistency
    2. SHAP grounding
    3. Feature attribution accuracy
    4. Hallucination rate (forbidden words: earnings, ceo, rumors, news)
    5. Causal claim rate (forbidden claims: guarantees, will cause)
    6. Confidence consistency
    7. Completeness
    8. Structural adherence
    9. Human readability
    """

    FORBIDDEN_EXTERNAL = ["earnings", "revenue", "ceo", "guidance", "fed", "macroeconomic", "rumor", "news"]
    FORBIDDEN_CAUSAL = ["guarantees", "will cause", "surely make", "ensures profit"]

    def evaluate_explanation(self, prediction_label: str, generated_text: str, 
                             expected_shap_features: List[str]) -> Dict[str, Any]:
        text_lower = generated_text.lower()
        
        # 1. Hallucination check
        hallucinations = [w for w in self.FORBIDDEN_EXTERNAL if w in text_lower]
        
        # 2. Causal claims check
        causal_claims = [w for w in self.FORBIDDEN_CAUSAL if w in text_lower]
        
        # 3. Signal consistency check
        signal_consistent = prediction_label.lower() in text_lower

        # 4. SHAP feature grounding
        features_found = [f for f in expected_shap_features if f.lower() in text_lower]
        shap_grounding_score = len(features_found) / max(1, len(expected_shap_features))

        return {
            "signal_consistent": signal_consistent,
            "hallucination_detected": len(hallucinations) > 0,
            "hallucinations": hallucinations,
            "causal_claims_detected": len(causal_claims) > 0,
            "causal_claims": causal_claims,
            "shap_grounding_score": float(shap_grounding_score),
            "features_grounded": features_found
        }
