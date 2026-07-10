import os
import warnings
import numpy as np
import joblib
from google import genai
from typing import Tuple

class HybridRoleRouter:
    """
    A router that tries a fast XGBoost prediction and falls back to an LLM
    if the confidence is below a defined threshold.
    """
    def __init__(self, model_dir: str = "models", model_name: str = "role_classifier_calibrated.pkl", threshold: float = 0.80):
        self.threshold = threshold
        
        # Load label encoder
        self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        
        # Load XGBoost model
        calibrated_model_path = os.path.join(model_dir, model_name)
        plain_model_path = os.path.join(model_dir, "role_classifier.pkl")
        
        if os.path.exists(calibrated_model_path):
            self.xgb_model = joblib.load(calibrated_model_path)
        elif os.path.exists(plain_model_path):
            self.xgb_model = joblib.load(plain_model_path)
        else:
            raise FileNotFoundError("Could not find role classifier in models directory.")
            
        # Store feature names if available in the model
        self.feature_names = getattr(self.xgb_model, "feature_names_in_", None)
        
        # Initialize Gemini LLM using the modern SDK
        # Assumes GOOGLE_API_KEY environment variable is configured in the terminal
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def predict(self, features: np.ndarray, raw_text: str) -> Tuple[str, str, float]:
        """
        Run inference using XGBoost and fallback to Gemini if max probability is below threshold.
        """
        # Predict probabilities using XGBoost
        proba = self.xgb_model.predict_proba(features)[0]
        max_prob = float(np.max(proba))
        pred_idx = int(np.argmax(proba))
        
        predicted_role = str(self.label_encoder.inverse_transform([pred_idx])[0])
        
        # Return XGBoost prediction if confident
        if max_prob >= self.threshold:
            return (predicted_role, "XGBoost", max_prob)
            
        # Fallback to LLM if not confident
        return self._llm_fallback(raw_text, predicted_role, max_prob)
        
    def _llm_fallback(self, raw_text: str, default_role: str, xgb_confidence: float) -> Tuple[str, str, float]:
        """
        Trigger Gemini fallback for classification.
        """
        if self.client is None:
            return (default_role, "XGBoost (LLM Unavailable)", xgb_confidence)

        system_prompt = (
            "You are a sociolinguist expert. Your task is to classify the speaker's role based on their "
            "conversational dominance, process-oriented language, or uncertainty.\n"
            "The available roles are: Lead, HR, Junior, Other.\n\n"
            "Roles Definitions:\n"
            "- Lead: Takes charge, gives hard directives, sets deadlines.\n"
            "- HR: Focuses on process, well-being, generic facilitation.\n"
            "- Junior: Shows uncertainty, asks for help, mentions being stuck.\n"
            "- Other: General contributions, small talk, etc.\n\n"
            "Analyze the following raw text from a meeting transcript and classify it into EXACTLY ONE of the four roles. "
            "Respond ONLY with the role name (e.g., Lead) and nothing else."
        )
        
        prompt = f"{system_prompt}\n\nRaw Text:\n\"{raw_text}\"\n\nRole:"
        
        try:
            # Generate content using the new SDK syntax
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            llm_response = response.text.strip().lower()
            
            # Robustly parse the response
            if "lead" in llm_response:
                llm_role = "Lead"
            elif "hr" in llm_response:
                llm_role = "HR"
            elif "junior" in llm_response:
                llm_role = "Junior"
            elif "other" in llm_response:
                llm_role = "Other"
            else:
                # If the LLM generates something unexpected, use the default (best XGboost)
                llm_role = default_role
                
            return (llm_role, "Gemini Fallback", xgb_confidence)
            
        except Exception as e:
            warnings.warn(f"LLM API call failed: {str(e)}. Defaulting to XGBoost prediction.")
            return (default_role, "XGBoost (Fallback Error)", xgb_confidence)