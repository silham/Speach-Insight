# predict_role.py (robust version)
import os
os.environ["OMP_NUM_THREADS"] = "1"
import re
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from .feature_pipeline import RoleFeatureExtractor
from .agentic_router import HybridRoleRouter

# --- Paths (adjust if needed) ---
MODEL_DIR = Path(__file__).parent / "models"

# --- Lazy Initialization Helpers ---
_router_inst = None
_extractor = None

def get_router():
    global _router_inst
    if _router_inst is None:
        calibrated_model_path = MODEL_DIR / "role_classifier_calibrated.pkl"
        model_name = "role_classifier_calibrated.pkl" if calibrated_model_path.exists() else "role_classifier.pkl"
        
        try:
            print("📦 Initializing HybridRoleRouter...")
            _router_inst = HybridRoleRouter(model_dir=str(MODEL_DIR), model_name=model_name, threshold=0.80)
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize HybridRoleRouter ({e}). Falling back to local-only mode.")
            # Graceful degradation class
            class LocalDegradedRouter:
                def __init__(self):
                    self.threshold = 0.80
                    plain_path = MODEL_DIR / "role_classifier.pkl"
                    cal_path = MODEL_DIR / "role_classifier_calibrated.pkl"
                    self.xgb_model = joblib.load(cal_path if cal_path.exists() else plain_path)
                    self.label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
                def _llm_fallback(self, text, role, conf):
                    return (role, "XGBoost (LLM Unavailable)", conf)
            _router_inst = LocalDegradedRouter()
    return _router_inst

def get_extractor():
    global _extractor
    if _extractor is None:
        print("📦 Initializing RoleFeatureExtractor...")
        _extractor = RoleFeatureExtractor(model_dir=str(MODEL_DIR))
    return _extractor

# --- Helpers ---
def _find_evidence_sentence(full_text: str, role: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", full_text.strip())
    if not sentences:
        return full_text
    
    extractor = get_extractor()
    role_lower = str(role).lower()
    if role_lower == "manager":
        scores = [len(extractor.hard_re.findall(s)) + len(extractor.soft_re.findall(s)) for s in sentences]
    elif role_lower == "junior":
        scores = [len(extractor.uncertainty_re.findall(s)) for s in sentences]
    else:
        scores = [len(s.split()) for s in sentences]
        
    best_idx = int(np.argmax(scores))
    return sentences[best_idx].strip()

# --- Main API function ---
def predict_role(transcript_segments, use_llm=True):
    """
    Predicts the speaker roles for diarized transcript segments with diagnostic metadata.
    """
    if not transcript_segments:
        return {}

    # Aggregate by speaker
    speaker_texts = {}
    for seg in transcript_segments:
        if not isinstance(seg, dict):
            continue
        spk = seg.get("speaker_id")
        txt = seg.get("text", "")
        if not spk:
            continue
        speaker_texts.setdefault(spk, []).append(str(txt).strip())

    speaker_full_texts = {spk: " ".join(txts).strip() for spk, txts in speaker_texts.items()}

    results = {}
    
    # Initialize components lazily
    router_inst = get_router()
    extractor = get_extractor()
    
    model = router_inst.xgb_model
    label_encoder = router_inst.label_encoder
    
    calibrated_model_path = MODEL_DIR / "role_classifier_calibrated.pkl"
    model_name = "role_classifier_calibrated.pkl" if calibrated_model_path.exists() else "role_classifier.pkl"

    feature_order = getattr(model, "feature_names_in_", None)
    if feature_order is not None:
        feature_order = [str(c) for c in feature_order]

    feature_count = len(feature_order) if feature_order is not None else 40

    for spk, full_text in speaker_full_texts.items():
        t_start = time.perf_counter()

        # 1. Feature extraction
        X_row = extractor.transform(full_text, feature_names=feature_order)

        # 2. XGBoost Prediction
        proba = model.predict_proba(X_row)[0]
        xgb_pred_idx = int(np.argmax(proba))
        xgb_role = label_encoder.inverse_transform([xgb_pred_idx])[0]
        xgb_confidence = float(proba[xgb_pred_idx])

        # 3. Router decision
        source = "XGBoost"
        llm_used = False
        llm_confidence = "N/A"
        fallback_reason = None
        predicted_role = xgb_role
        probability = xgb_confidence
        llm_role = None

        if xgb_confidence < router_inst.threshold:
            # Fallback to LLM
            llm_used = True
            fallback_reason = "Confidence below routing threshold"
            
            # Use the LLM fallback from the router
            llm_role, fallback_source, _ = router_inst._llm_fallback(full_text, xgb_role, xgb_confidence)
            predicted_role = llm_role
            source = "Gemini" if "Gemini" in fallback_source else fallback_source
            
            if "Fallback Error" in fallback_source:
                llm_used = False
                fallback_reason = f"LLM API call failed ({fallback_source})"
                source = "XGBoost (Fallback Error)"
                predicted_role = xgb_role
            elif "LLM Unavailable" in fallback_source:
                llm_used = False
                fallback_reason = "Gemini API key is not configured locally"
                source = "XGBoost (Local Fallback)"
                predicted_role = xgb_role

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        evidence = [_find_evidence_sentence(full_text, predicted_role)]

        # Normalize predicted role to canonical set: Leader, HR, Junior, Other
        def normalize_role(role_str):
            if not role_str:
                return "Other"
            r = str(role_str).strip().lower()
            if r in ("manager", "leader", "lead"):
                return "Leader"
            elif r == "hr":
                return "HR"
            elif r == "junior":
                return "Junior"
            return "Other"

        final_role = normalize_role(predicted_role)
        final_xgb_role = normalize_role(xgb_role)
        final_llm_role = normalize_role(llm_role) if llm_used else None
        
        xgb_probs_canonical = {
            normalize_role(label_encoder.inverse_transform([i])[0]): float(p)
            for i, p in enumerate(proba)
        }

        results[spk] = {
            "role": final_role,
            "probability": float(probability),
            "probs": xgb_probs_canonical,
            "evidence": evidence,
            "prediction_details": {
                "source": source,
                "llm_used": llm_used,
                "router_threshold": float(router_inst.threshold),
                "xgb_confidence": float(xgb_confidence),
                "llm_confidence": llm_confidence,
                "fallback_reason": fallback_reason,
                "model_name": model_name,
                "feature_count": int(feature_count),
                "inference_time_ms": float(t_elapsed_ms)
            },
            # Prediction Provenance Schema
            "speaker": spk,
            "xgboost": {
                "role": final_xgb_role,
                "confidence": xgb_confidence,
                "probabilities": xgb_probs_canonical
            },
            "gemini": {
                "used": llm_used,
                "role": final_llm_role
            },
            "final_role": final_role,
            "prediction_source": "gemini" if llm_used else "xgboost"
        }

    return results

# CLI demo
if __name__ == "__main__":
    segments = [
        {"speaker_id": "spk_1", "text": "Morning everyone. Let’s keep it quick—what did you do yesterday?"},
        {"speaker_id": "spk_1", "text": "Sure, check the /docs/api folder. And feel better!"},
        {"speaker_id": "spk_2", "text": "Yesterday I finished the auth middleware. Today I’ll start rate-limiting."},
        {"speaker_id": "spk_3", "text": "I was out sick yesterday. I’m not sure where the doc files live—can someone point me?"},
    ]
    print(json.dumps(predict_role(segments), indent=2))
