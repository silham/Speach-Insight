import os
import re
import numpy as np
import joblib
from textblob import TextBlob
from typing import List, Optional

class RoleFeatureExtractor:
    """
    Deterministic Feature Extractor for Speaker Role Classification.
    Extracts linguistic, syntactic, and TF-IDF features from raw text.
    """
    def __init__(self, model_dir: str = "models"):
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        svd_path = os.path.join(model_dir, "tfidf_svd.joblib")
        
        if not os.path.exists(tfidf_path) or not os.path.exists(svd_path):
            raise FileNotFoundError(f"Missing TF-IDF artifacts in {model_dir}")
            
        self.tfidf = joblib.load(tfidf_path)
        self.svd = joblib.load(svd_path)
        
        self.hard_directive_patterns = [
            r"\byou should\b", r"\byou need to\b", r"\bhave to\b",
            r"\bdeadline\b", r"\bby friday\b", r"\bblocker\b",
            r"\bi expect\b", r"\bensure\b"
        ]
        
        self.soft_help_patterns = [
            r"\bi['’]?ll help\b", r"\bi can help\b", r"\blet me help\b",
            r"\bwe can\b", r"\bi can walk you through\b"
        ]
        
        self.uncertainty_patterns = [
            r"\bnot sure\b", r"\bstuck\b", r"\btrying to\b",
            r"\bi think\b", r"\bmaybe\b", r"\bconfused\b"
        ]
        
        self.hard_re = re.compile("|".join(self.hard_directive_patterns), re.IGNORECASE)
        self.soft_re = re.compile("|".join(self.soft_help_patterns), re.IGNORECASE)
        self.uncertainty_re = re.compile("|".join(self.uncertainty_patterns), re.IGNORECASE)
        self.greeting_re = re.compile(
            r"^(hi|hello|hey|good morning|good afternoon|good evening)[\s!.,]*$",
            re.IGNORECASE
        )
        
        # Determine expected SVD dimension
        if hasattr(self.svd, "n_components"):
            self.expected_tfidf_dim = self.svd.n_components
        elif hasattr(self.svd, "components_"):
            self.expected_tfidf_dim = self.svd.components_.shape[0]
        else:
            self.expected_tfidf_dim = 32

    def transform(self, speaker_text: str, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Transforms a single speaker's text into the feature vector expected by XGBoost.
        """
        text = (speaker_text or "").strip()
        words = text.split()
        word_count = len(words)
        sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
        avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0
        question_count = len(re.findall(r"\?", text))
        
        sentiment_score = 0.0
        if text:
            try:
                sentiment_score = TextBlob(text).sentiment.polarity
            except Exception:
                pass

        hard_directive_count = len(self.hard_re.findall(text))
        soft_help_count = len(self.soft_re.findall(text))
        directive_count = hard_directive_count + soft_help_count
        uncertainty_count = len(self.uncertainty_re.findall(text))
        is_greeting_only = 1 if self.greeting_re.match(text) else 0
        
        base_features = {
            "word_count": float(word_count),
            "avg_sentence_len": float(avg_sentence_len),
            "question_count": float(question_count),
            "hard_directive_count": float(hard_directive_count),
            "soft_help_count": float(soft_help_count),
            "directive_count": float(directive_count),
            "uncertainty_count": float(uncertainty_count),
            "sentiment_score": float(sentiment_score),
        }
        
        # Process TF-IDF + SVD features
        vec = self.tfidf.transform([text])
        sv = self.svd.transform(vec).flatten()
        
        if len(sv) < self.expected_tfidf_dim:
            sv = np.concatenate([sv, np.zeros(self.expected_tfidf_dim - len(sv))])
        elif len(sv) > self.expected_tfidf_dim:
            sv = sv[:self.expected_tfidf_dim]
            
        tfidf_features = {f"tfidf_{i}": float(sv[i]) for i in range(self.expected_tfidf_dim)}
        all_features = {**base_features, **tfidf_features}
        
        # Order features correctly
        if feature_names is None:
            base_cols = [
                "word_count", "avg_sentence_len", "question_count",
                "hard_directive_count", "soft_help_count", "directive_count", 
                "uncertainty_count", "sentiment_score"
            ]
            tfidf_cols = [f"tfidf_{i}" for i in range(self.expected_tfidf_dim)]
            feature_names = base_cols + tfidf_cols
            
        # Ensure all columns expected by the model are present in the exact order
        X_row = np.array([[all_features.get(c, 0.0) for c in feature_names]], dtype=float)
        return X_row
