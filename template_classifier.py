"""
template_classifier.py
======================
Classifies transcript segments into appraisal-template categories using a
fine-tuned BERT model (BertForSequenceClassification).

Labels
------
Direct   — direct instruction / directive
Listen   — active listening / acknowledgement
NSuggest — negative / constructive suggestion
PSuggest — positive suggestion
Praise   — praise / positive feedback
WarmUp   — warm-up / ice-breaker
"""

from __future__ import annotations

import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "./Template_classifier_model"

# Label map must match config.json id2label
ID2LABEL = {0: "Direct", 1: "Listen", 2: "NSuggest", 3: "PSuggest", 4: "Praise", 5: "WarmUp"}


class TemplateClassifier:
    """Loads the template-classifier model once and exposes a `classify()` method."""

    def __init__(self, model_dir: str = MODEL_DIR):
        print(f"⏳ Loading Template Classifier from {model_dir}...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.model = BertForSequenceClassification.from_pretrained(model_dir)
            self.model.eval()
            print("✅ Template Classifier loaded.")
        except Exception as exc:
            print(f"❌ Template Classifier failed to load: {exc}")
            self.tokenizer = None
            self.model = None

    def classify(self, text: str) -> dict:
        """
        Classify a single transcript string.

        Returns
        -------
        dict  with keys:
            label      — predicted template label (str)
            confidence — softmax probability of the predicted label (float, 0-1)
            all_scores — {label: probability} for every class
        """
        if self.model is None or not text or not text.strip():
            return {"label": "unknown", "confidence": 0.0, "all_scores": {}}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze()
        pred_id = int(torch.argmax(probs))
        label = ID2LABEL.get(pred_id, f"class_{pred_id}")
        confidence = float(probs[pred_id])

        all_scores = {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(len(probs))}

        return {"label": label, "confidence": confidence, "all_scores": all_scores}


def load_template_classifier() -> TemplateClassifier:
    """Factory function — mirrors load_transcriber() pattern."""
    return TemplateClassifier()
