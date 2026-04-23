"""
Phase 6 — Multimodal Emotion Classifier
Chains all four sub-components (acoustic encoder, linguistic encoder,
VADER analyzer, cross-attention fusion) through a shared trunk and
exposes three prediction heads:

1. Emotion head   → Ekman 7 classes (angry, disgust, fear, happy, sad, surprise, neutral)
2. Sarcasm head   → binary probability
3. Ambiguity score → Shannon entropy of the emotion distribution

For inference **before training**, a zero-shot ensemble fallback is
available that blends:
  • j-hartmann/emotion-english-distilroberta-base (text-side)
  • superb/wav2vec2-base-superb-er  (audio-side, upstream logits)
  • VADER compound as a re-weighting signal

Once the cross-attention fusion head is trained on IEMOCAP (Phase 7),
set `use_zero_shot=False` and load the checkpoint.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import CrossAttentionFusion, FUSED_DIM

# Downloaded weights go here — never touches final_model/
_EMOTION_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_EMOTION_MODELS_DIR, exist_ok=True)

# --------------- label definitions ---------------
EMOTION_LABELS: list[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]
NUM_EMOTIONS = len(EMOTION_LABELS)

# Mapping from zero-shot text model labels → our Ekman 7 index
_ZS_TEXT_LABEL_MAP: dict[str, int] = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}


class EmotionHead(nn.Module):
    """Linear(512→256) → ReLU → Dropout → Linear(256→7)."""

    def __init__(self, fused_dim: int = FUSED_DIM, num_classes: int = NUM_EMOTIONS, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits [B, 7]


class SarcasmHead(nn.Module):
    """Linear(512→128) → ReLU → Linear(128→1)."""

    def __init__(self, fused_dim: int = FUSED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logit [B]


class MultimodalEmotionClassifier(nn.Module):
    """
    Complete multimodal classifier wrapping fusion + heads.

    Parameters
    ----------
    use_zero_shot : bool
        If True the fusion MLP weights are ignored and predictions come
        from a blended zero-shot ensemble.  Use this before training.
    checkpoint_path : str | None
        If provided, load trained fusion + head weights from this file.
    """

    def __init__(
        self,
        use_zero_shot: bool = True,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.use_zero_shot = use_zero_shot

        # Fusion + heads (used after training)
        self.fusion = CrossAttentionFusion()
        self.emotion_head = EmotionHead()
        self.sarcasm_head = SarcasmHead()

        # Optionally load trained weights
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"⏳ Loading emotion classifier checkpoint: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=False)
            self.use_zero_shot = False
            print("✅ Checkpoint loaded — using trained fusion.")

        # Zero-shot text pipeline (lazy-loaded)
        self._zs_text_pipe = None

    # ------------------------------------------------------------------
    # Zero-shot helpers
    # ------------------------------------------------------------------
    def _get_zs_text_pipeline(self):
        """Lazy-load the Hugging Face text-emotion pipeline."""
        if self._zs_text_pipe is None:
            from transformers import pipeline

            self._zs_text_pipe = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=-1,  # CPU — lightweight model
                model_kwargs={"cache_dir": _EMOTION_MODELS_DIR},
            )
        return self._zs_text_pipe

    def _zero_shot_predict(
        self,
        text: str,
        vader_compound: float,
        acoustic_pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Blend text-side zero-shot emotion probs with a VADER-weighted
        acoustic signal to produce an emotion distribution [7].

        Returns (probs [7], sarcasm_score float).
        """
        # --- Text side ---
        pipe = self._get_zs_text_pipeline()
        raw_preds = pipe(text[:512])[0]  # list of {label, score}

        text_probs = torch.zeros(NUM_EMOTIONS)
        for pred in raw_preds:
            label = pred["label"].lower()
            idx = _ZS_TEXT_LABEL_MAP.get(label)
            if idx is not None:
                text_probs[idx] = pred["score"]

        # Normalise in case labels didn't cover all 7
        if text_probs.sum() > 0:
            text_probs = text_probs / text_probs.sum()

        # --- Acoustic side (simple heuristic from pooled embedding norm) ---
        # High-energy embeddings correlate with anger/surprise; low with sadness/neutral
        energy = acoustic_pooled.norm().item()
        # We don't have a pre-trained acoustic emotion head in zero-shot mode,
        # so bias the text distribution slightly toward high-arousal emotions
        # when acoustic energy is high.
        arousal_bias = torch.zeros(NUM_EMOTIONS)
        if energy > 5.0:
            arousal_bias[0] = 0.10  # angry
            arousal_bias[5] = 0.05  # surprise
        else:
            arousal_bias[6] = 0.10  # neutral
            arousal_bias[4] = 0.05  # sad

        combined = text_probs + 0.15 * arousal_bias
        combined = combined / combined.sum()

        # --- Sarcasm heuristic ---
        # VADER positive but text-emotion is negative → sarcasm candidate
        dominant_idx = combined.argmax().item()
        dominant_is_negative = dominant_idx in (0, 1, 2, 4)  # angry, disgust, fear, sad
        sarcasm_score = 0.0
        if vader_compound > 0.05 and dominant_is_negative:
            sarcasm_score = min(1.0, vader_compound + 0.3)
        elif vader_compound < -0.3 and dominant_idx == 3:  # linguistically happy but tone negative
            sarcasm_score = min(1.0, abs(vader_compound) + 0.2)

        return combined, sarcasm_score

    # ------------------------------------------------------------------
    # Trained forward pass
    # ------------------------------------------------------------------
    def forward_trained(
        self,
        audio_pooled: torch.Tensor,
        audio_frames: torch.Tensor,
        text_cls: torch.Tensor,
        text_tokens: torch.Tensor,
        vader_features: torch.Tensor,
    ) -> dict:
        """
        Full forward pass through cross-attention fusion + heads.
        Used after the model has been trained.
        """
        fused = self.fusion(
            audio_pooled, audio_frames, text_cls, text_tokens, vader_features,
        )  # [B, 512]

        emotion_logits = self.emotion_head(fused)         # [B, 7]
        sarcasm_logit = self.sarcasm_head(fused)          # [B]

        emotion_probs = F.softmax(emotion_logits, dim=-1)
        sarcasm_prob = torch.sigmoid(sarcasm_logit)

        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "sarcasm_prob": sarcasm_prob,
        }

    # ------------------------------------------------------------------
    # Unified predict
    # ------------------------------------------------------------------
    def predict(
        self,
        text: str,
        audio_pooled: torch.Tensor,
        audio_frames: torch.Tensor,
        text_cls: torch.Tensor,
        text_tokens: torch.Tensor,
        vader_features: torch.Tensor,
    ) -> dict:
        """
        High-level prediction that routes through zero-shot or trained path.

        Returns
        -------
        dict with keys:
            emotion       : str          dominant label
            confidence    : float        probability of dominant label
            all_emotions  : dict         {label: prob} for all 7
            sarcasm       : bool         whether sarcasm is detected
            sarcasm_score : float        raw sarcasm probability
            ambiguity_score : float      Shannon entropy of emotion dist (normalised 0-1)
        """
        if self.use_zero_shot:
            vader_compound = vader_features[3].item() if vader_features.dim() == 1 else vader_features[0, 3].item()
            emotion_probs, sarcasm_score = self._zero_shot_predict(
                text, vader_compound, audio_pooled,
            )
        else:
            with torch.no_grad():
                out = self.forward_trained(
                    audio_pooled, audio_frames, text_cls, text_tokens, vader_features,
                )
            emotion_probs = out["emotion_probs"].squeeze(0)   # [7]
            sarcasm_score = out["sarcasm_prob"].item()

        # --- Resolve labels ---
        probs_np = emotion_probs.numpy()
        dominant_idx = int(emotion_probs.argmax())
        dominant_label = EMOTION_LABELS[dominant_idx]
        confidence = float(probs_np[dominant_idx])

        all_emotions = {label: round(float(probs_np[i]), 4) for i, label in enumerate(EMOTION_LABELS)}

        # --- Ambiguity via Shannon entropy ---
        # Max entropy for 7 classes = log2(7) ≈ 2.807
        eps = 1e-9
        entropy = -sum(float(p) * math.log2(float(p) + eps) for p in probs_np)
        max_entropy = math.log2(NUM_EMOTIONS)
        ambiguity_score = round(entropy / max_entropy, 4)

        # --- Sarcasm threshold ---
        sarcasm = sarcasm_score >= 0.5

        return {
            "emotion": dominant_label,
            "confidence": round(confidence, 4),
            "all_emotions": all_emotions,
            "sarcasm": sarcasm,
            "sarcasm_score": round(sarcasm_score, 4),
            "ambiguity_score": ambiguity_score,
        }
