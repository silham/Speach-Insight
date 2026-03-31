"""
Multimodal Emotion Recognition Package
=======================================
Phase 1 — Exposes a single ``EmotionAnalyzer`` façade that orchestrates:

* **AcousticEncoder**  — Wav2Vec 2.0 (superb/wav2vec2-base-superb-er)
* **LinguisticEncoder** — BERT-base-uncased
* **VaderAnalyzer**     — VADER lexicon-based sentiment
* **CrossAttentionFusion** + **MultimodalEmotionClassifier**

Usage
-----
>>> from emotion import EmotionAnalyzer
>>> analyzer = EmotionAnalyzer()
>>> result = analyzer.analyze("path/to/clip.wav", "hello how are you")
>>> result["emotion"]      # e.g. "happy"
>>> result["confidence"]   # e.g. 0.73
>>> result["sarcasm"]      # True / False
"""

from __future__ import annotations

from .acoustic_encoder import AcousticEncoder
from .linguistic_encoder import LinguisticEncoder
from .vader_analyzer import VaderAnalyzer
from .emotion_classifier import MultimodalEmotionClassifier, EMOTION_LABELS


class EmotionAnalyzer:
    """
    High-level façade for multimodal emotion recognition.

    Parameters
    ----------
    checkpoint_path : str | None
        Path to a trained fusion checkpoint.  If ``None`` the system uses
        the zero-shot ensemble fallback (works out of the box).
    device : str | None
        Torch device string.  Auto-detected if ``None``.
    """

    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        print("\n" + "=" * 55)
        print("  Multimodal Emotion Recognition — Initialising")
        print("=" * 55)

        self.acoustic = AcousticEncoder(device=device)
        self.linguistic = LinguisticEncoder(device=device)
        self.vader = VaderAnalyzer()
        self.classifier = MultimodalEmotionClassifier(
            use_zero_shot=(checkpoint_path is None),
            checkpoint_path=checkpoint_path,
        )

        mode = "zero-shot ensemble" if self.classifier.use_zero_shot else "trained fusion"
        print(f"✅ Emotion Analyzer ready  (mode: {mode})\n")

    def analyze(self, audio_path: str, text: str) -> dict:
        """
        Run full multimodal emotion analysis on one utterance.

        Parameters
        ----------
        audio_path : str
            Path to a mono/stereo WAV file.
        text : str
            Transcript of the utterance (lowercase is fine).

        Returns
        -------
        dict
            emotion         : str     — dominant Ekman label
            confidence      : float   — probability of dominant label
            all_emotions    : dict    — {label: prob} for all 7 classes
            sarcasm         : bool    — sarcasm detected?
            sarcasm_score   : float   — raw sarcasm probability
            ambiguity_score : float   — 0-1 normalised Shannon entropy
            vader           : dict    — raw VADER scores {pos, neg, neu, compound}
            paralinguistic  : dict    — {pitch, energy, speaking_rate}
        """
        # 1. Acoustic features
        acoustic_out = self.acoustic.encode(audio_path)

        # 2. Linguistic features
        linguistic_out = self.linguistic.encode(text)

        # 3. VADER sentiment
        vader_out = self.vader.analyze(text)

        # 4. Classify via fusion (or zero-shot)
        result = self.classifier.predict(
            text=text,
            audio_pooled=acoustic_out["pooled"],
            audio_frames=acoustic_out["frame_features"],
            text_cls=linguistic_out["cls_embedding"],
            text_tokens=linguistic_out["token_features"],
            vader_features=vader_out["tensor"],
        )

        # 5. Attach auxiliary info
        para = acoustic_out["paralinguistic"]
        result["vader"] = vader_out["scores"]
        result["paralinguistic"] = {
            "pitch": round(float(para[0]), 2),
            "energy": round(float(para[1]), 6),
            "speaking_rate": round(float(para[2]), 2),
        }

        return result
