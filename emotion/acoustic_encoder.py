"""
Phase 2 — Acoustic Encoder
Extracts acoustic and paralinguistic features from audio using Wav2Vec 2.0.
Uses the `superb/wav2vec2-base-superb-er` encoder (pre-trained on emotion recognition),
plus optional interpretable features (pitch, energy, speaking rate) via librosa.
"""

import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np

ACOUSTIC_MODEL_NAME = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE = 16000

# Downloaded weights go here — never touches final_model/
_EMOTION_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_EMOTION_MODELS_DIR, exist_ok=True)


class AcousticEncoder(nn.Module):
    """
    Wraps a Wav2Vec 2.0 encoder pre-trained for emotion recognition.

    Outputs
    -------
    pooled : Tensor [batch, 768]
        Mean-pooled hidden states from the last encoder layer.
    frame_features : Tensor [batch, seq_len, 768]
        Full frame-level hidden states (for cross-attention).
    paralinguistic : Tensor [batch, 3]
        Hand-crafted features: [pitch_mean, rms_energy, speaking_rate].
    """

    def __init__(self, model_name: str = ACOUSTIC_MODEL_NAME, device: str | None = None):
        super().__init__()
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        print(f"⏳ Loading Acoustic Encoder ({model_name})...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, cache_dir=_EMOTION_MODELS_DIR
        )
        self.encoder = Wav2Vec2Model.from_pretrained(
            model_name, cache_dir=_EMOTION_MODELS_DIR
        )
        self.encoder.eval()
        self.encoder.to(self.device)
        print("✅ Acoustic Encoder loaded.")

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_audio(audio_path: str) -> torch.Tensor:
        """Load an audio file and return a mono 16 kHz waveform tensor [1, samples]."""
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    # ------------------------------------------------------------------
    # Paralinguistic (hand-crafted) features
    # ------------------------------------------------------------------
    @staticmethod
    def extract_paralinguistic(waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute interpretable acoustic features from a mono 16 kHz waveform.

        Returns Tensor [3] — [pitch_mean, rms_energy, speaking_rate_proxy].
        """
        import librosa

        y = waveform.squeeze().numpy().astype(np.float32)

        # Pitch (fundamental frequency) via YIN
        try:
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=SAMPLE_RATE)
            # Filter out unvoiced frames (f0 == fmax is typically unvoiced)
            voiced = f0[f0 < 500]
            pitch_mean = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
        except Exception:
            pitch_mean = 0.0

        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        rms_energy = float(np.mean(rms))

        # Speaking rate proxy: number of energy onsets per second
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SAMPLE_RATE)
            duration_sec = len(y) / SAMPLE_RATE
            speaking_rate = len(onsets) / max(duration_sec, 0.1)
        except Exception:
            speaking_rate = 0.0

        return torch.tensor([pitch_mean, rms_energy, speaking_rate], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run Wav2Vec2 encoder on a mono 16 kHz waveform [1, samples].

        Returns
        -------
        pooled : Tensor [1, 768]
        frame_features : Tensor [1, seq_len, 768]
        """
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        outputs = self.encoder(input_values, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state          # [1, seq_len, 768]
        pooled = hidden_states.mean(dim=1)                 # [1, 768]

        return pooled.cpu(), hidden_states.cpu()

    # ------------------------------------------------------------------
    # High-level convenience
    # ------------------------------------------------------------------
    def encode(self, audio_path: str) -> dict:
        """
        Full pipeline: load audio → encoder forward → paralinguistic features.

        Returns dict with keys: pooled, frame_features, paralinguistic.
        """
        waveform = self.load_audio(audio_path)
        pooled, frame_features = self.forward(waveform)
        paralinguistic = self.extract_paralinguistic(waveform)
        return {
            "pooled": pooled,                         # [1, 768]
            "frame_features": frame_features,         # [1, seq_len, 768]
            "paralinguistic": paralinguistic,         # [3]
        }
