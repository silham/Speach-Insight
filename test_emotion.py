"""Quick smoke test for the emotion package (phases 1-6)."""
import sys

# --- Test 1: Module imports ---
print("=== Test 1: Module imports ===")
from emotion.vader_analyzer import VaderAnalyzer
from emotion.cross_attention import CrossAttentionFusion
from emotion.emotion_classifier import (
    MultimodalEmotionClassifier,
    EmotionHead,
    SarcasmHead,
    EMOTION_LABELS,
)
print(f"  Emotion labels: {EMOTION_LABELS}")
print("  OK\n")

# --- Test 2: VADER ---
print("=== Test 2: VADER ===")
v = VaderAnalyzer()
r = v.analyze("oh great another boring meeting")
print(f"  scores: {r['scores']}")
print(f"  tensor shape: {r['tensor'].shape}")
assert r["tensor"].shape == (4,), f"Expected (4,), got {r['tensor'].shape}"
print("  OK\n")

# --- Test 3: CrossAttentionFusion shapes ---
print("=== Test 3: CrossAttentionFusion ===")
import torch

fusion = CrossAttentionFusion()
B = 1
audio_pooled = torch.randn(B, 768)
audio_frames = torch.randn(B, 50, 768)
text_cls = torch.randn(B, 768)
text_tokens = torch.randn(B, 128, 768)
vader_feat = torch.randn(B, 4)
out = fusion(audio_pooled, audio_frames, text_cls, text_tokens, vader_feat)
print(f"  Fusion output shape: {out.shape}")
assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"
print("  OK\n")

# --- Test 4: Emotion & Sarcasm heads ---
print("=== Test 4: Classifier heads ===")
eh = EmotionHead()
sh = SarcasmHead()
e_out = eh(out)
s_out = sh(out)
print(f"  Emotion head: {e_out.shape}")
print(f"  Sarcasm head: {s_out.shape}")
assert e_out.shape == (1, 7), f"Expected (1, 7), got {e_out.shape}"
assert s_out.shape == (1,), f"Expected (1,), got {s_out.shape}"
print("  OK\n")

print("=" * 40)
print("ALL PHASE 1-6 SMOKE TESTS PASSED")
print("=" * 40)
