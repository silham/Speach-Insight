# Plan: Multimodal Emotion Recognition System

**TL;DR:** Build a new `emotion/` Python package with four components — Wav2Vec2 acoustic encoder, BERT linguistic encoder, VADER sentiment layer, and a Cross-Attention Fusion classifier. Wire it into the existing `/analyze` endpoint so each segment gets `emotion`, `confidence`, `sarcasm`, and `ambiguity_score` fields. Update the React table to display them with color-coded labels. Also deliver a training pipeline (for later fine-tuning on IEMOCAP) as a parallel track.

Inference uses pre-trained HuggingFace models from day one, so the system works out of the box before any training.

---

## Steps

### Phase 1 — Package Scaffold

1. Create `emotion/__init__.py` that exposes a single `EmotionAnalyzer` class and its `analyze(audio_path, text) → dict` method.

### Phase 2 — Acoustic Encoder (`emotion/acoustic_encoder.py`)

2. Load `superb/wav2vec2-base-superb-er` (pre-trained on emotion recognition, not CTC) via `Wav2Vec2Model` (encoder only, no CTC head). The existing `final_model/` is CTC-only and unsuitable here — a separate dedicated model is needed.
3. Pass each `.wav` clip through the encoder, extract the full hidden state sequence from the last layer, apply mean pooling → `[768]` acoustic embedding per segment.
4. Optionally extract interpretable paralinguistic features (pitch via `librosa.yin`, RMS energy, speaking rate) as a `[3]` auxiliary vector that supplements the learned embedding.

### Phase 3 — Linguistic Encoder (`emotion/linguistic_encoder.py`)

5. Load `bert-base-uncased` via `BertModel`. Feed the lowercase transcript text, take the `[CLS]` token from the last hidden layer → `[768]` linguistic embedding.
6. For sarcasm/ambiguity, also retain the full token-level sequence `[seq_len, 768]` so the cross-attention can operate at token granularity rather than just the pooled vector.

### Phase 4 — VADER Wrapper (`emotion/vader_analyzer.py`)

7. Wrap `vaderSentiment.SentimentIntensityAnalyzer` into a small class that returns a normalized `[4]` tensor (positive, negative, neutral, compound). This is used as an explicit feature gate in fusion — not as a standalone predictor.

### Phase 5 — Cross-Attention Fusion (`emotion/cross_attention.py`)

8. Implement a `CrossAttentionFusion` `nn.Module` with two `nn.MultiheadAttention` heads:
   - **Audio→Text**: acoustic embedding (as Q) attends over BERT token sequence (K, V) → captures which words the acoustic tone aligns with.
   - **Text→Audio**: BERT CLS (as Q) attends over Wav2Vec2 frame sequence (K, V) → captures which acoustic frames the semantic content aligns with.
9. Concatenate both attended outputs + the 4-dim VADER vector → project through a `Linear(768+768+4 → 512)` + `LayerNorm` + `ReLU`.

### Phase 6 — Emotion Classifier (`emotion/emotion_classifier.py`)

10. Build a `MultimodalEmotionClassifier` `nn.Module` that chains all four components through a shared trunk:
    - **Emotion head**: `Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→7)` → softmax → Ekman 7 labels.
    - **Sarcasm head**: `Linear(512→128) → ReLU → Linear(128→1)` → sigmoid → sarcasm probability. Sarcasm is flagged when VADER compound is positive (`>0.05`) but the dominant acoustic emotion is negative (angry/sad/disgust) — a rule-based trigger that the trained head eventually supersedes.
    - **Ambiguity score**: computed from Shannon entropy of the 7-class softmax. High entropy → the model is uncertain → flag as ambiguous.
11. For inference before training, the classifier weights can optionally be replaced by a zero-shot ensemble: `j-hartmann/emotion-english-distilroberta-base` for text-side emotion + `superb/wav2vec2-base-superb-er` for audio-side, averaged with VADER weighting.

### Phase 7 — Training Pipeline (`emotion/training/`)

12. `emotion/training/dataset.py`: `IEMOCAPDataset` — loads IEMOCAP sessions, maps its 8 emotion labels to Ekman 7 (merge `excitement` into `happy`, drop `frustrated`/`other`), returns `(wav_path, text, label)` triples.
13. `emotion/training/train.py`: Training loop — `AdamW`, `CrossEntropyLoss`, cosine LR scheduler with warmup, gradient clipping. Freezes Wav2Vec2 and BERT encoders for the first 5 epochs, then unfreezes with a lower LR (`1e-5`) for the remaining epochs. Logs loss + weighted F1 per epoch. Saves best checkpoint to `emotion/checkpoints/`.
14. `emotion/training/evaluate.py`: Computes weighted accuracy, macro F1, per-class F1, and confusion matrix. Designed to run on a held-out test split.

### Phase 8 — API Integration (`api.py`)

15. In the startup event, instantiate `EmotionAnalyzer` alongside the existing `Transcriber`.
16. In the `/analyze` loop, after `transcriber.transcribe(clip_path)` returns `text`, call `emotion_analyzer.analyze(clip_path, text)` → returns `{"emotion": "happy", "confidence": 0.82, "sarcasm": false, "ambiguity_score": 0.14, "vader": {...}}`.
17. Merge the emotion dict into each segment's response object.

### Phase 9 — Frontend Update (`speech-insight-frontend/src/App.jsx`)

18. Add an "Emotion" column to the results `<table>`.
19. Render `emotion` as a color-coded badge (e.g., `happy` → green, `angry` → red, `sad` → blue, `neutral` → grey) with the confidence score as a percentage.
20. Render `sarcasm: true` as a secondary badge. Show `ambiguity_score` as a subtle tooltip or small grey indicator.

### Phase 10 — Dependencies (`requirements.txt`)

21. Add: `vaderSentiment`, `transformers[torch]` (already present), `scikit-learn` (for F1 metrics in training). No new heavy audio libraries needed — `librosa` is already listed.

---

## Verification

- Unit test `AcousticEncoder`, `LinguisticEncoder`, and `CrossAttentionFusion` independently with dummy tensors to confirm output shapes — `[batch, 768]`, `[batch, 768]`, `[batch, 7]`.
- Run the full `/analyze` endpoint against the existing sample in `processed/24e05130/`, confirm response includes emotion fields per segment.
- Check sarcasm detection with a manually crafted example (e.g., transcript = "oh great, another boring meeting" with a flat/low-energy audio tone).
- Training: run `train.py` for 1 epoch on a small IEMOCAP subset to confirm the loop completes without error.

---

## Decisions

- **No timestamps needed**: Cross-attention operates on the saved `.wav` clips directly; no need to propagate start/end times since each clip is already a self-contained utterance.
- **Separate Wav2Vec2 model for emotion**: The `final_model/` CTC model is not suitable (its encodings are optimized for character prediction, not affective features). `superb/wav2vec2-base-superb-er` is the right encoder.
- **Zero-shot fallback before training**: The system works out of the box via pre-trained model ensembling; training on IEMOCAP later improves the cross-attention fusion head specifically.
- **Sarcasm initially rule-based**: VADER positive + acoustic negative = sarcasm candidate. The dedicated sarcasm binary head takes over once trained.
