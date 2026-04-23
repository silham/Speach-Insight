# SpeechInSight — Data Flow Reference

> **Audience:** developers joining the project at any stage.  
> **Last updated:** April 2026

---

## Table of Contents

1. [High-Level Pipeline Overview](#1-high-level-pipeline-overview)
2. [Shared Data Structures](#2-shared-data-structures)
3. [Stage-by-Stage Walkthrough](#3-stage-by-stage-walkthrough)
   - [Stage 1 — Media Ingestion](#stage-1--media-ingestion)
   - [Stage 2 — Speaker Diarization](#stage-2--speaker-diarization)
   - [Stage 3 — Transcription](#stage-3--transcription)
   - [Stage 4 — Emotion Recognition](#stage-4--emotion-recognition)
   - [Stage 5 — Lead Speaker Identification ← **Next to build**](#stage-5--lead-speaker-identification)
4. [API Response Schema](#4-api-response-schema)
5. [Adding a New Model](#5-adding-a-new-model)
6. [Training the Emotion Model](#6-training-the-emotion-model)
7. [Project File Map](#7-project-file-map)

---

## 1. High-Level Pipeline Overview

```
                         ┌───────────────────────────────────────────────────────┐
                         │                  AnalysisPipeline                     │
                         │                (pipeline/__init__.py)                  │
                         └───────────────────────────────────────────────────────┘

  HTTP multipart upload
  audio / video file
         │
         ▼
  ┌─────────────────┐   WAV file (16 kHz mono)
  │ Media Ingestion │ ─────────────────────────────────────────────────────────────┐
  │ media_utils.py  │                                                              │
  └─────────────────┘                                                              │
                                                                                   │
         ┌─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────┐  list[dict]  ←  {segment_id, path, speaker, start, end}
  │  Diarization        │ ──────────────────────────────────────────────────────────┐
  │  segmentation.py    │                                                           │
  │  Pyannote 3.1       │                                                           │
  └─────────────────────┘                                                           │
                                                                                    │
         ┌──────────────────────────────────────────────────────────────────────────┘
         │  (one SegmentResult per turn, populated from SegmentMeta)
         │
         ▼
  ┌─────────────────────┐  fills  SegmentResult.text
  │  Transcription      │ ──────────────────────────────────────────────────────────┐
  │  model.py           │                                                           │
  │  Wav2Vec2 CTC       │                                                           │
  └─────────────────────┘                                                           │
                                                                                    │
         ┌──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Emotion Recognition           (emotion/)                │
  │                                                          │
  │  AcousticEncoder  (Wav2Vec2 superb-er)  ──┐              │
  │  LinguisticEncoder  (BERT-base-uncased) ──┤→ CrossAttn   │  fills SegmentResult
  │  VaderAnalyzer  (lexicon sentiment)     ──┘   Fusion     │  .emotion / .confidence
  │                                          └──► Classifier │  .sarcasm / .vader / …
  └──────────────────────────────────────────────────────────┘
         │
         │  (all segments resolved → JobResult ready)
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Lead Speaker Identification   (pipeline/lead_speaker/) │  fills JobResult
  │  ← NEXT TO BUILD                                        │  .lead_speaker
  │                                                          │
  │  StubLeadSpeakerIdentifier  (ships with repo)           │
  │  → picks speaker with most total talking time           │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────┐
  │  << future models >> │  write into SegmentResult.extras
  │  topic modelling     │  or JobResult.metadata
  │  sentiment trend     │
  └──────────────────────┘
         │
         ▼
     JobResult  →  JSON  →  /analyze  response
```

---

## 2. Shared Data Structures

All models communicate through two dataclasses defined in
`pipeline/schemas.py`.  **Never pass raw dicts between stages** —
always use these types so serialisation, typing, and future extensions
stay consistent.

### `SegmentMeta` — diarization output (read-only)

| Field | Type | Description |
|---|---|---|
| `segment_id` | `int` | 0-based index in the original audio |
| `speaker` | `str` | Pyannote label, e.g. `"SPEAKER_00"` |
| `start_time` | `float` | Turn start in seconds |
| `end_time` | `float` | Turn end in seconds |
| `audio_path` | `str` | Absolute path to the `.wav` clip |
| `audio_url` | `str` | Public URL served by FastAPI `/audio` |

`SegmentMeta` is internal to the pipeline.  Downstream stages receive
`SegmentResult` (which includes the same fields).

---

### `SegmentResult` — one enriched speaker turn

Built incrementally — fields are filled by each stage in order.

| Field | Set by stage | Type | Description |
|---|---|---|---|
| `segment_id` | Diarization | `int` | Sequential index |
| `speaker` | Diarization | `str` | `"SPEAKER_00"` etc. |
| `start_time` | Diarization | `float` | Seconds |
| `end_time` | Diarization | `float` | Seconds |
| `audio_path` | Diarization | `str` | Local path |
| `audio_url` | Diarization | `str` | `/audio/{job_id}/seg_000_SPEAKER_00.wav` |
| `text` | Transcription | `str` | Transcript text (lowercase) |
| `emotion` | Emotion | `str` | Dominant Ekman label |
| `confidence` | Emotion | `float` | Probability of dominant label |
| `all_emotions` | Emotion | `dict[str, float]` | `{label: prob}` for all 7 classes |
| `sarcasm` | Emotion | `bool` | Sarcasm detected |
| `sarcasm_score` | Emotion | `float` | Raw sarcasm probability |
| `ambiguity_score` | Emotion | `float` | Shannon entropy, normalised 0–1 |
| `vader` | Emotion | `dict[str, float]` | `{pos, neg, neu, compound}` |
| `paralinguistic` | Emotion | `dict[str, float]` | `{pitch, energy, speaking_rate}` |
| `extras` | Future models | `dict[str, Any]` | Extensibility hook — see §5 |

---

### `JobResult` — one uploaded file

| Field | Set by stage | Type | Description |
|---|---|---|---|
| `job_id` | Pipeline init | `str` | UUID-based job identifier |
| `segments` | Diarization→ | `list[SegmentResult]` | All speaker turns |
| `lead_speaker` | Lead Speaker | `str \| None` | Winning speaker label |
| `total_speakers` | `finalise()` | `int` | Unique speaker count |
| `total_segments` | `finalise()` | `int` | Total turn count |
| `total_duration` | `finalise()` | `float` | Sum of all segment durations (s) |
| `metadata` | Future models | `dict` | Extensibility hook — see §5 |

Helper:  
`job.speaker_talk_times() → dict[str, float]` — returns `{speaker: seconds}`,
available to every stage after diarization.

---

## 3. Stage-by-Stage Walkthrough

### Stage 1 — Media Ingestion

**File:** `media_utils.py`

```
upload (audio or video)
        │
        ▼
convert_video_to_audio(path) → WAV (16 kHz, mono)
        │
        └── if already audio → returned unchanged
```

- Uses `ffmpeg` for video → audio conversion.
- Output is always 16 kHz mono WAV — the format required by all downstream
  audio models (Pyannote, Wav2Vec2).

---

### Stage 2 — Speaker Diarization

**File:** `segmentation.py`  
**Model:** `pyannote/speaker-diarization-3.1`

```
WAV file
   │
   ▼
segment_and_save(input_file, output_folder)
   │
   └── returns list[dict]:
       [
         {"segment_id": 0, "path": ".../seg_000_SPEAKER_00.wav",
          "speaker": "SPEAKER_00", "start": 0.0, "end": 4.2},
         {"segment_id": 1, "path": ".../seg_001_SPEAKER_01.wav",
          "speaker": "SPEAKER_01", "start": 4.5, "end": 8.1},
         ...
       ]
```

Each entry maps to a `SegmentResult` via `SegmentResult.from_meta(meta)`.

**Important:** Speaker labels (`SPEAKER_00`, `SPEAKER_01`) are
per-job ordinal labels.  Two different jobs may assign different labels
to the same physical person.  The lead-speaker stage works on these ordinal
labels per job.

---

### Stage 3 — Transcription

**File:** `model.py`  
**Model:** `Wav2Vec2ForCTC` loaded from `final_model/` (fine-tuned CTC model)

```
seg.audio_path  →  Transcriber.transcribe(path)  →  seg.text (str)
```

- Resamples to 16 kHz, converts stereo → mono.
- Returns lowercase text.
- The transcription model is **separate** from the acoustic emotion encoder
  (`superb/wav2vec2-base-superb-er`).  Both are Wav2Vec2 variants but serve
  different purposes (CTC decoding vs. affective feature extraction).

---

### Stage 4 — Emotion Recognition

**Package:** `emotion/`

```
seg.audio_path ─┐
seg.text        ─┤
                 ▼
          EmotionAnalyzer.analyze(audio_path, text) → dict
                 │
                 ▼
          [AcousticEncoder]     WAV → pooled [768] + frames [T_a, 768]
                                    + paralinguistic [3]
          [LinguisticEncoder]   text → CLS [768] + tokens [T_t, 768]
          [VaderAnalyzer]       text → sentiment [4]
                 │
                 ▼
          [CrossAttentionFusion]
            audio-Q → text-KV  (audio attends text)
            text-Q  → audio-KV (text attends audio)
            concat + VADER → Linear → [512]
                 │
                 ▼
          [MultimodalEmotionClassifier]
            EmotionHead   → 7-class softmax  → emotion, confidence
            SarcasmHead   → sigmoid          → sarcasm, sarcasm_score
            Shannon entropy                  → ambiguity_score
```

**Mode:**
- **Zero-shot (default, no training needed):**  
  Uses `j-hartmann/emotion-english-distilroberta-base` + acoustic energy
  heuristic + VADER weighting.
- **Trained (after Phase 7):**  
  Pass `checkpoint_path` to `EmotionAnalyzer` to load the IEMOCAP-trained
  cross-attention fusion weights.

Fields written to `SegmentResult`:  
`emotion`, `confidence`, `all_emotions`, `sarcasm`, `sarcasm_score`,
`ambiguity_score`, `vader`, `paralinguistic`

---

### Stage 5 — Lead Speaker Identification

**Package:** `pipeline/lead_speaker/`  
**Status:** Stub ships with the repo — **this is the next model to build.**

```
JobResult (all segments resolved)
        │
        ▼
LeadSpeakerIdentifier.identify(job) → "SPEAKER_00"  (or None)
        │
        └── writes JobResult.lead_speaker
```

The stub (`StubLeadSpeakerIdentifier`) picks the speaker with the most total
talking time.  See `pipeline/lead_speaker/__init__.py` for the full list of
suggested features for a trained classifier.

---

## 4. API Response Schema

`POST /analyze` returns:

```jsonc
{
  "job_id": "abc12345",
  "lead_speaker": "SPEAKER_00",       // null until trained model is swapped in
  "total_speakers": 2,
  "total_segments": 14,
  "total_duration": 182.4,
  "data": [
    {
      "segment_id": 0,
      "speaker": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 4.2,
      "audio_path": "processed/abc12345/seg_000_SPEAKER_00.wav",
      "audio_url": "/audio/abc12345/seg_000_SPEAKER_00.wav",
      "text": "good morning everyone",
      "emotion": "happy",
      "confidence": 0.73,
      "all_emotions": {"angry": 0.02, "disgust": 0.01, "fear": 0.01,
                       "happy": 0.73, "sad": 0.05, "surprise": 0.08,
                       "neutral": 0.10},
      "sarcasm": false,
      "sarcasm_score": 0.02,
      "ambiguity_score": 0.41,
      "vader": {"pos": 0.45, "neg": 0.0, "neu": 0.55, "compound": 0.61},
      "paralinguistic": {"pitch": 182.3, "energy": 0.000412, "speaking_rate": 3.1}
      // future models write extra keys here (from SegmentResult.extras)
    }
    // ... more segments
  ]
}
```

New models that write into `SegmentResult.extras` will have their keys
**automatically flattened** into each segment object by `SegmentResult.to_dict()`.  
No changes to the serialisation code are required.

---

## 5. Adding a New Model

Follow these four steps to add any model (topic detection, interruption
analysis, tone trend, etc.).

### Step 1 — Decide where the result lives

| Result granularity | Where to write |
|---|---|
| Per segment (e.g. topic tag) | `seg.extras["topic"] = "budget"` |
| Per job (e.g. meeting sentiment arc) | `job.metadata["sentiment_arc"] = [...]` |

### Step 2 — Create your module

```
pipeline/
    your_model/
        __init__.py    ← implement YourModel class
        model.py       ← ML code if needed
        README.md      ← describe inputs / outputs
```

Your class needs one method:

```python
# Per-segment model
def analyse(self, job: JobResult) -> None:
    for seg in job.segments:
        seg.extras["your_key"] = your_inference(seg)

# OR — per-job model
def analyse(self, job: JobResult) -> None:
    job.metadata["your_key"] = your_inference(job)
```

### Step 3 — Register it in the pipeline

Open `pipeline/__init__.py` and add your stage after the existing ones:

```python
# ── Stage 5: Your New Model ─────────────────────────────────────────
if self.your_model is not None:
    print("🔍 Running your model…")
    try:
        self.your_model.analyse(job)
    except Exception as exc:
        print(f"⚠️  Your model failed: {exc}")
```

Add the parameter to `AnalysisPipeline.__init__`:

```python
def __init__(self, ..., your_model=None):
    ...
    self.your_model = your_model
```

### Step 4 — Instantiate in `api.py`

```python
from pipeline.your_model import YourModel

your_model = YourModel()

pipeline = AnalysisPipeline(
    transcriber=transcriber,
    emotion_analyzer=emotion_analyzer,
    lead_speaker=lead_speaker,
    your_model=your_model,          # ← add here
)
```

That's it.  The new keys appear automatically in the `/analyze` JSON response.

---

## 6. Training the Emotion Model

The emotion package ships in **zero-shot mode** (no training required).
A full training pipeline for the cross-attention fusion head is planned for
Phase 7 but not yet implemented.

### What exists

| Component | Status |
|---|---|
| `AcousticEncoder` (Wav2Vec2 superb-er) | ✅ ready |
| `LinguisticEncoder` (BERT-base-uncased) | ✅ ready |
| `VaderAnalyzer` | ✅ ready |
| `CrossAttentionFusion` | ✅ ready |
| `MultimodalEmotionClassifier` | ✅ ready (heads defined) |
| `EmotionAnalyzer` zero-shot ensemble | ✅ working |
| `emotion/training/dataset.py` (IEMOCAP) | ❌ **to build** |
| `emotion/training/train.py` | ❌ **to build** |
| `emotion/training/evaluate.py` | ❌ **to build** |

### Training contract (Phase 7 spec)

1. **Dataset:** IEMOCAP — `IEMOCAPDataset` yields `(wav_path, text, label)`.
   Map 8 IEMOCAP labels → Ekman 7 (merge `excitement` → `happy`,
   drop `frustrated` / `other`).

2. **Training:**
   - Freeze Wav2Vec2 + BERT for first 5 epochs.  
   - Unfreeze both encoders at `lr = 1e-5` for remaining epochs.  
   - `AdamW`, `CrossEntropyLoss`, cosine LR with warmup.  
   - Save best checkpoint to `emotion/checkpoints/`.

3. **Switching to trained mode:**

   ```python
   # In api.py
   emotion_analyzer = EmotionAnalyzer(checkpoint_path="emotion/checkpoints/best.pt")
   ```

   No other code changes are needed — `MultimodalEmotionClassifier` detects
   the checkpoint and switches from zero-shot to the trained forward pass.

---

## 7. Project File Map

```
Transcribe_Model/
│
├── api.py                        FastAPI entry point — mounts pipeline
├── app.py                        Streamlit dev dashboard
├── media_utils.py                Video → WAV conversion (ffmpeg)
├── model.py                      Wav2Vec2 CTC transcription model
├── segmentation.py               Pyannote speaker diarization
├── requirements.txt
│
├── pipeline/                     ← PIPELINE ORCHESTRATION LAYER
│   ├── __init__.py               AnalysisPipeline class
│   ├── schemas.py                SegmentMeta, SegmentResult, JobResult
│   └── lead_speaker/
│       └── __init__.py           LeadSpeakerIdentifier ABC + StubLeadSpeakerIdentifier
│
├── emotion/                      ← EMOTION RECOGNITION PACKAGE
│   ├── __init__.py               EmotionAnalyzer façade
│   ├── acoustic_encoder.py       Wav2Vec2 superb-er features
│   ├── linguistic_encoder.py     BERT-base-uncased features
│   ├── vader_analyzer.py         Lexicon sentiment [4]-tensor
│   ├── cross_attention.py        CrossAttentionFusion module
│   ├── emotion_classifier.py     MultimodalEmotionClassifier + heads
│   └── models/                   HuggingFace model cache (gitignored)
│
├── final_model/                  Wav2Vec2 CTC weights (transcription)
├── processed/                    Per-job audio clip outputs
├── uploads/                      Raw uploaded files
│
└── speech-insight-frontend/      React + Vite frontend
    └── src/
        └── App.jsx               Renders results table with emotion badges
```
