# SpeechInSight

AI-powered speech analysis pipeline that transcribes audio/video, identifies speakers, recognises emotion per utterance, and determines the lead speaker.

---

## Features

- Upload audio (WAV, MP3, FLAC) or video (MP4, MOV, MKV, …)
- Automatic video → audio conversion via FFmpeg
- Speaker diarization — detects who is speaking and when
- Per-segment speech transcription (Wav2Vec2 CTC)
- Multimodal emotion recognition per utterance:
  - **Acoustic** — Wav2Vec2 (`superb/wav2vec2-base-superb-er`)
  - **Linguistic** — BERT (`bert-base-uncased`)
  - **Sentiment** — VADER lexicon
  - **Fusion** — Cross-attention + classifier heads (7 Ekman emotions, sarcasm, ambiguity)
- Lead speaker identification (stub ships; see `pipeline/lead_speaker/`)
- React dashboard with colour-coded emotion badges and audio playback

---

## Tech Stack

### Backend
| Library | Role |
|---|---|
| Python 3.10+ | Runtime |
| FastAPI + Uvicorn | REST API server |
| PyTorch + torchaudio | Tensor ops, audio I/O |
| Hugging Face Transformers | Wav2Vec2, BERT model loading |
| pyannote.audio 3.1 | Speaker diarization |
| librosa | Pitch / energy feature extraction |
| vaderSentiment | Lexicon-based sentiment |
| scikit-learn | Metrics (training phase) |
| FFmpeg | Video → audio conversion |

### Frontend
| Library | Role |
|---|---|
| React 19 + Vite 7 | UI framework & build tool |
| Axios | HTTP client |

---

## Prerequisites

- Python ≥ 3.10
- Node.js ≥ 18
- FFmpeg installed and on `$PATH`
- A Hugging Face account with access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (requires accepting the model licence)

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd Transcribe_Model
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set your Hugging Face token

The diarization model requires authentication. Create a read token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and export it:

```bash
export HF_TOKEN=hf_...
```

Add this to your shell profile (`.zshrc`, `.bashrc`) so it persists across sessions.

### 4. Install frontend dependencies

```bash
cd speech-insight-frontend
npm install
cd ..
```

---

## Running the Project

### Start the API server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The server starts at `http://localhost:8000`.  
On first run the models download automatically into `emotion/models/`.

### Start the React frontend

In a separate terminal:

```bash
cd speech-insight-frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

### (Optional) Streamlit dev dashboard

```bash
streamlit run app.py
```

A local dashboard at `http://localhost:8501` that runs the diarization +
transcription pipeline without the full React frontend.

---

## API

### `GET /`
Health check. Returns `{"status": "SpeechInSight Backend is Running"}`.

### `POST /analyze`
Upload an audio or video file for full analysis.

**Request:** `multipart/form-data`, field `file`.

**Response:**
```jsonc
{
  "job_id": "abc12345",
  "lead_speaker": "SPEAKER_00",
  "total_speakers": 2,
  "total_segments": 14,
  "total_duration": 182.4,
  "data": [
    {
      "segment_id": 0,
      "speaker": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 4.2,
      "audio_url": "/audio/abc12345/seg_000_SPEAKER_00.wav",
      "text": "good morning everyone",
      "emotion": "happy",
      "confidence": 0.73,
      "all_emotions": { "happy": 0.73, "neutral": 0.10, ... },
      "sarcasm": false,
      "sarcasm_score": 0.02,
      "ambiguity_score": 0.41,
      "vader": { "pos": 0.45, "neg": 0.0, "neu": 0.55, "compound": 0.61 },
      "paralinguistic": { "pitch": 182.3, "energy": 0.000412, "speaking_rate": 3.1 }
    }
  ]
}
```

Segment audio clips are served statically at `/audio/{job_id}/{filename}`.

---

## Project Structure

```
Transcribe_Model/
│
├── api.py                        FastAPI server — pipeline entry point
├── app.py                        Streamlit developer dashboard
├── media_utils.py                Video → WAV conversion (FFmpeg)
├── model.py                      Wav2Vec2 CTC transcription model
├── segmentation.py               Pyannote speaker diarization
├── requirements.txt
│
├── pipeline/                     Pipeline orchestration layer
│   ├── __init__.py               AnalysisPipeline — chains all stages
│   ├── schemas.py                SegmentMeta, SegmentResult, JobResult
│   └── lead_speaker/
│       └── __init__.py           LeadSpeakerIdentifier ABC + working stub
│
├── emotion/                      Multimodal emotion recognition package
│   ├── __init__.py               EmotionAnalyzer façade
│   ├── acoustic_encoder.py       Wav2Vec2 superb-er audio features
│   ├── linguistic_encoder.py     BERT text features
│   ├── vader_analyzer.py         Lexicon sentiment tensor
│   ├── cross_attention.py        CrossAttentionFusion module
│   ├── emotion_classifier.py     Classifier heads (emotion, sarcasm)
│   └── models/                   HuggingFace model cache
│
├── final_model/                  Wav2Vec2 CTC weights (transcription)
├── processed/                    Per-job segmented audio clips
├── uploads/                      Raw uploaded files
│
└── speech-insight-frontend/      React + Vite frontend
    └── src/
        └── App.jsx               Results table with emotion badges
```

See [Dataflow.md](Dataflow.md) for a detailed walkthrough of how data moves
between stages and how to add new models to the pipeline.

---

## Extending the Pipeline

The pipeline is designed to be extended without touching existing code.
To add a new model (e.g. topic detection, interruption analysis):

1. Create a module under `pipeline/your_model/`.
2. Implement a class with an `analyse(job: JobResult) -> None` method.
3. Register it in `pipeline/__init__.py`.
4. Instantiate it in `api.py`.

All new fields written to `SegmentResult.extras` are automatically exposed
in the API response. Full instructions are in [Dataflow.md](Dataflow.md).

---

## What's Next — Lead Speaker Identification

The next module to build is a trained lead-speaker classifier in
`pipeline/lead_speaker/`. The stub (most talking time) is already wired
into the pipeline and will be replaced by your implementation.

A fully-enriched `JobResult` is available at that stage, including:
talk times, turn counts, transcripts, emotion distributions, paralinguistic
features (pitch, energy, speaking rate), and segment timestamps.

See the docstring in `pipeline/lead_speaker/__init__.py` for the full
feature list and integration steps.
