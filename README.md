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
- Template classification — Zero-shot topic classification per utterance
- **Multi-Category Scoring & Evidence** — Automatically scores five categories from each analysed recording:
  - **Template** (10 pts) — Validates presence and order of template labels
  - **WarmUp** (15 pts) — RAG similarity (10) + emotion tone (5)
  - **Praise** (20 pts) — RAG similarity (10) + emotion tone (10), flags sad/angry tones
  - **Suggest** (20 pts) — RAG similarity (20), balance penalty (-5 if PSuggest or NSuggest < 30%), angry tone penalty (-10)
- Speaker Role Classification — Identifies speaker domains (Leader, HR, Junior, Other) using a hybrid local XGBoost and Gemini fallback architecture.
- Evaluation Leader Resolution — Decoupled resolution engine that deterministic elects a single evaluation target using meeting-level structure and confidence margins.
- Prediction Provenance — Records and serializes full inference paths, model confidence scores, and fallback invocation logs.
- Developer Mode — User-friendly dashboard for analytics, with a toggleable developer view exposing pipeline provenance and resolution trees.
- Speaker Analytics Dashboard — Restructured vertical flex header layout with color-coded badges, circular initials avatars, audio playback, and dedicated RAG chat interface.
- **Knowledge Base (RAG)** — Upload documents (PDF/DOCX/TXT) and evaluate speaking guidelines using Gemini and ChromaDB. Sentences are classified using the local template classifier model to build 5 separate category-specific RAG databases (`warmup`, `praise`, `suggest`, `listen`, `direct`).

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
| LangChain + ChromaDB | Vector search / RAG knowledge base |
| Google Gemini API | RAG LLM engine (`gemini-2.5-flash`) |

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

### 1. Download Required Models

The models for transcription (`final_model`), template classification (`Template_classifier_model`), and role classification (`pipeline/role_classification/models/`) are too large for Git. 
Please download them from this **[Google Drive Link](https://drive.google.com/drive/folders/17riUj4NUOTFX4QPVl9_gjI2_40A_fh43?usp=sharing)**.

Extract and place:
- The `final_model` and `Template_classifier_model` folders directly in the project root directory.
- The role classification model files (`scaler.joblib`, `label_encoder.joblib`, `svd.joblib`, `tfidf.joblib`, `xgboost_model.joblib`) inside the `pipeline/role_classification/models/` directory.

### 2. Clone the repo

```bash
git clone <repo-url>
cd Transcribe_Model
```

### 3. Create a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Set your Hugging Face token

The diarization model requires authentication. Create a read token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and export it:

```bash
export HF_TOKEN=hf_...
```

Add this to your shell profile (`.zshrc`, `.bashrc`) so it persists across sessions.

### 5. Set your Google API Key (for RAG)

Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_gemini_api_key
```
You can get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 6. Install frontend dependencies

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

## Speaker Role Classification

The speaker role classification engine uses a confidence-aware routing strategy to optimize processing costs:
- **XGBoost Classifiers:** Speakers' aggregated turns are vectorized using TF-IDF and SVD, then classified locally.
- **Confidence Filter:** If the maximum class probability is >= 0.80, the local prediction is accepted.
- **Gemini Fallback:** If confidence is below 0.80, the transcript and meeting context are routed to Gemini to resolve ambiguities.
- **Label Normalization:** Outputs from all prediction paths are normalized to canonical roles (`Leader`, `HR`, `Junior`, `Other`).
- **Prediction Provenance:** All classification paths and raw probabilities are serialized for transparency.

---

## Evaluation Leader Resolution

In SpeechInSight, **Speaker Role Classification** and **Evaluation Leader Resolution** are separated:
* **Role Classification:** Answers *"What organizational domain does this participant represent?"*
* **Evaluation Leader:** Answers *"Which speaker is the primary target for behavioral scoring?"*

Instead of relying on simple heuristics like longest speaking duration (which often misclassifies active junior speakers as leaders), the system uses a post-inference resolver (`RoleBasedLeadSpeakerIdentifier`). It runs a deterministic decision tree on the normalized roles, checking confidence margins and meeting opener/closer sequences, falling back to speaking duration only as a final tie-breaker.

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
  "speaker_roles": {
    "SPEAKER_00": {
      "speaker": "SPEAKER_00",
      "predicted_role": "Leader",
      "confidence": 0.94,
      "xgboost": {
        "role": "Leader",
        "confidence": 0.94,
        "probabilities": { "Leader": 0.94, "HR": 0.02, "Junior": 0.02, "Other": 0.02 }
      },
      "gemini": { "used": false, "role": null },
      "final_role": "Leader",
      "prediction_source": "xgboost"
    }
  },
  "metadata": {
    "leader_resolution": {
      "required": false,
      "method": "single_leader_resolved",
      "candidates": ["SPEAKER_00"],
      "logs": ["Electing single predicted leader: SPEAKER_00"]
    }
  },
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
      "role": "Leader"
    }
  ]
}
```

Segment audio clips are served statically at `/audio/{job_id}/{filename}`.

### `POST /rag/upload`

Upload a document (PDF/DOCX/TXT) to index into the ChromaDB vector stores. Individual sentences in the document are automatically classified using the local `Template_classifier_model` and routed to 5 separate RAG databases (`warmup`, `praise`, `suggest`, `listen`, `direct`).

**Request:** `multipart/form-data`, field `file`.

**Response:**
```jsonc
{
  "status": "success",
  "chunks_added": 15,
  "filename": "document.docx"
}
```

### `POST /rag/ask`
Query the knowledge base using the Gemini LLM.

**Request:** `application/json`
```jsonc
{
  "query": "What is the document about?"
}
```

**Response:**
```jsonc
{
  "answer": "The document is about...",
  "context": [
    "chunk text 1",
    "chunk text 2"
  ]
}
```

---

## Project Structure

```
Transcribe_Model/
│
├── api.py                        FastAPI server — pipeline entry point
├── app.py                        Streamlit developer dashboard
├── rag.py                        Retrieval-Augmented Generation (RAG) module
├── media_utils.py                Video → WAV conversion (FFmpeg)
├── model.py                      Wav2Vec2 CTC transcription model
├── segmentation.py               Pyannote speaker diarization
├── requirements.txt
│
├── pipeline/                     Pipeline orchestration layer
│   ├── __init__.py               AnalysisPipeline — chains all stages
│   ├── schemas.py                SegmentMeta, SegmentResult, JobResult
│   ├── scoring.py                Template category scoring and evidence generation
│   ├── role_classification/      Hybrid role prediction stage
│   └── lead_speaker/
│       └── __init__.py           LeadSpeakerIdentifier ABC + post-inference resolver
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
├── template_classifier.py        Template classification module
├── Template_classifier_model/    Zero-shot classifier weights
├── final_model/                  Wav2Vec2 CTC weights (transcription)
├── processed/                    Per-job segmented audio clips
├── uploads/                      Raw uploaded files
│
└── speech-insight-frontend/      React + Vite frontend
    └── src/
        ├── components/           Dashboard cards and summary tables
        ├── hooks/                UseSpeakerAnalysis and filter hooks
        ├── pages/                Dashboard layouts
        └── utils/                Formating and theme helpers
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

## Developer Reference Documentation

Detailed architectural specifications, interface definitions, execution flow diagrams, and contributor extension guides are available in the module folders:
* **Role Prediction & Falling Back:** [pipeline/role_classification/README.md](file:///Users/denuwanwijesinghe/Documents/BSc. Artficial Intellligence/se_project/project/Speach-Insight/pipeline/role_classification/README.md)
* **Lead Speaker Target Resolver:** [pipeline/lead_speaker/README.md](file:///Users/denuwanwijesinghe/Documents/BSc. Artficial Intellligence/se_project/project/Speach-Insight/pipeline/lead_speaker/README.md)
