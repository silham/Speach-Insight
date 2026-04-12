from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
from media_utils import convert_video_to_audio
import uuid

from model import load_transcriber
from emotion import EmotionAnalyzer
from pipeline import AnalysisPipeline
from pipeline.lead_speaker import StubLeadSpeakerIdentifier

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"

for folder in [UPLOAD_DIR, PROCESSED_DIR]:
    os.makedirs(folder, exist_ok=True)

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=PROCESSED_DIR), name="audio")

# --- Load AI Models (once at startup) ---
print("⏳ Initializing AI Models...")
transcriber = load_transcriber()
emotion_analyzer = EmotionAnalyzer()

# Swap StubLeadSpeakerIdentifier for your trained model when ready.
# The pipeline contract does not change — only this one line.
lead_speaker = StubLeadSpeakerIdentifier()

pipeline = AnalysisPipeline(
    transcriber=transcriber,
    emotion_analyzer=emotion_analyzer,
    lead_speaker=lead_speaker,
)
print("✅ Pipeline ready.")


@app.get("/")
def home():
    return {"status": "SpeechInSight Backend is Running"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    # ── 1. Save uploaded file ──────────────────────────────────────────
    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📂 Processing: {filename}")

    # ── 2. Convert video → audio if needed ────────────────────────────
    audio_path = convert_video_to_audio(file_path)

    # ── 3. Run full analysis pipeline ─────────────────────────────────
    job = pipeline.run(
        audio_path=audio_path,
        job_id=file_id,
        processed_dir=PROCESSED_DIR,
    )

    if not job.segments:
        raise HTTPException(status_code=400, detail="No speech detected")

    # ── 4. Serialise and return ────────────────────────────────────────
    job_dict = job.to_dict()

    # Keep the "data" key the frontend already expects
    return {
        "job_id": job_dict["job_id"],
        "lead_speaker": job_dict["lead_speaker"],
        "total_speakers": job_dict["total_speakers"],
        "total_segments": job_dict["total_segments"],
        "total_duration": job_dict["total_duration"],
        "data": job_dict["segments"],
    }
