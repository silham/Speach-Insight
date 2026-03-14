from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
from media_utils import convert_video_to_audio
import uuid

# --- Your helper functions ---
from segmentation import segment_and_save
from model import load_transcriber

UPLOAD_DIR = "uploads"
AUDIO_DIR = "audio"
SEGMENT_DIR = "segments"
OUTPUT_DIR = "outputs"

for folder in [UPLOAD_DIR, AUDIO_DIR, SEGMENT_DIR, OUTPUT_DIR]:
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

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=PROCESSED_DIR), name="audio")

# --- Load Transcriber ---
print("⏳ Initializing AI Models...")
transcriber = load_transcriber()

@app.get("/")
def home():
    return {"status": "SpeechInSight Backend is Running"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    # Save file
    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📂 Processing: {filename}")

    # Segment
    job_output_folder = os.path.join(PROCESSED_DIR, file_id)
    file_path = convert_video_to_audio(file_path)
    clips = segment_and_save(file_path, job_output_folder)

    if not clips:
        raise HTTPException(status_code=400, detail="No speech detected")

    # Transcribe
    results = []
    for clip_path in clips:
        text = transcriber.transcribe(clip_path)
        relative_path = f"/audio/{file_id}/{os.path.basename(clip_path)}"
        filename_parts = os.path.basename(clip_path).split('_')
        speaker = filename_parts[-1].replace(".wav", "") if "SPEAKER" in filename_parts[-1] else "Unknown"

        results.append({
            "speaker": speaker,
            "text": text,
            "audio_url": relative_path,
            "file_path": clip_path
        })

    return {"job_id": file_id, "data": results}
