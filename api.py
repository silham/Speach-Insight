import warnings
warnings.filterwarnings("ignore", message="(?s).*torchcodec.*")

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
# Force stdout/stderr to use UTF-8 on Windows to prevent UnicodeEncodeError from emoji prints
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from auth import (
    Analysis,
    User,
    create_token,
    get_current_user,
    get_db,
    init_db,
    require_admin,
    verify_password,
    visible_analysis_or_404,
)
import shutil
import os
from media_utils import (
    convert_media_to_wav,
    validate_media_file,
    sanitize_filename,
    check_ffmpeg_installed
)
import uuid
import json

from rag import add_document_to_db
from dotenv import load_dotenv

from model import load_transcriber
from emotion import EmotionAnalyzer
from template_classifier import load_template_classifier
from pipeline import AnalysisPipeline
from pipeline.lead_speaker import StubLeadSpeakerIdentifier
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"

for folder in [UPLOAD_DIR, PROCESSED_DIR]:
    os.makedirs(folder, exist_ok=True)

# Verify FFmpeg is installed at startup
try:
    check_ffmpeg_installed()
    print("✅ FFmpeg check passed.")
except Exception as e:
    print(f"⚠️ FFmpeg check failed at startup: {e}")

load_dotenv()
app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database (users + analysis ownership) ---
init_db()
print("✅ Database ready.")

app.mount("/audio", StaticFiles(directory=PROCESSED_DIR), name="audio")

# --- Load AI Models (once at startup) ---
print("Initializing AI Models...")
transcriber = load_transcriber()
emotion_analyzer = EmotionAnalyzer()
template_clf = load_template_classifier()

# Swap StubLeadSpeakerIdentifier for your trained model when ready.
# The pipeline contract does not change — only this one line.
lead_speaker = StubLeadSpeakerIdentifier()

pipeline = AnalysisPipeline(
    transcriber=transcriber,
    emotion_analyzer=emotion_analyzer,
    template_classifier=template_clf,
    lead_speaker=lead_speaker,
)
print("Pipeline ready.")


@app.get("/")
def home():
    return {"status": "SpeechInSight Backend is Running"}


# ── AUTH ─────────────────────────────────────────────────────
# Login only. Accounts are created with `python seed_users.py`.

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/login")
def login(payload: LoginRequest, db=Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.strip().lower()).first()

    # Same message either way so the response can't be used to enumerate emails.
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    return {"access_token": create_token(user), "user": user.to_dict()}


@app.get("/auth/me")
def me(user: User = Depends(get_current_user)):
    return user.to_dict()


# ── ANALYSIS HISTORY ─────────────────────────────────────────

@app.get("/analyses")
def list_analyses(user: User = Depends(get_current_user), db=Depends(get_db)):
    """Analyses the caller may see: their own, or everything for an admin."""
    q = db.query(Analysis, User.name).join(User, Analysis.owner_id == User.id)
    if not user.is_admin:
        q = q.filter(Analysis.owner_id == user.id)

    rows = q.order_by(Analysis.created_at.desc()).all()
    return {
        "analyses": [a.to_dict(owner_name=owner_name) for a, owner_name in rows],
        "scope": "all" if user.is_admin else "own",
    }


@app.get("/analyses/{job_id}")
def get_analysis(job_id: str, user: User = Depends(get_current_user), db=Depends(get_db)):
    """Reload a past analysis in the same shape /analyze returns."""
    visible_analysis_or_404(db, job_id, user)

    job_result_path = os.path.join(PROCESSED_DIR, job_id, "job_result.json")
    if not os.path.exists(job_result_path):
        raise HTTPException(
            status_code=404,
            detail="Stored result for this analysis is no longer on disk.",
        )

    with open(job_result_path, "r", encoding="utf-8") as f:
        job_dict = json.load(f)

    return {
        "job_id": job_dict["job_id"],
        "lead_speaker": job_dict["lead_speaker"],
        "total_speakers": job_dict["total_speakers"],
        "total_segments": job_dict["total_segments"],
        "total_duration": job_dict["total_duration"],
        "speaker_roles": job_dict.get("speaker_roles", {}),
        "data": job_dict["segments"],
    }


@app.delete("/analyses/{job_id}")
def delete_analysis(job_id: str, user: User = Depends(get_current_user), db=Depends(get_db)):
    row = visible_analysis_or_404(db, job_id, user)
    db.delete(row)
    db.commit()

    # Best-effort cleanup of the on-disk artefacts.
    job_folder = os.path.join(PROCESSED_DIR, job_id)
    if os.path.isdir(job_folder):
        shutil.rmtree(job_folder, ignore_errors=True)

    return {"status": "deleted", "job_id": job_id}


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    # ── 1. Validate & Save uploaded file ───────────────────────────────
    try:
        validate_media_file(file.filename, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    file_id = str(uuid.uuid4())[:8]
    safe_filename = sanitize_filename(file.filename)
    filename = f"{file_id}_{safe_filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    print(f"Processing: {filename}")

    # ── 2. Convert media to WAV ──────────────────────────────────────
    try:
        audio_path = convert_media_to_wav(file_path)
    except (ValueError, RuntimeError) as e:
        # Clean up uploaded file on validation or conversion failure
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Clean up uploaded file on validation or conversion failure
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Unexpected error during media conversion: {e}")

    # ── 3. Run full analysis pipeline ─────────────────────────────────
    try:
        job = pipeline.run(
            audio_path=audio_path,
            job_id=file_id,
            processed_dir=PROCESSED_DIR,
        )
    except Exception as e:
        # Clean up files on pipeline failure
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        if 'audio_path' in locals() and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")

    if not job.segments:
        raise HTTPException(status_code=400, detail="No speech detected")

    # ── 4. Serialise and return ────────────────────────────────────────
    job_dict = job.to_dict()

    # ── 5. Record ownership so this job shows up in the user's history ─
    total_score = None
    report_path = os.path.join(PROCESSED_DIR, file_id, "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                total_score = json.load(f).get("total_score")
        except Exception:
            pass

    try:
        db.add(Analysis(
            job_id=job_dict["job_id"],
            owner_id=user.id,
            filename=safe_filename,
            total_speakers=job_dict["total_speakers"],
            total_segments=job_dict["total_segments"],
            total_duration=job_dict["total_duration"],
            lead_speaker=job_dict["lead_speaker"],
            total_score=total_score,
        ))
        db.commit()
    except Exception as exc:
        # The analysis itself succeeded — don't fail the request over history.
        db.rollback()
        print(f"⚠️ Failed to record analysis ownership: {exc}")

    # Keep the "data" key the frontend already expects
    return {
        "job_id": job_dict["job_id"],
        "lead_speaker": job_dict["lead_speaker"],
        "total_speakers": job_dict["total_speakers"],
        "total_segments": job_dict["total_segments"],
        "total_duration": job_dict["total_duration"],
        "speaker_roles": job_dict.get("speaker_roles", {}),
        "data": job_dict["segments"],
    }


# ── REPORT ENDPOINT ──────────────────────────────────────────

@app.get("/report/{job_id}")
async def get_report(
    job_id: str,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Return the generated report for a processed job enriched with speaker stats."""
    visible_analysis_or_404(db, job_id, user)

    report_path = os.path.join(PROCESSED_DIR, job_id, "report.json")
    job_result_path = os.path.join(PROCESSED_DIR, job_id, "job_result.json")
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found for this job.")

    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    speaker_stats = {}
    lead_speaker = None
    
    if os.path.exists(job_result_path):
        try:
            with open(job_result_path, "r", encoding="utf-8") as f:
                job_data = json.load(f)
            
            lead_speaker = job_data.get("lead_speaker")
            segments = job_data.get("segments", [])
            speaker_roles = job_data.get("speaker_roles", {})
            
            from collections import defaultdict
            spk_segments = defaultdict(list)
            for seg in segments:
                spk_segments[seg["speaker"]].append(seg)
                
            for spk, segs in spk_segments.items():
                talk_time = sum(float(s.get("end_time", 0.0)) - float(s.get("start_time", 0.0)) for s in segs)
                
                # Dominant emotion
                emotions = {}
                total_sentiment = 0.0
                sentiment_count = 0
                for s in segs:
                    emo = s.get("emotion", "neutral")
                    # Split 'happy 86%' to 'happy'
                    if " " in emo:
                        emo = emo.split()[0]
                    emotions[emo] = emotions.get(emo, 0) + 1
                    
                    vader = s.get("vader", {})
                    if vader and isinstance(vader.get("compound"), (int, float)):
                        total_sentiment += float(vader["compound"])
                        sentiment_count += 1
                        
                dominant_emotion = "neutral"
                if emotions:
                    dominant_emotion = max(emotions, key=emotions.get)
                    
                avg_sentiment = total_sentiment / sentiment_count if sentiment_count > 0 else 0.0
                
                role_info = speaker_roles.get(spk, {})
                
                speaker_stats[spk] = {
                    "role": role_info.get("role", "Other"),
                    "confidence": float(role_info.get("probability", 0.0)),
                    "emotion": dominant_emotion,
                    "sentiment": round(avg_sentiment, 2),
                    "speaking_time": round(talk_time, 1),
                    "turns": len(segs),
                    "lead_speaker": (spk == lead_speaker),
                    "evidence": role_info.get("evidence", []),
                    "xgboost": role_info.get("xgboost"),
                    "gemini": role_info.get("gemini"),
                    "final_role": role_info.get("final_role"),
                    "prediction_source": role_info.get("prediction_source")
                }
        except Exception as exc:
            print(f"⚠️ Error compiling speaker stats for report: {exc}")

    # Fetch resolution metadata from job result
    leader_resolution = None
    if os.path.exists(job_result_path):
        try:
            with open(job_result_path, "r", encoding="utf-8") as f:
                job_data = json.load(f)
            leader_resolution = job_data.get("metadata", {}).get("leader_resolution")
        except Exception:
            pass

    if not leader_resolution:
        leader_resolution = {
            "required": False,
            "candidate_count": len(speaker_stats) if speaker_stats else 0,
            "candidate_speakers": list(speaker_stats.keys()),
            "selected": lead_speaker,
            "method": "duration_heuristic_legacy" if lead_speaker else "none",
            "reason": "Legacy report loaded. Resolution metadata unavailable."
        }

    return {
        **report_data,
        "lead_speaker": lead_speaker,
        "speaker_stats": speaker_stats,
        "leader_resolution": leader_resolution
    }


# ── RAG UPLOAD ENDPOINT ──────────────────────────────────────

@app.post("/rag/upload")
async def rag_upload(
    file: UploadFile = File(...),
    user: User = Depends(require_admin),
):
    """Index a guideline document. Admin only — the guideline base is shared
    by every user, so a normal account must not be able to change it."""
    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_added = add_document_to_db(file_path)
        return {"status": "success", "chunks_added": chunks_added, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
