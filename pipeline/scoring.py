import json
import os
import sys

# Ensure we can import from rag.py if executed context requires it
try:
    from rag import evaluate_warmup_with_rag
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag import evaluate_warmup_with_rag


def calculate_emotion_score(emotion_str):
    """
    Calculates score out of 5 based on emotion percentage rules.
    e.g., 'happy 98%' -> parses to 'happy' and 98.
    """
    try:
        parts = emotion_str.split()
        label = parts[0].lower()
        conf = float(parts[1].replace('%', ''))
        
        if label == "happy":
            return ((80 + 0.20 * conf) / 100.0) * 5
        elif label == "neutral":
            return ((50 + 0.30 * conf) / 100.0) * 5
        elif label == "sad":
            return ((30 + 0.20 * conf) / 100.0) * 5
        else:
            return 0.0
    except Exception:
        return 0.0


def generate_score_and_evidence(job_output_folder: str):
    """
    Reads transcript.json from the job_output_folder, calculates template scores
    and warmup metrics (RAG + Emotion), and writes score.json and evidence.json.
    """
    transcript_path = os.path.join(job_output_folder, "transcript.json")
    if not os.path.exists(transcript_path):
        print(f"[WARNING] Scoring skipped: {transcript_path} not found.")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    segments = transcript_data.get("segments", [])

    # 1. TEMPLATE SCORING
    category_indices = {
        "WarmUp": [], "Praise": [], "PSuggest": [], 
        "NSuggest": [], "Listen": [], "Direct": []
    }

    warmup_transcripts = []
    warmup_emotion_scores = []

    for i, seg in enumerate(segments):
        label = seg.get("template_label")
        if label in category_indices:
            category_indices[label].append(i)
        
        if label == "WarmUp":
            warmup_transcripts.append(seg.get("transcript", ""))
            warmup_emotion_scores.append(calculate_emotion_score(seg.get("emotion", "neutral 0%")))

    completed = []
    missed = []

    first_warmup_idx = float('inf')
    if category_indices["WarmUp"]:
        completed.append("WarmUp")
        first_warmup_idx = min(category_indices["WarmUp"])
    else:
        missed.append("WarmUp")

    if category_indices["Praise"] and any(idx > first_warmup_idx for idx in category_indices["Praise"]):
        completed.append("Praise")
    else:
        missed.append("Praise")

    for cat in ["PSuggest", "NSuggest", "Listen", "Direct"]:
        if category_indices[cat]:
            completed.append(cat)
        else:
            missed.append(cat)

    template_score_val = round((len(completed) / 6.0) * 10.0, 2)

    # 2. WARMUP RAG SCORING
    warmup_rag_score = 0.0
    rag_suggestions = "No warmup segments detected."
    if warmup_transcripts:
        combined_warmup_text = " ".join(warmup_transcripts)
        rag_result = evaluate_warmup_with_rag(combined_warmup_text)
        warmup_rag_score = round(rag_result.get("similarity_score_out_of_10", 0.0), 2)
        rag_suggestions = rag_result.get("suggestions", "No suggestions.")

    # 3. WARMUP EMOTION SCORING
    warmup_emotion_score = 0.0
    if warmup_emotion_scores:
        warmup_emotion_score = round(sum(warmup_emotion_scores) / len(warmup_emotion_scores), 2)

    total_warmup_score = round(warmup_rag_score + warmup_emotion_score, 2)

    # 4. PREPARE EVIDENCE & SCORE FILES
    score_data = {
        "template_score": template_score_val,
        "warmup_rag_score": warmup_rag_score,
        "warmup_emotion_score": warmup_emotion_score,
        "warmup_total_score": total_warmup_score
    }

    evidence_data = []
    
    # Template Evidence
    evidence_data.append({
        "category": "template",
        "score": template_score_val,
        "evidence": f"completed categories - [{', '.join(completed)}], missed categories - [{', '.join(missed)}]"
    })

    # WarmUp Evidence
    rag_text = ""
    if warmup_rag_score > 7:
        rag_text = f"Warmup recommendations followed ({int(warmup_rag_score * 10)}%). Suggestions: {rag_suggestions}"
    else:
        rag_text = f"Warm up recommendation follow percentage {int(warmup_rag_score * 10)}%, needs to improve. Suggestions: {rag_suggestions}"

    tone_text = ""
    if warmup_emotion_score >= 4:
        tone_text = "Good speaking tone maintained."
    else:
        tone_text = "Tone must improve."

    evidence_data.append({
        "category": "warmup",
        "score": total_warmup_score,
        "evidence": f"{rag_text} {tone_text}".strip()
    })

    score_path = os.path.join(job_output_folder, "score.json")
    evidence_path = os.path.join(job_output_folder, "evidence.json")

    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=4)
    
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_data, f, indent=4)

    print(f"[INFO] Score and Evidence saved to {job_output_folder}")
