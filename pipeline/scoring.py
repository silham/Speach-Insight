import json
import os
import sys

# Ensure we can import from rag.py if executed context requires it
try:
    from rag import evaluate_categories_with_rag
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag import evaluate_categories_with_rag


def calculate_warmup_emotion_score(emotion_str):
    """
    Calculates score out of 5 based on WarmUp emotion rules.
    e.g., 'happy 98%' -> parses to 'happy' and 98.
    Happy  -> 80% + balance 20% from confidence
    Neutral-> 50% + balance 30% from confidence
    Sad    -> 30% + balance 20% from confidence
    Others -> 0
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


def calculate_praise_emotion_score(emotion_str):
    """
    Calculates score out of 10 based on Praise emotion rules.
    e.g., 'happy 98%' -> parses to 'happy' and 98.
    Happy   -> 70% + balance 30% from confidence
    Neutral -> 40% + balance 30% from confidence
    Others  -> 0
    """
    try:
        parts = emotion_str.split()
        label = parts[0].lower()
        conf = float(parts[1].replace('%', ''))

        if label == "happy":
            return ((70 + 0.30 * conf) / 100.0) * 10
        elif label == "neutral":
            return ((40 + 0.30 * conf) / 100.0) * 10
        else:
            return 0.0
    except Exception:
        return 0.0


def generate_score_and_evidence(job_output_folder: str):
    """
    Reads transcript.json from the job_output_folder, calculates template scores,
    warmup metrics (RAG + Emotion), and praise metrics (RAG + Emotion),
    then writes score.json and evidence.json.
    """
    transcript_path = os.path.join(job_output_folder, "transcript.json")
    if not os.path.exists(transcript_path):
        print(f"[WARNING] Scoring skipped: {transcript_path} not found.")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    segments = transcript_data.get("segments", [])

    # ── Collect segments per category ──────────────────────────────────
    category_indices = {
        "WarmUp": [], "Praise": [], "PSuggest": [],
        "NSuggest": [], "Listen": [], "Direct": []
    }

    warmup_transcripts = []
    warmup_emotion_scores = []
    praise_transcripts = []
    praise_emotion_scores = []

    for i, seg in enumerate(segments):
        label = seg.get("template_label")
        if label in category_indices:
            category_indices[label].append(i)

        if label == "WarmUp":
            warmup_transcripts.append(seg.get("transcript", ""))
            warmup_emotion_scores.append(
                calculate_warmup_emotion_score(seg.get("emotion", "neutral 0%"))
            )
        elif label == "Praise":
            praise_transcripts.append(seg.get("transcript", ""))
            praise_emotion_scores.append(
                calculate_praise_emotion_score(seg.get("emotion", "neutral 0%"))
            )

    # ══════════════════════════════════════════════════════════════════
    # 1. TEMPLATE SCORING (out of 10)
    # ══════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════
    # 2. RAG SCORING — single LLM call for both WarmUp + Praise
    # ══════════════════════════════════════════════════════════════════
    rag_input = {
        "warmup": " ".join(warmup_transcripts),
        "praise": " ".join(praise_transcripts),
    }
    rag_results = evaluate_categories_with_rag(rag_input)

    warmup_rag_score = round(rag_results["warmup"].get("similarity_score_out_of_10", 0.0), 2)
    warmup_rag_suggestions = rag_results["warmup"].get("suggestions", "No suggestions.")

    praise_rag_score = round(rag_results["praise"].get("similarity_score_out_of_10", 0.0), 2)
    praise_rag_suggestions = rag_results["praise"].get("suggestions", "No suggestions.")

    # ══════════════════════════════════════════════════════════════════
    # 3. WARMUP EMOTION SCORING (out of 5)
    # ══════════════════════════════════════════════════════════════════
    warmup_emotion_score = 0.0
    if warmup_emotion_scores:
        warmup_emotion_score = round(sum(warmup_emotion_scores) / len(warmup_emotion_scores), 2)

    total_warmup_score = round(warmup_rag_score + warmup_emotion_score, 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. PRAISE EMOTION SCORING (out of 10)
    # ══════════════════════════════════════════════════════════════════
    praise_emotion_score = 0.0
    if praise_emotion_scores:
        praise_emotion_score = round(sum(praise_emotion_scores) / len(praise_emotion_scores), 2)

    total_praise_score = round(praise_rag_score + praise_emotion_score, 2)

    # ══════════════════════════════════════════════════════════════════
    # 5. BUILD score.json
    # ══════════════════════════════════════════════════════════════════
    score_data = {
        "template_score": template_score_val,
        "warmup_rag_score": warmup_rag_score,
        "warmup_emotion_score": warmup_emotion_score,
        "warmup_total_score": total_warmup_score,
        "praise_rag_score": praise_rag_score,
        "praise_emotion_score": praise_emotion_score,
        "praise_total_score": total_praise_score,
    }

    # ══════════════════════════════════════════════════════════════════
    # 6. BUILD evidence.json
    # ══════════════════════════════════════════════════════════════════
    evidence_data = []

    # — Template Evidence —
    evidence_data.append({
        "category": "template",
        "score": template_score_val,
        "evidence": f"completed categories - [{', '.join(completed)}], missed categories - [{', '.join(missed)}]"
    })

    # — WarmUp Evidence —
    if warmup_rag_score > 7:
        warmup_rag_text = f"Warmup recommendations followed ({int(warmup_rag_score * 10)}%)."
    else:
        warmup_rag_text = (
            f"Warm up recommendation follow percentage {int(warmup_rag_score * 10)}%, "
            f"needs to improve. Suggestions: {warmup_rag_suggestions}"
        )

    if warmup_emotion_score >= 4:
        warmup_tone_text = "Good speaking tone maintained."
    else:
        warmup_tone_text = "Tone must improve."

    evidence_data.append({
        "category": "warmup",
        "score": total_warmup_score,
        "evidence": f"{warmup_rag_text} {warmup_tone_text}".strip()
    })

    # — Praise Evidence —
    praise_rag_pct = int(praise_rag_score * 10)
    if praise_rag_score >= 7:
        praise_rag_text = f"Praise recommendations followed ({praise_rag_pct}%)."
    else:
        praise_rag_text = (
            f"Praise recommendation follow percentage {praise_rag_pct}%, "
            f"needs to improve. Recommendations not followed: {praise_rag_suggestions}"
        )

    if praise_emotion_score >= 7:
        praise_tone_text = "Recommended emotion followed."
    else:
        praise_tone_text = "Tone should be improved for a happy friendly tone."

    evidence_data.append({
        "category": "praise",
        "score": total_praise_score,
        "evidence": f"{praise_rag_text} {praise_tone_text}".strip()
    })

    # ══════════════════════════════════════════════════════════════════
    # 7. WRITE FILES
    # ══════════════════════════════════════════════════════════════════
    score_path = os.path.join(job_output_folder, "score.json")
    evidence_path = os.path.join(job_output_folder, "evidence.json")

    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=4)

    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_data, f, indent=4)

    print(f"[INFO] Score and Evidence saved to {job_output_folder}")
