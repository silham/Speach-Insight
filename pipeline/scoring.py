import json
import os
import sys

# Ensure we can import from rag.py if executed context requires it
try:
    from rag import evaluate_categories_with_rag, generate_report
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag import evaluate_categories_with_rag, generate_report


def _parse_emotion(emotion_str):
    """Parse emotion string like 'happy 98%' into (label, confidence)."""
    try:
        parts = emotion_str.split()
        label = parts[0].lower()
        conf = float(parts[1].replace('%', ''))
        return label, conf
    except Exception:
        return "neutral", 0.0


def calculate_warmup_emotion_score(emotion_str):
    """
    Calculates score out of 5 based on WarmUp emotion rules.
    Happy  -> 80% + balance 20% from confidence
    Neutral-> 50% + balance 30% from confidence
    Sad    -> 30% + balance 20% from confidence
    Others -> 0
    """
    label, conf = _parse_emotion(emotion_str)

    if label == "happy":
        return ((80 + 0.20 * conf) / 100.0) * 5
    elif label == "neutral":
        return ((50 + 0.30 * conf) / 100.0) * 5
    elif label == "sad":
        return ((30 + 0.20 * conf) / 100.0) * 5
    else:
        return 0.0


def calculate_praise_emotion_score(emotion_str):
    """
    Calculates score out of 10 based on Praise emotion rules.
    Happy   -> 70% + balance 30% from confidence
    Neutral -> 40% + balance 30% from confidence
    Others  -> 0
    """
    label, conf = _parse_emotion(emotion_str)

    if label == "happy":
        return ((70 + 0.30 * conf) / 100.0) * 10
    elif label == "neutral":
        return ((40 + 0.30 * conf) / 100.0) * 10
    else:
        return 0.0


def generate_score_and_evidence(job_output_folder: str):
    """
    Reads transcript.json from the job_output_folder, calculates template scores,
    warmup metrics (RAG + Emotion), praise metrics (RAG + Emotion),
    suggest metrics (RAG + balance + tone penalty), listen metrics
    (coverage + RAG), direct metrics (RAG /20), total score /100,
    then writes score.json, evidence.json, and report.json.
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
    praise_emotions_raw = []
    praise_emotion_scores = []
    psuggest_transcripts = []
    nsuggest_transcripts = []
    suggest_emotions_raw = []
    listen_transcripts = []
    direct_transcripts = []

    for i, seg in enumerate(segments):
        label = seg.get("template_label")
        if label in category_indices:
            category_indices[label].append(i)

        emotion_str = seg.get("emotion", "neutral 0%")

        if label == "WarmUp":
            warmup_transcripts.append(seg.get("transcript", ""))
            warmup_emotion_scores.append(
                calculate_warmup_emotion_score(emotion_str)
            )
        elif label == "Praise":
            praise_transcripts.append(seg.get("transcript", ""))
            praise_emotions_raw.append(emotion_str)
            praise_emotion_scores.append(
                calculate_praise_emotion_score(emotion_str)
            )
        elif label == "PSuggest":
            psuggest_transcripts.append(seg.get("transcript", ""))
            suggest_emotions_raw.append(emotion_str)
        elif label == "NSuggest":
            nsuggest_transcripts.append(seg.get("transcript", ""))
            suggest_emotions_raw.append(emotion_str)
        elif label == "Listen":
            listen_transcripts.append(seg.get("transcript", ""))
        elif label == "Direct":
            direct_transcripts.append(seg.get("transcript", ""))

    # ══════════════════════════════════════════════════════════════════
    # 1. TEMPLATE SCORING (out of 10)
    #    WarmUp must be at the beginning, Praise must follow WarmUp
    # ══════════════════════════════════════════════════════════════════
    completed = []
    missed = []

    first_warmup_idx = float('inf')
    warmup_at_beginning = False

    if category_indices["WarmUp"]:
        first_warmup_idx = min(category_indices["WarmUp"])
        # Check if WarmUp is at the beginning: no non-WarmUp segments before first WarmUp
        non_warmup_before = [
            i for i in range(first_warmup_idx)
            if segments[i].get("template_label") in category_indices
            and segments[i].get("template_label") != "WarmUp"
        ]
        if len(non_warmup_before) == 0:
            warmup_at_beginning = True
            completed.append("WarmUp")
        else:
            # WarmUp exists but not at the beginning — partial credit
            missed.append("WarmUp (not at beginning)")
    else:
        missed.append("WarmUp")

    # Praise must come after WarmUp
    if category_indices["Praise"] and any(idx > first_warmup_idx for idx in category_indices["Praise"]):
        completed.append("Praise")
    else:
        if category_indices["Praise"]:
            missed.append("Praise (not after WarmUp)")
        else:
            missed.append("Praise")

    for cat in ["PSuggest", "NSuggest", "Listen", "Direct"]:
        if category_indices[cat]:
            completed.append(cat)
        else:
            missed.append(cat)

    template_score_val = round((len(completed) / 6.0) * 10.0, 2)

    # ══════════════════════════════════════════════════════════════════
    # 2. RAG SCORING — single LLM call for all categories (incl. Direct)
    # ══════════════════════════════════════════════════════════════════
    all_suggest_transcripts = psuggest_transcripts + nsuggest_transcripts
    rag_input = {
        "warmup": " ".join(warmup_transcripts),
        "praise": " ".join(praise_transcripts),
        "suggest": " ".join(all_suggest_transcripts),
        "listen": " ".join(listen_transcripts),
        "direct": " ".join(direct_transcripts),
    }
    rag_results = evaluate_categories_with_rag(rag_input)

    warmup_rag_score = round(rag_results["warmup"].get("similarity_score_out_of_10", 0.0), 2)
    warmup_rag_suggestions = rag_results["warmup"].get("suggestions", "No suggestions.")

    praise_rag_score = round(rag_results["praise"].get("similarity_score_out_of_10", 0.0), 2)
    praise_rag_suggestions = rag_results["praise"].get("suggestions", "No suggestions.")

    # Suggest RAG is scored out of 10 from LLM, then scaled to 20
    suggest_rag_raw = round(rag_results["suggest"].get("similarity_score_out_of_10", 0.0), 2)
    suggest_rag_score = round(suggest_rag_raw * 2.0, 2)
    suggest_rag_suggestions = rag_results["suggest"].get("suggestions", "No suggestions.")

    # Listen RAG is scored out of 10 from LLM, then scaled to 8
    listen_rag_raw = round(rag_results["listen"].get("similarity_score_out_of_10", 0.0), 2)
    listen_rag_score = round(listen_rag_raw * 0.8, 2)
    listen_rag_suggestions = rag_results["listen"].get("suggestions", "No suggestions.")

    # Direct RAG is scored out of 10 from LLM, then scaled to 20
    direct_rag_raw = round(rag_results["direct"].get("similarity_score_out_of_10", 0.0), 2)
    direct_score = round(direct_rag_raw * 2.0, 2)
    direct_suggestions = rag_results["direct"].get("suggestions", "No suggestions.")

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

    # Check for sad/angry tones in praise segments
    praise_bad_tones = []
    for emo_str in praise_emotions_raw:
        lbl, _ = _parse_emotion(emo_str)
        if lbl in ("sad", "angry"):
            praise_bad_tones.append(lbl)

    total_praise_score = round(praise_rag_score + praise_emotion_score, 2)

    # ══════════════════════════════════════════════════════════════════
    # 5. SUGGEST SCORING (out of 20)
    # ══════════════════════════════════════════════════════════════════
    total_suggest_count = len(psuggest_transcripts) + len(nsuggest_transcripts)
    suggest_balance_penalty = 0
    balance_evidence_parts = []

    if total_suggest_count > 0:
        psuggest_pct = len(psuggest_transcripts) / total_suggest_count * 100
        nsuggest_pct = len(nsuggest_transcripts) / total_suggest_count * 100

        if psuggest_pct < 30:
            suggest_balance_penalty += 5
            balance_evidence_parts.append(
                f"Too much NSuggest (negative suggestions) -- PSuggest is only {psuggest_pct:.0f}%."
            )
        if nsuggest_pct < 30:
            suggest_balance_penalty += 5
            balance_evidence_parts.append(
                f"Too much PSuggest (positive suggestions) -- NSuggest is only {nsuggest_pct:.0f}%."
            )

    # Check for angry tone in suggest segments
    suggest_angry_penalty = 0
    suggest_angry_found = False
    for emo_str in suggest_emotions_raw:
        lbl, _ = _parse_emotion(emo_str)
        if lbl == "angry":
            suggest_angry_found = True
            break

    if suggest_angry_found:
        suggest_angry_penalty = 10

    suggest_total_score = max(0.0, round(suggest_rag_score - suggest_balance_penalty - suggest_angry_penalty, 2))

    # ══════════════════════════════════════════════════════════════════
    # 6. LISTEN SCORING (out of 15: coverage /7 + RAG /8)
    # ══════════════════════════════════════════════════════════════════
    total_segments = len(segments)
    listen_count = len(listen_transcripts)
    listen_coverage_pct = (listen_count / total_segments * 100) if total_segments > 0 else 0

    if listen_coverage_pct >= 10:
        listen_coverage_score = 7.0
    else:
        listen_coverage_score = round((listen_coverage_pct / 10.0) * 7.0, 2)

    listen_total_score = round(listen_coverage_score + listen_rag_score, 2)

    # ══════════════════════════════════════════════════════════════════
    # 7. TOTAL SCORE (out of 100)
    #    Template(10) + WarmUp(15) + Praise(20) + Suggest(20)
    #    + Listen(15) + Direct(20)
    # ══════════════════════════════════════════════════════════════════
    total_score = round(
        template_score_val + total_warmup_score + total_praise_score
        + suggest_total_score + listen_total_score + direct_score, 2
    )
    total_score = min(100.0, total_score)

    # ══════════════════════════════════════════════════════════════════
    # 8. BUILD score.json
    # ══════════════════════════════════════════════════════════════════
    score_data = {
        "template_score": template_score_val,
        "warmup_rag_score": warmup_rag_score,
        "warmup_emotion_score": warmup_emotion_score,
        "warmup_total_score": total_warmup_score,
        "praise_rag_score": praise_rag_score,
        "praise_emotion_score": praise_emotion_score,
        "praise_total_score": total_praise_score,
        "suggest_rag_score": suggest_rag_score,
        "suggest_balance_penalty": suggest_balance_penalty,
        "suggest_angry_penalty": suggest_angry_penalty,
        "suggest_total_score": suggest_total_score,
        "listen_rag_score": listen_rag_score,
        "listen_coverage_score": listen_coverage_score,
        "listen_total_score": listen_total_score,
        "direct_score": direct_score,
        "total_score": total_score,
    }

    # ══════════════════════════════════════════════════════════════════
    # 9. BUILD evidence.json
    # ══════════════════════════════════════════════════════════════════
    evidence_data = []

    # — Template Evidence —
    template_order_note = ""
    if not warmup_at_beginning and category_indices["WarmUp"]:
        template_order_note = " WarmUp was not at the beginning of the conversation."
    evidence_data.append({
        "category": "template",
        "score": template_score_val,
        "evidence": (
            f"completed categories - [{', '.join(completed)}], "
            f"missed categories - [{', '.join(missed)}].{template_order_note}"
        )
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

    # Flag sad/angry tones in praise
    if praise_bad_tones:
        unique_bad = sorted(set(praise_bad_tones))
        praise_tone_text += (
            f" It is encouraged to use a friendly tone; "
            f"{', '.join(unique_bad)} tone(s) are not recommended for praise."
        )

    evidence_data.append({
        "category": "praise",
        "score": total_praise_score,
        "evidence": f"{praise_rag_text} {praise_tone_text}".strip()
    })

    # — Suggest Evidence —
    suggest_rag_pct = int(suggest_rag_raw * 10)
    if suggest_rag_score >= 14:   # 70% of 20
        suggest_rag_text = f"Suggest recommendations followed ({suggest_rag_pct}%)."
    else:
        suggest_rag_text = (
            f"Suggest recommendation follow percentage {suggest_rag_pct}%, "
            f"needs to improve. Suggestions: {suggest_rag_suggestions}"
        )

    suggest_evidence_parts = [suggest_rag_text]

    if balance_evidence_parts:
        suggest_evidence_parts.extend(balance_evidence_parts)

    if suggest_angry_found:
        suggest_evidence_parts.append(
            "Angry tone detected. It is always encouraged to use a friendly tone; "
            "angry tone is not recommended."
        )

    evidence_data.append({
        "category": "suggest",
        "score": suggest_total_score,
        "evidence": " ".join(suggest_evidence_parts).strip()
    })

    # — Listen Evidence —
    listen_rag_pct = int(listen_rag_raw * 10)
    listen_evidence_parts = []

    if listen_rag_score >= 5.6:   # 70% of 8
        listen_evidence_parts.append(f"Listen recommendations followed ({listen_rag_pct}%).")
    else:
        listen_evidence_parts.append(
            f"Listen recommendation follow percentage {listen_rag_pct}%, "
            f"needs to improve. Suggestions: {listen_rag_suggestions}"
        )

    if listen_coverage_pct >= 10:
        listen_evidence_parts.append(
            f"Listening coverage is {listen_coverage_pct:.0f}% of total segments (meets 10% threshold)."
        )
    else:
        listen_evidence_parts.append(
            f"Listening coverage is only {listen_coverage_pct:.0f}% of total segments (below 10% threshold). "
            "It is always encouraged to ask feedback from the junior and listen to him/her actively."
        )

    evidence_data.append({
        "category": "listen",
        "score": listen_total_score,
        "evidence": " ".join(listen_evidence_parts).strip()
    })

    # — Direct Evidence —
    direct_rag_pct = int(direct_rag_raw * 10)
    if direct_score >= 14:   # 70% of 20
        direct_rag_text = f"Direct recommendations followed ({direct_rag_pct}%). Directions are clear and practical."
    else:
        direct_rag_text = (
            f"Direct recommendation follow percentage {direct_rag_pct}%, "
            f"needs to improve. Directions should be clearer and more practical. "
            f"Suggestions: {direct_suggestions}"
        )

    evidence_data.append({
        "category": "direct",
        "score": direct_score,
        "evidence": direct_rag_text
    })

    # ══════════════════════════════════════════════════════════════════
    # 10. WRITE score.json and evidence.json
    # ══════════════════════════════════════════════════════════════════
    score_path = os.path.join(job_output_folder, "score.json")
    evidence_path = os.path.join(job_output_folder, "evidence.json")

    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=4)

    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_data, f, indent=4)

    print(f"[INFO] Score and Evidence saved to {job_output_folder}")

    # ══════════════════════════════════════════════════════════════════
    # 11. GENERATE report.json (2nd and final LLM call)
    # ══════════════════════════════════════════════════════════════════
    try:
        report_data = generate_report(evidence_data, score_data)
        report_path = os.path.join(job_output_folder, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        print(f"[INFO] Report saved to {report_path}")
    except Exception as exc:
        print(f"[WARNING] Failed to generate report: {exc}")
