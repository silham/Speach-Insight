import json
import os

def generate_score_and_evidence(job_output_folder: str):
    """
    Reads transcript.json from the job_output_folder, calculates a template score,
    and writes score.json and evidence.json to the same folder.
    """
    transcript_path = os.path.join(job_output_folder, "transcript.json")
    if not os.path.exists(transcript_path):
        print(f"⚠️  Scoring skipped: {transcript_path} not found.")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    segments = transcript_data.get("segments", [])

    # Collect indices for each category
    category_indices = {
        "WarmUp": [],
        "Praise": [],
        "PSuggest": [],
        "NSuggest": [],
        "Listen": [],
        "Direct": []
    }

    for i, seg in enumerate(segments):
        label = seg.get("template_label")
        if label in category_indices:
            category_indices[label].append(i)

    completed = []
    missed = []

    # 1. WarmUp
    first_warmup_idx = float('inf')
    if category_indices["WarmUp"]:
        completed.append("WarmUp")
        first_warmup_idx = min(category_indices["WarmUp"])
    else:
        missed.append("WarmUp")

    # 2. Praise (must be after at least one WarmUp)
    if category_indices["Praise"] and any(idx > first_warmup_idx for idx in category_indices["Praise"]):
        completed.append("Praise")
    else:
        missed.append("Praise")

    # 3-6. Any order is fine
    for cat in ["PSuggest", "NSuggest", "Listen", "Direct"]:
        if category_indices[cat]:
            completed.append(cat)
        else:
            missed.append(cat)

    # Calculate score
    # Number of completed categories / 6 * 10
    score_val = (len(completed) / 6.0) * 10.0
    score_val = round(score_val, 2)

    evidence_data = {
        "category": "template",
        "score": score_val,
        "evidence": f"completed categories - [{', '.join(completed)}], missed categories - [{', '.join(missed)}]"
    }

    score_data = {
        "score": score_val
    }

    score_path = os.path.join(job_output_folder, "score.json")
    evidence_path = os.path.join(job_output_folder, "evidence.json")

    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=4)
    
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_data, f, indent=4)

    print(f"[INFO] Score and Evidence saved to {job_output_folder}")
