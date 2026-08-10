"""
Phase 7 — Manifest builders
===========================

Turns a corpus (or your own pipeline output) into the flat manifest that
``evaluate.py`` and, later, ``train.py`` consume.

Two sources
-----------
``from-processed``
    Scans ``processed/*/job_result.json`` and emits a **labelling sheet** —
    every row pre-filled except ``label``, which you fill in by hand.  This
    needs no downloads and, because it is your own meeting audio, it measures
    the domain you actually ship on.  Start here.

``from-meld``
    Builds a manifest from the MELD corpus.  MELD's seven emotion labels map
    exactly onto :data:`EMOTION_LABELS`, and it is freely downloadable, which
    makes it the practical substitute for licence-gated IEMOCAP.

    MELD ships as MP4 video; extract 16 kHz mono WAV first:

        for f in train_splits/*.mp4; do
            ffmpeg -i "$f" -ar 16000 -ac 1 -y "wav/$(basename "${f%.mp4}").wav"
        done

Usage
-----
    python -m emotion.training.build_manifest from-processed \
        --out emotion/data/inhouse.csv --limit 200

    python -m emotion.training.build_manifest from-meld \
        --csv MELD/test_sent_emo.csv --audio-dir MELD/wav \
        --out emotion/data/meld_test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from .dataset import normalize_label

REPO_ROOT = Path(__file__).resolve().parents[2]

SHEET_COLUMNS = ["id", "audio_path", "text", "label", "current_prediction", "current_confidence"]


def _write_sheet(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sample(rows: list[dict], limit: int | None, strategy: str, seed: int) -> list[dict]:
    """
    Reduce ``rows`` to ``limit`` entries.

    ``random`` draws uniformly, which keeps the class priors of real audio and
    therefore gives an unbiased accuracy estimate — but on meeting recordings
    it will hand you a sheet that is mostly neutral, with too few angry/fear
    examples to say anything about those classes.

    ``balanced`` round-robins across the model's *current* predicted class so
    every emotion appears.  Per-class F1 becomes meaningful; overall accuracy
    no longer reflects production traffic.  Read the two numbers accordingly.
    """
    if not limit or limit >= len(rows):
        return rows

    rng = random.Random(seed)
    if strategy == "random":
        return rng.sample(rows, limit)

    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row.get("current_prediction") or "unknown", []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked: list[dict] = []
    order = sorted(buckets)
    while len(picked) < limit and any(buckets[k] for k in order):
        for key in order:
            if buckets[key] and len(picked) < limit:
                picked.append(buckets[key].pop())
    return picked


# ---------------------------------------------------------------------------
# from-processed
# ---------------------------------------------------------------------------

def cmd_from_processed(args: argparse.Namespace) -> int:
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_dir():
        print(f"❌ Not a directory: {processed_dir}")
        return 1

    rows: list[dict] = []
    jobs_seen = 0
    for job_file in sorted(processed_dir.glob("*/job_result.json")):
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"⚠️  Skipping {job_file}: {exc}")
            continue

        jobs_seen += 1
        job_id = job.get("job_id") or job_file.parent.name

        for seg in job.get("segments", []):
            audio_path = seg.get("audio_path") or ""
            text = (seg.get("text") or "").strip()
            if not audio_path or not text:
                continue
            if not (REPO_ROOT / audio_path).is_file() and not Path(audio_path).is_file():
                continue

            rows.append({
                "id": f"{job_id}_{int(seg.get('segment_id', len(rows))):03d}",
                "audio_path": audio_path,
                "text": text,
                "label": "",  # <- you fill this in
                "current_prediction": seg.get("emotion", ""),
                "current_confidence": seg.get("confidence", ""),
            })

    if not rows:
        print(f"❌ No usable segments found under {processed_dir}/*/job_result.json")
        return 1

    total = len(rows)
    rows = _sample(rows, args.limit, args.sample, args.seed)
    _write_sheet(rows, Path(args.out))

    print(f"\n✅ Wrote {len(rows)} rows (of {total} available, from {jobs_seen} job(s)) "
          f"to {args.out}")
    if args.sample == "balanced" and args.limit:
        print("   Sampling: balanced across the model's current predictions — good "
              "per-class coverage, but overall accuracy will not reflect real priors.")
    elif args.limit:
        print("   Sampling: uniform random — unbiased overall accuracy, but rare "
              "emotions may be under-represented.")
    print(
        "\nNext:\n"
        "  1. Open the CSV and fill the `label` column with one of:\n"
        "       angry, disgust, fear, happy, sad, surprise, neutral\n"
        "     Leave a row blank to exclude it. `current_prediction` is only a hint —\n"
        "     label from the audio, or you will simply reproduce the model's bias.\n"
        "  2. Validate:  python -m emotion.training.evaluate "
        f"--manifest {args.out} --dry-run"
    )
    return 0


# ---------------------------------------------------------------------------
# from-meld
# ---------------------------------------------------------------------------

def cmd_from_meld(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    audio_dir = Path(args.audio_dir)

    if not csv_path.is_file():
        print(f"❌ MELD CSV not found: {csv_path}")
        return 1
    if not audio_dir.is_dir():
        print(f"❌ Audio directory not found: {audio_dir}")
        return 1

    # utf-8-sig: MELD's CSVs are commonly BOM-prefixed.
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        meld_rows = list(csv.DictReader(fh))

    rows: list[dict] = []
    missing_audio = 0
    unmapped: dict[str, int] = {}

    for row in meld_rows:
        dia, utt = row.get("Dialogue_ID"), row.get("Utterance_ID")
        if dia is None or utt is None:
            continue
        clip_id = f"dia{dia}_utt{utt}"

        wav = audio_dir / f"{clip_id}.wav"
        if not wav.is_file():
            missing_audio += 1
            continue

        raw = (row.get("Emotion") or "").strip()
        label = normalize_label(raw)
        if label is None:
            unmapped[raw] = unmapped.get(raw, 0) + 1
            continue

        text = (row.get("Utterance") or "").strip()
        if not text:
            continue

        rows.append({
            "id": clip_id,
            "audio_path": str(wav),
            "text": text,
            "label": label,
            "current_prediction": "",
            "current_confidence": "",
        })

    if not rows:
        print("❌ No usable rows. Check that WAVs were extracted into --audio-dir "
              "with names like dia0_utt0.wav")
        return 1

    total = len(rows)
    rows = _sample(rows, args.limit, "random", args.seed)
    _write_sheet(rows, Path(args.out))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    print(f"\n✅ Wrote {len(rows)} labelled rows (of {total} matched) to {args.out}")
    if missing_audio:
        print(f"   ⚠️  {missing_audio} row(s) had no matching WAV and were skipped "
              "(MELD ships a few unreadable clips; extraction gaps look the same).")
    if unmapped:
        detail = ", ".join(f"{k}×{v}" for k, v in sorted(unmapped.items()))
        print(f"   ⚠️  Unmapped labels skipped: {detail}")
    dist = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    print(f"   Label distribution: {dist}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m emotion.training.build_manifest",
        description="Build evaluation/training manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_proc = sub.add_parser(
        "from-processed",
        help="Build a hand-labelling sheet from your own pipeline output.",
    )
    p_proc.add_argument("--processed-dir", default=str(REPO_ROOT / "processed"))
    p_proc.add_argument("--out", required=True, help="Output .csv path.")
    p_proc.add_argument("--limit", type=int, help="Cap the number of rows to label.")
    p_proc.add_argument(
        "--sample", choices=["random", "balanced"], default="random",
        help="How to pick rows when --limit is set (see module docstring).",
    )
    p_proc.add_argument("--seed", type=int, default=13, help="Sampling seed.")
    p_proc.set_defaults(func=cmd_from_processed)

    p_meld = sub.add_parser("from-meld", help="Build a manifest from the MELD corpus.")
    p_meld.add_argument("--csv", required=True, help="e.g. MELD/test_sent_emo.csv")
    p_meld.add_argument("--audio-dir", required=True, help="Directory of extracted WAVs.")
    p_meld.add_argument("--out", required=True, help="Output .csv path.")
    p_meld.add_argument("--limit", type=int, help="Cap the number of rows.")
    p_meld.add_argument("--seed", type=int, default=13, help="Sampling seed.")
    p_meld.set_defaults(func=cmd_from_meld)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
