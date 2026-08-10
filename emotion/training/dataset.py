"""
Phase 7 — Dataset / manifest handling
=====================================

Everything downstream (evaluation now, training later) reads a **manifest**:
a flat list of ``(id, audio_path, text, label)`` rows.  Keeping the corpus
behind this one format means the evaluator does not care whether the rows
came from MELD, IEMOCAP, or clips you labelled by hand.

Two on-disk formats are accepted, chosen by file extension:

* ``.csv``   — best for hand-labelling (opens in Excel / Numbers / Sheets)
* ``.jsonl`` — best for generated corpora (one JSON object per line)

Required columns/keys: ``audio_path``, ``text``, ``label``.
Optional: ``id`` (falls back to the audio filename).

A blank ``label`` marks a row as *not yet labelled* — it is loaded but
excluded from scoring, and the evaluator reports how many it skipped.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..emotion_classifier import EMOTION_LABELS

# Canonical label -> index, matching the classifier's output ordering exactly.
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(EMOTION_LABELS)}

# Corpus-specific spellings mapped onto our Ekman 7.
#   MELD      : anger, disgust, fear, joy, sadness, surprise, neutral
#   IEMOCAP   : ang, dis, fea, hap, sad, sur, neu, exc (+ fru/oth/xxx, dropped)
#   CREMA-D   : ANG, DIS, FEA, HAP, SAD, NEU
#   RAVDESS   : angry, disgust, fearful, happy, sad, surprised, neutral, calm
LABEL_ALIASES: dict[str, str] = {
    # angry
    "ang": "angry", "anger": "angry", "angry": "angry", "mad": "angry",
    # disgust
    "dis": "disgust", "disgust": "disgust", "disgusted": "disgust",
    # fear
    "fea": "fear", "fear": "fear", "fearful": "fear", "scared": "fear",
    "afraid": "fear", "anxious": "fear",
    # happy
    "hap": "happy", "happy": "happy", "happiness": "happy", "joy": "happy",
    "joyful": "happy", "exc": "happy", "excited": "happy", "excitement": "happy",
    # sad
    "sad": "sad", "sadness": "sad", "sorrow": "sad", "depressed": "sad",
    # surprise
    "sur": "surprise", "surprise": "surprise", "surprised": "surprise",
    "surprising": "surprise",
    # neutral
    "neu": "neutral", "neutral": "neutral", "calm": "neutral", "normal": "neutral",
}

# Labels that exist in source corpora but have no Ekman-7 home.  Rows carrying
# these are dropped with a counted reason rather than silently mangled.
DROP_LABELS: frozenset[str] = frozenset({
    "fru", "frustrated", "frustration",
    "oth", "other", "xxx", "unknown", "none", "nan",
})


def normalize_label(raw: str | None) -> str | None:
    """
    Map a corpus label onto one of :data:`EMOTION_LABELS`.

    Returns ``None`` when the row should not be scored — either because it is
    unlabelled (blank) or because the label has no Ekman-7 equivalent
    (e.g. IEMOCAP's ``frustrated``).  Callers are expected to count these
    separately; see :func:`load_manifest`.
    """
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if not key or key in DROP_LABELS:
        return None
    return LABEL_ALIASES.get(key)


@dataclass
class EmotionSample:
    """One scorable utterance: an audio clip, its transcript, and a gold label."""

    id: str
    audio_path: str
    text: str
    label: str | None = None          # canonical Ekman-7 label, or None if unlabelled
    raw_label: str | None = None      # whatever the manifest actually said
    extra: dict = field(default_factory=dict)

    @property
    def label_index(self) -> int | None:
        return LABEL_TO_INDEX[self.label] if self.label else None


def _rows_from_file(path: Path) -> list[dict]:
    """Read a .csv or .jsonl manifest into raw dicts."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(
        f"Unsupported manifest format {suffix!r} (expected .csv or .jsonl): {path}"
    )


def _resolve_audio(raw_path: str, manifest_dir: Path, root: Path) -> tuple[str, bool]:
    """
    Resolve a manifest's ``audio_path`` against the repo root, then against the
    manifest's own directory.  Returns ``(resolved_path, exists)``.
    """
    p = Path(raw_path)
    if p.is_absolute():
        return str(p), p.is_file()
    for base in (root, manifest_dir):
        candidate = base / p
        if candidate.is_file():
            return str(candidate), True
    # Nothing found — report the repo-root interpretation for a legible error.
    return str(root / p), False


def load_manifest(
    path: str | Path,
    root: str | Path | None = None,
    require_audio: bool = True,
) -> tuple[list[EmotionSample], dict]:
    """
    Load a manifest into :class:`EmotionSample` objects.

    Parameters
    ----------
    path :
        Manifest file (``.csv`` or ``.jsonl``).
    root :
        Base directory for relative ``audio_path`` values.  Defaults to the
        repository root (the parent of the ``emotion/`` package).
    require_audio :
        Drop rows whose audio file is missing.  Disable when you only want to
        inspect the label distribution.

    Returns
    -------
    (samples, stats)
        ``samples`` contains only rows that are usable *and* labelled.
        ``stats`` records every exclusion so nothing disappears silently.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    repo_root = Path(root) if root else Path(__file__).resolve().parents[2]
    manifest_dir = path.resolve().parent

    rows = _rows_from_file(path)
    samples: list[EmotionSample] = []
    stats = {
        "total_rows": len(rows),
        "kept": 0,
        "unlabelled": 0,
        "unmappable_label": 0,
        "missing_audio": 0,
        "empty_text": 0,
        "unmappable_examples": [],
        "missing_audio_examples": [],
        "label_counts": {},
    }

    for i, row in enumerate(rows):
        raw_label = (row.get("label") or "").strip()
        raw_audio = (row.get("audio_path") or "").strip()
        text = (row.get("text") or "").strip()
        sample_id = (row.get("id") or "").strip() or (Path(raw_audio).stem or f"row{i}")

        if not raw_label:
            stats["unlabelled"] += 1
            continue

        label = normalize_label(raw_label)
        if label is None:
            stats["unmappable_label"] += 1
            if len(stats["unmappable_examples"]) < 5:
                stats["unmappable_examples"].append(raw_label)
            continue

        resolved, exists = _resolve_audio(raw_audio, manifest_dir, repo_root)
        if require_audio and not exists:
            stats["missing_audio"] += 1
            if len(stats["missing_audio_examples"]) < 5:
                stats["missing_audio_examples"].append(raw_audio)
            continue

        # The linguistic branch needs *something*; an empty transcript would
        # score the text encoder on a bare [CLS]/[SEP] pair.
        if not text:
            stats["empty_text"] += 1
            continue

        samples.append(
            EmotionSample(
                id=sample_id,
                audio_path=resolved,
                text=text,
                label=label,
                raw_label=raw_label,
            )
        )
        stats["label_counts"][label] = stats["label_counts"].get(label, 0) + 1

    stats["kept"] = len(samples)
    return samples, stats


def write_manifest(samples: list[EmotionSample], path: str | Path) -> None:
    """Write samples back out, format chosen by ``path``'s extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "audio_path", "text", "label"])
            writer.writeheader()
            for s in samples:
                writer.writerow({
                    "id": s.id,
                    "audio_path": s.audio_path,
                    "text": s.text,
                    "label": s.label or "",
                })
    else:
        with path.open("w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps({
                    "id": s.id,
                    "audio_path": s.audio_path,
                    "text": s.text,
                    "label": s.label or "",
                }) + "\n")


def format_stats(stats: dict) -> str:
    """Human-readable summary of what :func:`load_manifest` kept and dropped."""
    lines = [
        f"  rows in manifest : {stats['total_rows']}",
        f"  usable + labelled: {stats['kept']}",
    ]
    for key, caption in (
        ("unlabelled", "skipped (no label)"),
        ("unmappable_label", "skipped (label not in Ekman-7)"),
        ("missing_audio", "skipped (audio file missing)"),
        ("empty_text", "skipped (empty transcript)"),
    ):
        if stats.get(key):
            lines.append(f"  {caption:<30}: {stats[key]}")
    if stats.get("unmappable_examples"):
        lines.append(f"    e.g. {', '.join(stats['unmappable_examples'])}")
    if stats.get("missing_audio_examples"):
        lines.append(f"    e.g. {', '.join(stats['missing_audio_examples'][:3])}")
    if stats.get("label_counts"):
        dist = ", ".join(
            f"{k}={v}" for k, v in sorted(stats["label_counts"].items(), key=lambda kv: -kv[1])
        )
        lines.append(f"  label distribution: {dist}")
    return "\n".join(lines)
