"""
pipeline/schemas.py
===================
Shared data structures that flow through the entire SpeechInSight pipeline.

All models (diarization, transcription, emotion, lead speaker, and any future
additions) read from and write into these dataclasses.  Nothing else is passed
between stages — keeping the contract explicit and serialisation trivial.

Data flow overview
------------------
                  ┌─────────────────────────────────────────┐
  raw audio  ──►  │  Diarization (Pyannote)                  │
                  │  produces: list[SegmentMeta]             │
                  └────────────────┬────────────────────────┘
                                   │  (one per speaker turn)
                  ┌────────────────▼────────────────────────┐
                  │  Transcription (Wav2Vec2 CTC)            │
                  │  fills:  SegmentResult.text              │
                  └────────────────┬────────────────────────┘
                                   │
                  ┌────────────────▼────────────────────────┐
                  │  Emotion Recognition (Multimodal)        │
                  │  fills:  SegmentResult.emotion / conf…   │
                  └────────────────┬────────────────────────┘
                                   │  (all segments collected)
                  ┌────────────────▼────────────────────────┐
                  │  Lead Speaker Identification  [Phase 8]  │
                  │  fills:  JobResult.lead_speaker          │
                  └────────────────┬────────────────────────┘
                                   │
                  ┌────────────────▼────────────────────────┐
                  │  <<future models>>                       │
                  │  e.g. topic modelling, sentiment trend   │
                  └─────────────────────────────────────────┘
                                   │
                             JobResult (JSON)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Segment-level: one speaker turn
# ---------------------------------------------------------------------------

@dataclass
class SegmentMeta:
    """
    Raw output from the diarization stage — everything known BEFORE any ML
    inference.  Passed into transcription and emotion stages.

    Fields
    ------
    segment_id   : sequential index (0-based) in the original audio
    speaker      : Pyannote label, e.g. "SPEAKER_00"
    start_time   : turn start in seconds (float)
    end_time     : turn end in seconds (float)
    audio_path   : absolute local path to the saved .wav clip
    audio_url    : public URL served by the FastAPI /audio mount
    """

    segment_id: int
    speaker: str
    start_time: float
    end_time: float
    audio_path: str
    audio_url: str

    @property
    def duration(self) -> float:
        """Length of this segment in seconds."""
        return self.end_time - self.start_time


@dataclass
class SegmentResult:
    """
    Fully-enriched segment after all inference stages have run.

    Built in stages:
      1. Populated from SegmentMeta  (diarization)
      2. .text filled               (transcription)
      3. .emotion* / .vader / .paralinguistic filled  (emotion recognition)

    Future models append extra fields to .extras rather than to this class
    so existing consumers are never broken.
    """

    # ── from diarization ──────────────────────────────────────────────
    segment_id: int
    speaker: str
    start_time: float
    end_time: float
    audio_path: str
    audio_url: str

    # ── from transcription ────────────────────────────────────────────
    text: str = ""

    # ── from emotion recognition ──────────────────────────────────────
    emotion: str = "neutral"
    confidence: float = 0.0
    all_emotions: dict[str, float] = field(default_factory=dict)
    sarcasm: bool = False
    sarcasm_score: float = 0.0
    ambiguity_score: float = 0.0
    vader: dict[str, float] = field(default_factory=dict)
    paralinguistic: dict[str, float] = field(default_factory=dict)

    # ── extensibility hook: future models write here ──────────────────
    # e.g.  result.extras["lead_score"] = 0.92
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @classmethod
    def from_meta(cls, meta: SegmentMeta) -> "SegmentResult":
        """Initialise a SegmentResult from a SegmentMeta (diarization output)."""
        return cls(
            segment_id=meta.segment_id,
            speaker=meta.speaker,
            start_time=meta.start_time,
            end_time=meta.end_time,
            audio_path=meta.audio_path,
            audio_url=meta.audio_url,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flatten extras into the top level for JSON consumers
        d.update(d.pop("extras", {}))
        return d


# ---------------------------------------------------------------------------
# Job-level: one uploaded file
# ---------------------------------------------------------------------------

@dataclass
class JobResult:
    """
    The complete result for one analysis job.

    Built in stages:
      1. job_id set, segments list populated from diarization
      2. Each segment enriched by transcription + emotion
      3. lead_speaker populated by the lead-speaker model
      4. Future models may add keys to .metadata

    Serialised to JSON and returned from the /analyze endpoint.
    """

    job_id: str
    segments: list[SegmentResult] = field(default_factory=list)

    # ── from lead speaker identification  [Phase 8] ───────────────────
    lead_speaker: Optional[str] = None

    # ── aggregate stats (computed on finalise()) ──────────────────────
    total_speakers: int = 0
    total_segments: int = 0
    total_duration: float = 0.0

    # ── extensibility hook ────────────────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def finalise(self) -> "JobResult":
        """
        Compute aggregate fields after all segments have been processed.
        Call this before serialising to JSON.
        """
        self.total_segments = len(self.segments)
        self.total_speakers = len({s.speaker for s in self.segments})
        self.total_duration = round(sum(s.duration for s in self.segments), 3)
        return self

    def speaker_talk_times(self) -> dict[str, float]:
        """
        Return {speaker_label: total_seconds} for every speaker in the job.
        Useful for the lead-speaker stage and UI summaries.
        """
        times: dict[str, float] = {}
        for seg in self.segments:
            times[seg.speaker] = round(times.get(seg.speaker, 0.0) + seg.duration, 3)
        return times

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flatten segment extras
        d["segments"] = [s.to_dict() for s in self.segments]
        return d
