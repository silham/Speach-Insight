"""
pipeline/lead_speaker/__init__.py
===================================
Lead Speaker Identification Module — Phase 8
=============================================

PURPOSE
-------
Given the fully-enriched :class:`JobResult` (speaker turns, transcripts,
emotion data), determine which participant is the "lead speaker" — the
person who drives, moderates, or dominates the conversation.

CONTRACT
--------
Every implementation (stub or trained) must subclass
:class:`LeadSpeakerIdentifier` and implement exactly one method::

    def identify(self, job: JobResult) -> str | None

Where the return value is a Pyannote speaker label (e.g. ``"SPEAKER_00"``)
or ``None`` if the lead cannot be determined.

The pipeline calls this method AFTER all segments have been transcribed
and emotion-analysed, so the full :class:`JobResult` is available including:

- ``job.segments``                      — list of SegmentResult
- ``job.speaker_talk_times()``          — {speaker: total_seconds}  ← fast baseline
- ``seg.text``                          — transcript per turn
- ``seg.emotion`` / ``seg.confidence``  — emotion per turn
- ``seg.paralinguistic``                — {"pitch", "energy", "speaking_rate"}
- ``seg.vader``                         — sentiment scores

WHAT TO BUILD NEXT (for the incoming developer)
-----------------------------------------------
The :class:`StubLeadSpeakerIdentifier` below uses a simple heuristic
(most talking time wins).  Replace it with a real model by:

1. Subclass ``LeadSpeakerIdentifier``.
2. Implement ``identify(job)`` — the full ``JobResult`` is available.
3. Pass an instance of your class to ``AnalysisPipeline(lead_speaker=...)``.

Suggested features for a trained classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-speaker aggregates (combine into a feature vector, one row per speaker):

    - total_talk_time          (from speaker_talk_times())
    - turn_count               (number of segments)
    - avg_segment_duration     (mean turn length)
    - question_ratio           (turns ending with "?")
    - avg_speaking_rate        (from paralinguistic["speaking_rate"])
    - avg_pitch                (from paralinguistic["pitch"])
    - emotion_distribution     (7-dim vector of mean emotion probs)
    - avg_vader_compound       (sentiment valence)
    - interruption_count       (turns < 1 s after previous speaker — needs timestamps)

A lightweight logistic regression or small MLP trained on labelled meeting
corpora (e.g. AMI, ICSI) can achieve good accuracy.  The timestamps are
already available in ``seg.start_time`` / ``seg.end_time``.

See ``Dataflow.md`` — "Adding a New Model" for integration steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pipeline.schemas import JobResult


# ---------------------------------------------------------------------------
# Abstract base — all implementations must honour this interface
# ---------------------------------------------------------------------------

class LeadSpeakerIdentifier(ABC):
    """
    Abstract base class for lead-speaker identification.

    Subclass this, implement ``identify()``, and pass your instance to
    ``AnalysisPipeline(lead_speaker=<your instance>)``.
    """

    @abstractmethod
    def identify(self, job: "JobResult") -> Optional[str]:
        """
        Inspect *job* and return the Pyannote label of the lead speaker,
        or ``None`` if a lead cannot be identified.

        Parameters
        ----------
        job :
            A fully-processed :class:`JobResult`.  All transcription and
            emotion fields on each segment are populated at this point.

        Returns
        -------
        str | None
            Speaker label (e.g. ``"SPEAKER_00"``) or ``None``.
        """
        ...


# ---------------------------------------------------------------------------
# Stub implementation — ships with the repo, works out of the box
# ---------------------------------------------------------------------------

class StubLeadSpeakerIdentifier(LeadSpeakerIdentifier):
    """
    Baseline heuristic: the speaker with the most total speaking time is
    declared the lead.

    This is intentionally simple so the pipeline has a working value in
    ``job.lead_speaker`` from day one.  Replace with a trained model when
    ready — the interface does not change.
    """

    def identify(self, job: "JobResult") -> Optional[str]:
        talk_times = job.speaker_talk_times()
        if not talk_times:
            return None
        lead = max(talk_times, key=lambda s: talk_times[s])
        return lead


# ---------------------------------------------------------------------------
# Convenience re-export
# ---------------------------------------------------------------------------

__all__ = [
    "LeadSpeakerIdentifier",
    "StubLeadSpeakerIdentifier",
]
