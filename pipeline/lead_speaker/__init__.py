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
# Role-Based Leader Resolution Engine
# ---------------------------------------------------------------------------

class RoleBasedLeadSpeakerIdentifier(LeadSpeakerIdentifier):
    """
    Intelligent Leader Resolution Engine.
    Operates strictly post-inference on canonical final role predictions.
    Resolves meeting-level ambiguity deterministically.
    """

    def __init__(self, probability_tolerance: float = 0.05):
        """
        Initialize the resolver with a configurable probability tolerance.

        Parameters
        ----------
        probability_tolerance : float, optional
            Tolerance within which two leader probabilities are considered effectively tied.
            Default is 0.05 (5%).
        """
        self.probability_tolerance = probability_tolerance

    def identify(self, job: "JobResult") -> Optional[str]:
        talk_times = job.speaker_talk_times()
        if not talk_times:
            # Set resolution metadata for empty case
            job.metadata["leader_resolution"] = {
                "required": False,
                "candidate_count": 0,
                "candidate_speakers": [],
                "selected": None,
                "method": "none",
                "reason": "No speaking duration or segments found. No leader selected."
            }
            return None

        speakers = list(talk_times.keys())
        if len(speakers) == 1:
            job.metadata["leader_resolution"] = {
                "required": False,
                "candidate_count": 1,
                "candidate_speakers": speakers,
                "selected": speakers[0],
                "method": "exact_match",
                "reason": f"Only one speaker ({speakers[0]}) is present in the meeting. No resolution required."
            }
            return speakers[0]

        # Fetch finalized role predictions
        roles_info = getattr(job, "speaker_roles", {}) or {}

        # 1. Filter candidates whose final role is "Leader"
        leader_candidates = []
        for spk in speakers:
            r_info = roles_info.get(spk, {})
            # Normalized roles: Leader, HR, Junior, Other
            if r_info.get("final_role") == "Leader" or r_info.get("role") == "Leader":
                leader_candidates.append(spk)

        # Helper to extract Leader probability
        def get_leader_prob(spk):
            r_info = roles_info.get(spk, {})
            probs = r_info.get("probs", {}) or {}
            # Probability of "Leader" role. Fallback to probability if distribution missing.
            prob = probs.get("Leader", probs.get("manager", probs.get("Lead", 0.0)))
            if prob == 0.0:
                prob = r_info.get("probability", 0.0)
            return float(prob)

        # Case A: Exactly One Leader
        if len(leader_candidates) == 1:
            selected_leader = leader_candidates[0]
            job.metadata["leader_resolution"] = {
                "required": False,
                "candidate_count": 1,
                "candidate_speakers": leader_candidates,
                "selected": selected_leader,
                "method": "exact_match",
                "reason": f"Exactly one speaker was predicted as Leader ({selected_leader}). No resolution required."
            }
            return selected_leader

        # Case B: Multiple Leaders or Case C: No Leaders
        is_no_leader = len(leader_candidates) == 0
        candidates = speakers if is_no_leader else leader_candidates
        candidate_count = len(candidates)

        # Sort candidates by Leader probability descending
        candidates_with_probs = [(spk, get_leader_prob(spk)) for spk in candidates]
        candidates_with_probs.sort(key=lambda x: x[1], reverse=True)

        top_spk, top_prob = candidates_with_probs[0]

        if len(candidates_with_probs) == 1:
            selected_leader = top_spk
            job.metadata["leader_resolution"] = {
                "required": not is_no_leader,
                "candidate_count": candidate_count,
                "candidate_speakers": candidates,
                "selected": selected_leader,
                "method": "probability_margin",
                "reason": f"Only one candidate resolved. Selected {top_spk}."
            }
            return selected_leader

        runner_up_spk, runner_up_prob = candidates_with_probs[1]

        # Rule 1: Probability Margin Check
        if abs(top_prob - runner_up_prob) > self.probability_tolerance:
            selected_leader = top_spk
            reason = (
                "Selected speaker with the highest Leader probability margin: "
                f"{top_spk} ({top_prob * 100:.1f}%) vs {runner_up_spk} ({runner_up_prob * 100:.1f}%)."
            )
            job.metadata["leader_resolution"] = {
                "required": not is_no_leader,
                "candidate_count": candidate_count,
                "candidate_speakers": candidates,
                "selected": selected_leader,
                "method": "probability_margin",
                "reason": reason
            }
            return selected_leader

        # Find all candidates tied within probability_tolerance of the top probability
        tied_candidates = [spk for spk, prob in candidates_with_probs if abs(top_prob - prob) <= self.probability_tolerance]

        # Rule 2: Conversation Structure Check
        if job.segments:
            first_spk = job.segments[0].speaker
            last_spk = job.segments[-1].speaker
            second_spk = job.segments[1].speaker if len(job.segments) > 1 else None
            second_last_spk = job.segments[-2].speaker if len(job.segments) > 1 else None

            for check_spk in [first_spk, last_spk, second_spk, second_last_spk]:
                if check_spk in tied_candidates:
                    selected_leader = check_spk
                    reason = (
                        f"Leader probabilities were within tolerance ({self.probability_tolerance * 100:.1f}%). "
                        f"Conversation structure selected {check_spk} based on turn sequence."
                    )
                    job.metadata["leader_resolution"] = {
                        "required": not is_no_leader,
                        "candidate_count": candidate_count,
                        "candidate_speakers": candidates,
                        "selected": selected_leader,
                        "method": "conversation_structure",
                        "reason": reason
                    }
                    return selected_leader

        # Rule 3: Speaking Duration (Last Resort Fallback)
        selected_leader = max(tied_candidates, key=lambda s: talk_times.get(s, 0.0))
        reason = (
            f"Leader probabilities and conversation structure were tied. "
            f"Speaking duration fallback selected {selected_leader} ({talk_times.get(selected_leader, 0.0):.1f}s)."
        )
        job.metadata["leader_resolution"] = {
            "required": not is_no_leader,
            "candidate_count": candidate_count,
            "candidate_speakers": candidates,
            "selected": selected_leader,
            "method": "speaking_duration",
            "reason": reason
        }
        return selected_leader


# ---------------------------------------------------------------------------
# Stub implementation — inherited for backward compatibility
# ---------------------------------------------------------------------------

class StubLeadSpeakerIdentifier(RoleBasedLeadSpeakerIdentifier):
    """
    Subclass of RoleBasedLeadSpeakerIdentifier for compatibility.
    """
    def __init__(self, probability_tolerance: float = 0.05):
        super().__init__(probability_tolerance=probability_tolerance)


# ---------------------------------------------------------------------------
# Convenience re-export
# ---------------------------------------------------------------------------

__all__ = [
    "LeadSpeakerIdentifier",
    "RoleBasedLeadSpeakerIdentifier",
    "StubLeadSpeakerIdentifier",
]
