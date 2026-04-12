"""
pipeline/__init__.py  —  SpeechInSight Analysis Pipeline
=========================================================
Central orchestrator that connects all models in a single, linear chain.

Pipeline stages (one per section below)
-----------------------------------------
1. Diarization     segmentation.segment_and_save()
2. Transcription   model.Transcriber.transcribe()
3. Emotion         emotion.EmotionAnalyzer.analyze()
4. Lead Speaker    pipeline.lead_speaker.LeadSpeakerIdentifier.identify()
<< add future stages here — see "Adding a New Model" in Dataflow.md >>

Usage
-----
Construct once at startup (heavy models are loaded in __init__):

    from pipeline import AnalysisPipeline
    from pipeline.lead_speaker import StubLeadSpeakerIdentifier

    pipeline = AnalysisPipeline(
        transcriber=load_transcriber(),
        emotion_analyzer=EmotionAnalyzer(),
        lead_speaker=StubLeadSpeakerIdentifier(),  # swap for trained model later
    )

Then call per request:

    job: JobResult = pipeline.run(
        audio_path="uploads/abc123.wav",
        job_id="abc123",
        processed_dir="processed",
    )
    return job.to_dict()
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from .schemas import JobResult, SegmentMeta, SegmentResult

if TYPE_CHECKING:
    # Avoid circular imports — only used for type hints
    from emotion import EmotionAnalyzer  # noqa: F401
    from model import Transcriber  # noqa: F401
    from pipeline.lead_speaker import LeadSpeakerIdentifier  # noqa: F401


class AnalysisPipeline:
    """
    Chains diarization → transcription → emotion → lead-speaker identification
    and returns a fully-populated :class:`JobResult`.

    Parameters
    ----------
    transcriber :
        An initialised ``Transcriber`` instance (``model.Transcriber``).
    emotion_analyzer :
        An initialised ``EmotionAnalyzer`` instance (``emotion.EmotionAnalyzer``).
    lead_speaker : optional
        Any object implementing ``LeadSpeakerIdentifier.identify(job)``.
        Pass ``None`` to skip the lead-speaker stage.
    audio_base_url : str
        URL prefix used to build ``audio_url`` for each segment.
        Default is ``"/audio"`` which matches the FastAPI static mount.
    """

    def __init__(
        self,
        transcriber,
        emotion_analyzer,
        lead_speaker: Optional["LeadSpeakerIdentifier"] = None,
        audio_base_url: str = "/audio",
    ):
        self.transcriber = transcriber
        self.emotion_analyzer = emotion_analyzer
        self.lead_speaker = lead_speaker
        self.audio_base_url = audio_base_url

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        audio_path: str,
        job_id: str,
        processed_dir: str,
    ) -> JobResult:
        """
        Execute the full pipeline on one audio file.

        Parameters
        ----------
        audio_path :
            Path to the raw (already video-converted) WAV file.
        job_id :
            Unique identifier for this analysis job (used in URLs and dirs).
        processed_dir :
            Root folder where segmented clips are saved.
            Each job gets its own subfolder: ``{processed_dir}/{job_id}/``.

        Returns
        -------
        JobResult
            Fully populated result including lead_speaker, aggregate stats,
            and the list of SegmentResult objects.
        """
        from segmentation import segment_and_save  # lazy import — avoids torch startup on import

        job = JobResult(job_id=job_id)

        # ── Stage 1: Diarization ────────────────────────────────────────
        job_output_folder = os.path.join(processed_dir, job_id)
        raw_segments = segment_and_save(audio_path, job_output_folder)
        # raw_segments is list[dict]: {segment_id, path, speaker, start, end}

        if not raw_segments:
            return job.finalise()

        # Build SegmentResult list from diarization output
        for raw in raw_segments:
            audio_url = f"{self.audio_base_url}/{job_id}/{os.path.basename(raw['path'])}"
            meta = SegmentMeta(
                segment_id=raw["segment_id"],
                speaker=raw["speaker"],
                start_time=raw["start"],
                end_time=raw["end"],
                audio_path=raw["path"],
                audio_url=audio_url,
            )
            job.segments.append(SegmentResult.from_meta(meta))

        # ── Stage 2: Transcription ──────────────────────────────────────
        print(f"🧠 Transcribing {len(job.segments)} segments…")
        for seg in job.segments:
            seg.text = self.transcriber.transcribe(seg.audio_path)

        # ── Stage 3: Emotion Recognition ────────────────────────────────
        print(f"🎭 Running emotion analysis on {len(job.segments)} segments…")
        for seg in job.segments:
            try:
                result = self.emotion_analyzer.analyze(seg.audio_path, seg.text)
                seg.emotion = result.get("emotion", "neutral")
                seg.confidence = result.get("confidence", 0.0)
                seg.all_emotions = result.get("all_emotions", {})
                seg.sarcasm = result.get("sarcasm", False)
                seg.sarcasm_score = result.get("sarcasm_score", 0.0)
                seg.ambiguity_score = result.get("ambiguity_score", 0.0)
                seg.vader = result.get("vader", {})
                seg.paralinguistic = result.get("paralinguistic", {})
            except Exception as exc:
                print(f"⚠️  Emotion analysis failed for {seg.audio_path}: {exc}")
                # Defaults already set by SegmentResult dataclass

        # ── Stage 4: Lead Speaker Identification ────────────────────────
        if self.lead_speaker is not None:
            print("👤 Identifying lead speaker…")
            try:
                job.lead_speaker = self.lead_speaker.identify(job)
                print(f"✅ Lead speaker: {job.lead_speaker}")
            except Exception as exc:
                print(f"⚠️  Lead-speaker identification failed: {exc}")

        # ── Finalise aggregate stats ─────────────────────────────────────
        job.finalise()

        print(
            f"✅ Pipeline complete — {job.total_segments} segments, "
            f"{job.total_speakers} speaker(s), "
            f"{job.total_duration:.1f}s total audio"
        )
        return job
