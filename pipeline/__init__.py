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

import json
import os
from typing import TYPE_CHECKING, Optional

from nltk.tokenize import sent_tokenize

from .schemas import JobResult, SegmentMeta, SegmentResult
from .scoring import generate_score_and_evidence

if TYPE_CHECKING:
    # Avoid circular imports — only used for type hints
    from emotion import EmotionAnalyzer  # noqa: F401
    from model import Transcriber  # noqa: F401
    from pipeline.lead_speaker import LeadSpeakerIdentifier  # noqa: F401


class AnalysisPipeline:
    """
    Chains diarization → transcription → emotion → template classification
    → lead-speaker identification and returns a fully-populated :class:`JobResult`.

    Parameters
    ----------
    transcriber :
        An initialised ``Transcriber`` instance (``model.Transcriber``).
    emotion_analyzer :
        An initialised ``EmotionAnalyzer`` instance (``emotion.EmotionAnalyzer``).
    template_classifier : optional
        An initialised ``TemplateClassifier`` (``template_classifier.TemplateClassifier``).
        Pass ``None`` to skip template classification.
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
        template_classifier=None,
        lead_speaker: Optional["LeadSpeakerIdentifier"] = None,
        audio_base_url: str = "/audio",
    ):
        self.transcriber = transcriber
        self.emotion_analyzer = emotion_analyzer
        self.template_classifier = template_classifier
        self.lead_speaker = lead_speaker
        self.audio_base_url = audio_base_url

    # ------------------------------------------------------------------
    # Semantic splitting helper
    # ------------------------------------------------------------------

    def _split_segment_semantically(
        self,
        seg: SegmentResult,
        job_id: str,
        job_output_folder: str,
    ) -> list[SegmentResult]:
        """
        Split a single long SegmentResult into multiple SegmentResults
        based on template-classification boundary detection.

        Uses the "Approximation Hack": character-length ratios estimate
        where in the audio each topic group starts and ends.
        """
        from media_utils import slice_audio  # lazy to avoid circular / heavy imports

        sentences = sent_tokenize(seg.text)
        if len(sentences) <= 1:
            return [seg]  # Nothing to split

        sub_segments: list[SegmentResult] = []
        current_label: str | None = None
        current_sentences: list[str] = []
        total_chars = len(seg.text)
        current_start_char = 0

        if total_chars == 0:
            return [seg]

        for sentence in sentences:
            try:
                result = self.template_classifier.classify(sentence)
                label = result.get("label", "neutral")
            except Exception as exc:
                print(f"⚠️  Sentence-level classification failed, skipping split: {exc}")
                return [seg]

            # Boundary detected — flush the current group
            if label != current_label and current_label is not None:
                group_text = " ".join(current_sentences)
                group_char_len = len(group_text)

                start_ratio = current_start_char / total_chars
                end_ratio = (current_start_char + group_char_len) / total_chars

                stem = os.path.splitext(os.path.basename(seg.audio_path))[0]
                new_audio_name = f"{stem}_sub_{len(sub_segments)}.wav"
                new_audio_path = os.path.join(job_output_folder, new_audio_name)

                try:
                    slice_audio(seg.audio_path, new_audio_path, start_ratio, end_ratio)
                except Exception as exc:
                    print(
                        f"⚠️  Audio slicing failed for {seg.audio_path} "
                        f"[{start_ratio:.3f}–{end_ratio:.3f}]: {exc}"
                    )
                    return [seg]  # Fall back to the unsplit segment

                audio_url = f"{self.audio_base_url}/{job_id}/{new_audio_name}"
                seg_duration = seg.end_time - seg.start_time
                new_seg = SegmentResult.from_meta(SegmentMeta(
                    segment_id=seg.segment_id,  # re-indexed later
                    speaker=seg.speaker,
                    start_time=seg.start_time + (seg_duration * start_ratio),
                    end_time=seg.start_time + (seg_duration * end_ratio),
                    audio_path=new_audio_path,
                    audio_url=audio_url,
                ))
                new_seg.text = group_text
                new_seg.template_label = current_label
                sub_segments.append(new_seg)

                # Advance the character pointer (+1 for the joining space)
                current_start_char += group_char_len + 1
                current_sentences = []

            current_sentences.append(sentence)
            current_label = label

        # --- Flush the final group ---
        if current_sentences:
            group_text = " ".join(current_sentences)
            start_ratio = current_start_char / total_chars
            end_ratio = 1.0

            stem = os.path.splitext(os.path.basename(seg.audio_path))[0]
            new_audio_name = f"{stem}_sub_{len(sub_segments)}.wav"
            new_audio_path = os.path.join(job_output_folder, new_audio_name)

            try:
                slice_audio(seg.audio_path, new_audio_path, start_ratio, end_ratio)
            except Exception as exc:
                print(
                    f"⚠️  Audio slicing failed for final group of {seg.audio_path}: {exc}"
                )
                return [seg]

            audio_url = f"{self.audio_base_url}/{job_id}/{new_audio_name}"
            seg_duration = seg.end_time - seg.start_time
            new_seg = SegmentResult.from_meta(SegmentMeta(
                segment_id=seg.segment_id,
                speaker=seg.speaker,
                start_time=seg.start_time + (seg_duration * start_ratio),
                end_time=seg.end_time,
                audio_path=new_audio_path,
                audio_url=audio_url,
            ))
            new_seg.text = group_text
            new_seg.template_label = current_label
            sub_segments.append(new_seg)

        if len(sub_segments) <= 1:
            return [seg]  # No actual boundary found — keep the original

        print(
            f"   ✂️  Segment {seg.segment_id} ({seg.duration:.1f}s) "
            f"→ {len(sub_segments)} sub-segments"
        )
        return sub_segments

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

        # ── Stage 2.5: Semantic Splitting ─────────────────────────────────
        if self.template_classifier is not None:
            print("✂️  Splitting long segments semantically…")
            new_segments_list: list[SegmentResult] = []
            for seg in job.segments:
                if seg.duration > 20.0:
                    split_segs = self._split_segment_semantically(
                        seg, job_id, job_output_folder
                    )
                    new_segments_list.extend(split_segs)
                else:
                    new_segments_list.append(seg)

            job.segments = new_segments_list

            # Re-index segment IDs so they remain sequential
            for i, s in enumerate(job.segments):
                s.segment_id = i

            print(f"   📊 Segments after splitting: {len(job.segments)}")

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

        # ── Stage 4: Template Classification ─────────────────────────────
        # Segments that were semantically split already carry a template_label
        # from the splitting stage.  We only classify unlabelled segments here.
        if self.template_classifier is not None:
            unlabelled = [s for s in job.segments if not s.template_label]
            if unlabelled:
                print(f"📋 Classifying {len(unlabelled)} unlabelled segments with template model…")
                for seg in unlabelled:
                    try:
                        result = self.template_classifier.classify(seg.text)
                        seg.template_label = result.get("label", "")
                        seg.template_confidence = result.get("confidence", 0.0)
                    except Exception as exc:
                        print(f"⚠️  Template classification failed for segment {seg.segment_id}: {exc}")
            else:
                print("📋 All segments already labelled by semantic splitting — skipping Stage 4.")
            print("✅ Template classification complete.")

        # ── Stage 5: Lead Speaker Identification ────────────────────────
        if self.lead_speaker is not None:
            print("👤 Identifying lead speaker…")
            try:
                job.lead_speaker = self.lead_speaker.identify(job)
                print(f"✅ Lead speaker: {job.lead_speaker}")
            except Exception as exc:
                print(f"⚠️  Lead-speaker identification failed: {exc}")

        # ── Finalise aggregate stats ─────────────────────────────────────
        job.finalise()

        # ── Save transcript JSON ─────────────────────────────────────────
        try:
            transcript_data = {
                "segments": [
                    {
                        "speaker": seg.speaker,
                        "transcript": seg.text,
                        "emotion": f"{seg.emotion} {round(seg.confidence * 100)}%",
                        "template_label": seg.template_label,
                        "template_confidence": f"{round(seg.template_confidence * 100)}%",
                    }
                    for seg in job.segments
                ]
            }
            transcript_path = os.path.join(job_output_folder, "transcript.json")
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=4, ensure_ascii=False)
            print(f"💾 Transcript saved to {transcript_path}")
        except Exception as exc:
            print(f"⚠️  Failed to save transcript JSON: {exc}")

        # ── Generate Score and Evidence ──────────────────────────────────
        try:
            generate_score_and_evidence(job_output_folder)
        except Exception as exc:
            print(f"⚠️  Failed to generate score and evidence: {exc}")

        print(
            f"✅ Pipeline complete — {job.total_segments} segments, "
            f"{job.total_speakers} speaker(s), "
            f"{job.total_duration:.1f}s total audio"
        )
        return job
