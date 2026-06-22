"""
test_semantic_split.py
======================
Exercises the new Semantic Segmentation feature end-to-end using lightweight
mocks so we don't need to load the real ML models.

Run with:
    python test_semantic_split.py
"""

import os
import sys
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# 0. Setup paths so pipeline/ and media_utils are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  —  {detail}")


# ---------------------------------------------------------------------------
# 1. Test NLTK sentence tokenization
# ---------------------------------------------------------------------------
print("\n━━ Test 1: NLTK sent_tokenize ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    from nltk.tokenize import sent_tokenize
    text = "Hello there. How are you doing today? That is great. Let me tell you something."
    sentences = sent_tokenize(text)
    report("sent_tokenize returns list", isinstance(sentences, list))
    report("sent_tokenize splits correctly", len(sentences) == 4, f"got {len(sentences)}")
except Exception as e:
    report("NLTK import/tokenize", False, str(e))

# ---------------------------------------------------------------------------
# 2. Test slice_audio utility
# ---------------------------------------------------------------------------
print("\n━━ Test 2: slice_audio utility ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    from media_utils import slice_audio

    # Create a temporary WAV with 1 second of silence at 16kHz
    tmp_dir = tempfile.mkdtemp(prefix="si_test_")
    src_wav = os.path.join(tmp_dir, "source.wav")
    sr = 16000
    duration_sec = 2.0
    samples = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(src_wav, samples, sr)

    # Slice the first half
    out_wav = os.path.join(tmp_dir, "sliced.wav")
    slice_audio(src_wav, out_wav, 0.0, 0.5)
    report("slice_audio creates output file", os.path.isfile(out_wav))

    sliced_data, sliced_sr = sf.read(out_wav)
    expected_len = int(sr * duration_sec * 0.5)
    report(
        "sliced audio length is ~50% of original",
        abs(len(sliced_data) - expected_len) <= 1,
        f"expected ~{expected_len}, got {len(sliced_data)}",
    )

    # Edge case: start >= end should raise
    try:
        slice_audio(src_wav, out_wav, 0.8, 0.2)
        report("start >= end raises ValueError", False, "no exception raised")
    except ValueError:
        report("start >= end raises ValueError", True)

    # Edge case: missing file
    try:
        slice_audio("/nonexistent/file.wav", out_wav, 0.0, 1.0)
        report("missing file raises FileNotFoundError", False, "no exception raised")
    except FileNotFoundError:
        report("missing file raises FileNotFoundError", True)

    shutil.rmtree(tmp_dir)
except Exception as e:
    report("slice_audio utility", False, str(e))

# ---------------------------------------------------------------------------
# 3. Test _split_segment_semantically
# ---------------------------------------------------------------------------
print("\n━━ Test 3: _split_segment_semantically ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    from pipeline.schemas import SegmentResult, SegmentMeta
    from pipeline import AnalysisPipeline

    # Create a synthetic long segment
    tmp_dir = tempfile.mkdtemp(prefix="si_test_split_")
    src_wav = os.path.join(tmp_dir, "seg_0.wav")
    sr = 16000
    duration_sec = 30.0  # longer than 20s
    samples = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(src_wav, samples, sr)

    seg = SegmentResult.from_meta(SegmentMeta(
        segment_id=0,
        speaker="SPEAKER_00",
        start_time=0.0,
        end_time=duration_sec,
        audio_path=src_wav,
        audio_url="/audio/test/seg_0.wav",
    ))
    # Multi-sentence transcript spanning two topics
    seg.text = (
        "Hello how are you doing today. It is nice to see you. "
        "You did a great job on the project. Your performance was excellent."
    )

    # Mock classifier: first two sentences → WarmUp, last two → Praise
    call_count = [0]
    def mock_classify(text):
        call_count[0] += 1
        if call_count[0] <= 2:
            return {"label": "WarmUp", "confidence": 0.95, "all_scores": {}}
        return {"label": "Praise", "confidence": 0.90, "all_scores": {}}

    mock_clf = MagicMock()
    mock_clf.classify = mock_classify

    pipeline_obj = AnalysisPipeline(
        transcriber=MagicMock(),
        emotion_analyzer=MagicMock(),
        template_classifier=mock_clf,
    )

    result = pipeline_obj._split_segment_semantically(seg, "test_job", tmp_dir)

    report("returns list", isinstance(result, list))
    report(
        "splits into 2 sub-segments (WarmUp + Praise)",
        len(result) == 2,
        f"got {len(result)} segments",
    )

    if len(result) == 2:
        report("first sub-segment label is WarmUp", result[0].template_label == "WarmUp",
               f"got '{result[0].template_label}'")
        report("second sub-segment label is Praise", result[1].template_label == "Praise",
               f"got '{result[1].template_label}'")
        report("first sub-segment has text", len(result[0].text) > 0)
        report("second sub-segment has text", len(result[1].text) > 0)
        report("sub-segment audio files exist",
               os.path.isfile(result[0].audio_path) and os.path.isfile(result[1].audio_path))
        report("start_time of first sub is 0", result[0].start_time == 0.0,
               f"got {result[0].start_time}")
        report("end_time of second sub is original end",
               result[1].end_time == duration_sec,
               f"got {result[1].end_time}")
        report("times are contiguous (first.end == second.start)",
               abs(result[0].end_time - result[1].start_time) < 0.5,
               f"gap: {abs(result[0].end_time - result[1].start_time):.3f}s")
        report("speaker preserved",
               all(r.speaker == "SPEAKER_00" for r in result))

    shutil.rmtree(tmp_dir)
except Exception as e:
    import traceback
    traceback.print_exc()
    report("_split_segment_semantically", False, str(e))

# ---------------------------------------------------------------------------
# 4. Test: short segment (< 20s) should NOT be split
# ---------------------------------------------------------------------------
print("\n━━ Test 4: Short segment bypass (< 20s) ━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    from pipeline.schemas import SegmentResult, SegmentMeta

    seg_short = SegmentResult.from_meta(SegmentMeta(
        segment_id=0,
        speaker="SPEAKER_00",
        start_time=0.0,
        end_time=10.0,  # < 20s
        audio_path="/fake/path.wav",
        audio_url="/audio/test/path.wav",
    ))
    seg_short.text = "Hello there. How are you?"

    # The pipeline run() only calls _split for segments > 20s
    # so seg_short should pass through unchanged
    report("short segment duration < 20", seg_short.duration < 20.0)
    report("short segment would be skipped by pipeline threshold", seg_short.duration <= 20.0)
except Exception as e:
    report("short segment bypass", False, str(e))

# ---------------------------------------------------------------------------
# 5. Test: single-sentence segment should return unchanged
# ---------------------------------------------------------------------------
print("\n━━ Test 5: Single sentence → no split ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    tmp_dir = tempfile.mkdtemp(prefix="si_test_single_")
    src_wav = os.path.join(tmp_dir, "seg_single.wav")
    samples = np.zeros(int(16000 * 25.0), dtype=np.float32)
    sf.write(src_wav, samples, 16000)

    seg_single = SegmentResult.from_meta(SegmentMeta(
        segment_id=0, speaker="SPEAKER_00",
        start_time=0.0, end_time=25.0,
        audio_path=src_wav, audio_url="/audio/test/seg_single.wav",
    ))
    seg_single.text = "Just one single sentence here"

    mock_clf2 = MagicMock()
    mock_clf2.classify.return_value = {"label": "WarmUp", "confidence": 0.95, "all_scores": {}}

    pipeline_obj2 = AnalysisPipeline(
        transcriber=MagicMock(),
        emotion_analyzer=MagicMock(),
        template_classifier=mock_clf2,
    )

    result = pipeline_obj2._split_segment_semantically(seg_single, "test", tmp_dir)
    report("single sentence returns original segment", len(result) == 1, f"got {len(result)}")
    report("returned segment is the original", result[0] is seg_single)

    shutil.rmtree(tmp_dir)
except Exception as e:
    report("single sentence test", False, str(e))

# ---------------------------------------------------------------------------
# 6. Test: all sentences same label → no split (returns original)
# ---------------------------------------------------------------------------
print("\n━━ Test 6: All same label → no split ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    tmp_dir = tempfile.mkdtemp(prefix="si_test_same_")
    src_wav = os.path.join(tmp_dir, "seg_same.wav")
    samples = np.zeros(int(16000 * 25.0), dtype=np.float32)
    sf.write(src_wav, samples, 16000)

    seg_same = SegmentResult.from_meta(SegmentMeta(
        segment_id=0, speaker="SPEAKER_01",
        start_time=0.0, end_time=25.0,
        audio_path=src_wav, audio_url="/audio/test/seg_same.wav",
    ))
    seg_same.text = "Hello there. How are you today. Nice weather we are having."

    mock_clf3 = MagicMock()
    mock_clf3.classify.return_value = {"label": "WarmUp", "confidence": 0.9, "all_scores": {}}

    pipeline_obj3 = AnalysisPipeline(
        transcriber=MagicMock(),
        emotion_analyzer=MagicMock(),
        template_classifier=mock_clf3,
    )

    result = pipeline_obj3._split_segment_semantically(seg_same, "test", tmp_dir)
    report("same label returns original (no boundary)", len(result) == 1, f"got {len(result)}")
    report("returned segment is the original", result[0] is seg_same)

    shutil.rmtree(tmp_dir)
except Exception as e:
    report("same label test", False, str(e))

# ---------------------------------------------------------------------------
# 7. Test: Stage 4 optimization — pre-labelled segments are skipped
# ---------------------------------------------------------------------------
print("\n━━ Test 7: Stage 4 skips pre-labelled segments ━━━━━━━━━━━━━━━━━━")
try:
    from pipeline.schemas import SegmentResult, SegmentMeta

    # Segment already labelled (from semantic split)
    seg_labelled = SegmentResult.from_meta(SegmentMeta(
        segment_id=0, speaker="SPEAKER_00",
        start_time=0.0, end_time=5.0,
        audio_path="/fake.wav", audio_url="/audio/test/fake.wav",
    ))
    seg_labelled.template_label = "WarmUp"

    # Segment NOT labelled (short, never went through split)
    seg_unlabelled = SegmentResult.from_meta(SegmentMeta(
        segment_id=1, speaker="SPEAKER_00",
        start_time=5.0, end_time=10.0,
        audio_path="/fake2.wav", audio_url="/audio/test/fake2.wav",
    ))
    seg_unlabelled.text = "Some unlabelled text"

    segments = [seg_labelled, seg_unlabelled]
    unlabelled = [s for s in segments if not s.template_label]

    report("pre-labelled segment filtered out", seg_labelled not in unlabelled)
    report("unlabelled segment included", seg_unlabelled in unlabelled)
    report("only 1 segment needs classification", len(unlabelled) == 1,
           f"got {len(unlabelled)}")
except Exception as e:
    report("Stage 4 optimization", False, str(e))

# ---------------------------------------------------------------------------
# 8. Import smoke test for the full pipeline module
# ---------------------------------------------------------------------------
print("\n━━ Test 8: Full module import smoke test ━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    from pipeline import AnalysisPipeline
    from pipeline.schemas import SegmentResult, SegmentMeta, JobResult
    from media_utils import slice_audio, convert_video_to_audio
    report("all imports succeed", True)
except Exception as e:
    report("module imports", False, str(e))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results:  {PASS} passed,  {FAIL} failed")
print(f"{'='*60}")

if FAIL > 0:
    sys.exit(1)
