import os
import shutil
import torch
import soundfile as sf
import warnings

# Suppress Pyannote/Torchcodec and PyTorch standard deviation warnings
warnings.filterwarnings("ignore", message="(?s).*torchcodec.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


def segment_and_save(input_file, output_folder="segmented_clips"):
    """
    Uses Pyannote diarization to split speakers and save audio segments.

    Returns:
        List of dicts with segment metadata.
    """

    # ── Prepare output folder ─────────────────────────────
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    print("⏳ Loading Pyannote Diarization Pipeline...")

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        print(f"✅ Pipeline loaded on {device}")

    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return []

    # ── Load audio safely (NO torchaudio) ────────────────
    print("🎵 Loading audio...")

    try:
        audio, sample_rate = sf.read(input_file)

        # convert stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

    except Exception as e:
        print(f"❌ Audio loading failed: {e}")
        return []

    # ── Convert to torch waveform for pyannote ───────────
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    # ── Run diarization (IMPORTANT FIX) ──────────────────
    print("🕵️ Analyzing speakers...")

    try:
        diarization = pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate
        })
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        return []

    # ── Unwrap DiarizeOutput (pyannote ≥ 3.3) ───────────
    # Newer pyannote versions return a DiarizeOutput wrapper;
    # the actual Annotation lives in .speaker_diarization.
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    # ── Segment extraction ───────────────────────────────
    segments = []
    print("✂️ Extracting speaker segments...")

    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

        start_sample = int(turn.start * sample_rate)
        end_sample = int(turn.end * sample_rate)

        segment_audio = audio[start_sample:end_sample]

        filename = f"seg_{i:03d}_{speaker}.wav"
        save_path = os.path.join(output_folder, filename)

        sf.write(save_path, segment_audio, sample_rate)

        segments.append({
            "segment_id": i,
            "path": save_path,
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    print("✅ Segmentation complete.")
    return segments