import os
<<<<<<< HEAD
import shutil
import torch
import soundfile as sf
import warnings

# Suppress Pyannote/Torchcodec and PyTorch standard deviation warnings
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

=======

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import shutil
import torch
import torchaudio
from pyannote.audio import Pipeline

>>>>>>> 82a73dc52e8f69a6ab9806ffa9137263868c8bf2
HF_TOKEN = os.environ.get("HF_TOKEN")


def segment_and_save(input_file, output_folder="segmented_clips"):
    """
<<<<<<< HEAD
    Uses Pyannote diarization to split speakers and save audio segments.

    Returns:
        List of dicts with segment metadata.
    """

    # ── Prepare output folder ─────────────────────────────
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    print("⏳ Loading Pyannote Diarization Pipeline...")

=======
    Uses Pyannote to detect WHO is speaking and save their parts.

    Returns
    -------
    list[dict]
        Each entry describes one speaker turn::

            {
                "segment_id": 0,
                "path":       "/abs/path/to/seg_000_SPEAKER_00.wav",
                "speaker":    "SPEAKER_00",
                "start":      1.23,   # seconds from start of original file
                "end":        4.56,
            }

        An empty list is returned if diarization fails or finds nothing.

    Note
    ----
    The ``path`` key supersedes the bare ``str`` return that used to be the
    contract.  All callers (``api.py``, ``app.py``, the pipeline) use
    ``entry["path"]`` to access the audio file.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    print("⏳ Loading Pyannote Diarization Pipeline...")
>>>>>>> 82a73dc52e8f69a6ab9806ffa9137263868c8bf2
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
<<<<<<< HEAD

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        print(f"✅ Pipeline loaded on {device}")

=======
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        pipeline.to(device)
        print(f"✅ Pipeline loaded on {device}")
>>>>>>> 82a73dc52e8f69a6ab9806ffa9137263868c8bf2
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return []

<<<<<<< HEAD
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
=======
    # --- Format Standardization ---
    print("🎵 Standardizing audio format to pure WAV...")
    waveform, sample_rate = torchaudio.load(input_file)

    # Convert stereo to mono for Pyannote
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    temp_wav_path = os.path.join(output_folder, "temp_clean.wav")
    torchaudio.save(temp_wav_path, waveform, sample_rate)

    print("🕵️ Analyzing speakers...")
    try:
        # Pass the clean WAV file directly to get the standard Annotation object
        diarization = pipeline(temp_wav_path)
>>>>>>> 82a73dc52e8f69a6ab9806ffa9137263868c8bf2
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        return []

<<<<<<< HEAD
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

=======
    segments = []
    print("✂️ Diarization complete! Slicing turns...")

    for i, (turn, _, speaker) in enumerate(
        diarization.speaker_diarization.itertracks(yield_label=True)
    ):
        start = turn.start
        end = turn.end

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        segment_waveform = waveform[:, start_sample:end_sample]
        filename = f"seg_{i:03d}_{speaker}.wav"
        save_path = os.path.join(output_folder, filename)

        torchaudio.save(save_path, segment_waveform, sample_rate)
>>>>>>> 82a73dc52e8f69a6ab9806ffa9137263868c8bf2
        segments.append({
            "segment_id": i,
            "path": save_path,
            "speaker": speaker,
<<<<<<< HEAD
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    print("✅ Segmentation complete.")
=======
            "start": round(start, 3),
            "end": round(end, 3),
        })

    # Clean up the temporary WAV file to save space
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

>>>>>>> 82a73dc52e8f69a6ab9806ffa9137263868c8bf2
    return segments