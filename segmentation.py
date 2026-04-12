import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import shutil
import torch
import torchaudio
from pyannote.audio import Pipeline

HF_TOKEN = os.environ.get("HF_TOKEN")


def segment_and_save(input_file, output_folder="segmented_clips"):
    """
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
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        pipeline.to(device)
        print(f"✅ Pipeline loaded on {device}")
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return []

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
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        return []

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
        segments.append({
            "segment_id": i,
            "path": save_path,
            "speaker": speaker,
            "start": round(start, 3),
            "end": round(end, 3),
        })

    # Clean up the temporary WAV file to save space
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

    return segments