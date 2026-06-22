import os
import subprocess

import librosa
import soundfile as sf

def convert_video_to_audio(input_path):
    """
    Convert video files to WAV audio (16kHz mono).
    If the file is already audio, return it unchanged.
    """

    base, ext = os.path.splitext(input_path)

    video_extensions = [".mp4", ".mov", ".mkv", ".avi", ".webm"]

    if ext.lower() not in video_extensions:
        return input_path  # already audio

    output_audio = base + ".wav"

    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",        # mono audio
        "-ar", "16000",    # 16kHz sample rate
        "-vn",             # remove video
        "-y",
        output_audio
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_audio


def slice_audio(input_path: str, output_path: str, start_ratio: float, end_ratio: float) -> None:
    """
    Slice a WAV file based on a percentage of its total duration.

    Parameters
    ----------
    input_path : str
        Path to the source WAV file.
    output_path : str
        Path where the sliced audio will be saved.
    start_ratio : float
        Start point as a fraction of total duration (0.0–1.0).
    end_ratio : float
        End point as a fraction of total duration (0.0–1.0).

    Raises
    ------
    ValueError
        If the resulting slice would be empty (start >= end after clamping).
    FileNotFoundError
        If `input_path` does not exist.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    # Clamp ratios to safe range
    start_ratio = max(0.0, min(1.0, start_ratio))
    end_ratio = max(0.0, min(1.0, end_ratio))

    if start_ratio >= end_ratio:
        raise ValueError(
            f"Invalid slice range: start_ratio ({start_ratio:.4f}) >= end_ratio ({end_ratio:.4f})"
        )

    # Load audio at 16 kHz to match the rest of the pipeline
    y, sr = librosa.load(input_path, sr=16000)
    total_samples = len(y)

    start_sample = int(total_samples * start_ratio)
    end_sample = int(total_samples * end_ratio)

    y_slice = y[start_sample:end_sample]

    if len(y_slice) == 0:
        raise ValueError(
            f"Empty audio slice for {input_path} "
            f"(start_sample={start_sample}, end_sample={end_sample}, total={total_samples})"
        )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sf.write(output_path, y_slice, sr)