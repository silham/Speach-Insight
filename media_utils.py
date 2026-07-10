import os
import re
import shutil
import subprocess

import librosa
import soundfile as sf

# Supported formats lists
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg"}
SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS.union(SUPPORTED_VIDEO_EXTENSIONS)

SUPPORTED_MIME_TYPES = {
    # Video MIME types
    "video/mp4",
    "video/quicktime",
    "video/x-matroska",
    "video/mkv",
    "video/webm",
    "video/x-msvideo",
    "video/avi",
    "video/mpeg",
    "video/mpg",
    # Audio MIME types
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/flac",
    "audio/x-flac",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/aac",
    "audio/x-aac",
    "audio/ogg",
    "audio/x-ogg",
    "audio/opus",
    "audio/x-opus"
}


def check_ffmpeg_installed() -> None:
    """
    Verify that FFmpeg is installed and executable.
    Raises RuntimeError if FFmpeg is missing or fails to execute.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is not installed or cannot be found in the system PATH. "
            "Please install FFmpeg and verify it is added to your PATH environment variable."
        )
    try:
        # Run a simple version command to confirm it executes correctly
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError(
            f"FFmpeg is present in PATH, but executing it failed: {e}. "
            "Please check your FFmpeg installation."
        )


def validate_media_file(filename: str, content_type: str | None = None) -> None:
    """
    Validate that the file's extension and MIME type are supported by the system.
    Raises ValueError if unsupported.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported_list = sorted(list(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported formats: {', '.join(supported_list)}."
        )

    # Validate MIME type if it is provided and is not generic application/octet-stream
    if content_type and content_type != "application/octet-stream":
        if content_type not in SUPPORTED_MIME_TYPES and not content_type.startswith(("audio/", "video/")):
            raise ValueError(
                f"Unsupported MIME type '{content_type}'. "
                "Please upload a valid audio or video file."
            )


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename by removing path traversal components
    and replacing unsafe characters for cross-platform compatibility.
    """
    base_name = os.path.basename(filename)
    name, ext = os.path.splitext(base_name)
    
    # Replace non-alphanumeric, dot, underscore, and hyphen with underscore
    sanitized_name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    sanitized_ext = re.sub(r'[^a-zA-Z0-9.]', '', ext).lower()
    
    sanitized = sanitized_name + sanitized_ext
    
    # If the name becomes empty or starts with a dot, prepend a generic string
    if not sanitized_name or sanitized.startswith('.'):
        sanitized = "upload_" + sanitized
        
    return sanitized


def convert_media_to_wav(input_path: str) -> str:
    """
    Convert any supported audio or video file to a standard PCM WAV file (16kHz, mono).
    Always generates a new WAV path and returns it.
    """
    check_ffmpeg_installed()

    base, ext = os.path.splitext(input_path)
    
    # Define a dedicated output path for the converted file
    output_wav = base + "_processed.wav"

    # FFmpeg command to convert to 16kHz mono WAV (PCM s16le)
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",          # mono audio
        "-ar", "16000",      # 16kHz sample rate
        "-vn",               # remove video track
        "-y",                # overwrite output if exists
        output_wav
    ]

    try:
        # Run conversion, capture stderr for meaningful error logging
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            raise RuntimeError(
                f"FFmpeg conversion failed (exit code {result.returncode}): {error_msg}"
            )
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise e
        raise RuntimeError(f"Failed to execute FFmpeg command: {e}")

    # Check that output file was actually created and is not empty
    if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
        raise RuntimeError("FFmpeg completed but did not produce a valid WAV output file.")

    return output_wav


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