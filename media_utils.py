import os
import subprocess

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