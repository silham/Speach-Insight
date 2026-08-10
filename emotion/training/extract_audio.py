"""
Phase 7 — Audio extraction
==========================

MELD ships as MP4 video; every encoder in this package wants 16 kHz mono WAV.
This converts a directory of clips in parallel and reports what failed.

MELD is known to contain a handful of unreadable/zero-length MP4s — those are
counted and skipped rather than aborting the run.

Usage
-----
    python -m emotion.training.extract_audio \
        --video-dir emotion/data/MELD.Raw/output_repeated_splits_test \
        --out-dir   emotion/data/MELD.Raw/wav/test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SAMPLE_RATE = 16000


def convert_one(video: Path, out_dir: Path, overwrite: bool = False) -> tuple[Path, str | None]:
    """Convert one clip. Returns ``(video, error_or_None)``."""
    out = out_dir / f"{video.stem}.wav"
    if out.is_file() and out.stat().st_size > 44 and not overwrite:
        return video, None  # already done (44 bytes = empty WAV header)

    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vn",                       # drop the video stream
        "-ar", str(SAMPLE_RATE),     # 16 kHz, matching Wav2Vec2
        "-ac", "1",                  # mono
        str(out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return video, "ffmpeg timed out"
    except FileNotFoundError:
        return video, "ffmpeg not found on PATH"

    if proc.returncode != 0:
        return video, (proc.stderr or "ffmpeg failed").strip().splitlines()[-1][:120]
    if not out.is_file() or out.stat().st_size <= 44:
        out.unlink(missing_ok=True)
        return video, "produced an empty WAV (no audio stream?)"
    return video, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m emotion.training.extract_audio",
        description="Convert a directory of MP4 clips to 16 kHz mono WAV.",
    )
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    if not video_dir.is_dir():
        print(f"❌ Not a directory: {video_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in video_dir.rglob("*.mp4")
        if not p.name.startswith("._")  # macOS AppleDouble sidecars
    )
    if not videos:
        print(f"❌ No .mp4 files under {video_dir}")
        return 1

    print(f"🎬 {len(videos)} clips → {out_dir}  ({args.workers} workers)")

    failures: list[tuple[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one, v, out_dir, args.overwrite) for v in videos]
        for future in as_completed(futures):
            video, error = future.result()
            done += 1
            if error:
                failures.append((video.name, error))
            if done % 250 == 0 or done == len(videos):
                print(f"   {done}/{len(videos)}  ({len(failures)} failed)", flush=True)

    ok = len(videos) - len(failures)
    print(f"\n✅ {ok} WAVs written to {out_dir}")
    if failures:
        print(f"⚠️  {len(failures)} clip(s) failed:")
        for name, err in failures[:10]:
            print(f"     {name}: {err}")
        if len(failures) > 10:
            print(f"     … and {len(failures) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
