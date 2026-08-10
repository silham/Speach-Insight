"""
Phase 7 — Feature precomputation
================================

Runs the frozen Wav2Vec2 / BERT encoders over a manifest **once** and caches
their outputs to disk.  Training then reads tensors instead of audio, which
turns each epoch from roughly an hour into roughly a minute — the difference
between "fine-tuning is a weekend job" and "fine-tuning is a coffee break".

This is what makes the frozen-encoder plan viable on a laptop.  The trade is
that the encoders cannot adapt (their weights never see a gradient); only the
cross-attention fusion and the classifier heads are trained.

Two deliberate choices
----------------------
*Audio is encoded one clip at a time*, not in padded batches.  ``wav2vec2-base``
uses group normalisation in its feature extractor and was pre-trained without
an attention mask, so HuggingFace advises against passing one — batching with
padding would therefore shift the embeddings away from what single-clip
inference produces.  Matching inference exactly is worth more than the speedup.

*Paralinguistic features are skipped.*  ``CrossAttentionFusion`` never consumes
them (they are reported to the API but not fed to the model), and
``librosa.yin`` is the single slowest call in the pipeline.

Usage
-----
    python -m emotion.training.precompute \
        --manifest emotion/data/meld_train.csv --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .dataset import LABEL_TO_INDEX, format_stats, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "emotion" / "data" / "cache"

# Wav2Vec2 emits ~50 frames/sec.  Striding by 2 halves storage and I/O at a
# resolution the fusion layer cannot exploit anyway — text→audio attention has
# a single query vector, so it is pooling, not reading fine temporal detail.
FRAME_STRIDE = 2
MAX_FRAMES = 500          # 500 strided frames == MAX_AUDIO_SECONDS of audio

# Wav2Vec2 self-attention is O(T²) in frames, so a long clip does not merely
# run slowly — it allocates quadratically. MELD's two outliers (305 s and
# 235 s) demand a ~10 GB buffer and hard-fail on MPS.  Truncating the waveform
# before encoding both bounds memory and costs nothing: anything past
# MAX_FRAMES*FRAME_STRIDE would be discarded from the cache regardless.
# 20 s covers 99.93% of MELD clips in full.
MAX_AUDIO_SECONDS = 20
SAMPLE_RATE = 16000

MAX_TEXT_TOKENS = 128     # must match LinguisticEncoder.MAX_LENGTH


def precompute_split(
    manifest: str | Path,
    split: str,
    cache_dir: Path,
    device: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Encode every sample in ``manifest`` and write one ``.pt`` per clip."""
    from ..acoustic_encoder import AcousticEncoder
    from ..linguistic_encoder import LinguisticEncoder
    from ..vader_analyzer import VaderAnalyzer

    samples, stats = load_manifest(manifest)
    print(f"\n📂 {manifest}")
    print(format_stats(stats))
    if limit:
        samples = samples[:limit]
        print(f"\n⚠️  --limit {limit}: caching a subset only.")
    if not samples:
        raise ValueError(f"No usable samples in {manifest}")

    out_dir = cache_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    acoustic = AcousticEncoder(device=device)
    linguistic = LinguisticEncoder(device=device)
    vader = VaderAnalyzer()

    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc=f"Encoding {split}", unit="clip")
    except ImportError:
        iterator = samples

    index: list[dict] = []
    failures: list[dict] = []
    truncated: list[str] = []
    skipped = 0

    for sample in iterator:
        target = out_dir / f"{sample.id}.pt"
        if target.is_file() and not overwrite:
            skipped += 1
            index.append({"id": sample.id, "file": target.name, "label": sample.label})
            continue

        try:
            # Bypass AcousticEncoder.encode() so the waveform can be truncated
            # before the O(T²) encoder sees it, and so librosa's paralinguistic
            # features (unused by the fusion layer) are skipped entirely.
            waveform = acoustic.load_audio(sample.audio_path)
            max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
            if waveform.shape[-1] > max_samples:
                waveform = waveform[..., :max_samples]
                truncated.append(sample.id)

            pooled, frame_features = acoustic.forward(waveform)
            linguistic_out = linguistic.encode(sample.text)
            vader_out = vader.analyze(sample.text)

            frames = frame_features.squeeze(0)                      # [T, 768]
            frames = frames[::FRAME_STRIDE][:MAX_FRAMES]            # [T', 768]
            if frames.shape[0] == 0:
                raise ValueError("clip produced zero encoder frames (empty audio?)")

            torch.save(
                {
                    "id": sample.id,
                    # fp16 halves cache size; these are frozen encoder outputs
                    # fed into a LayerNorm, so the precision loss is immaterial.
                    "audio_pooled": pooled.squeeze(0).half(),
                    "audio_frames": frames.half(),
                    "text_cls": linguistic_out["cls_embedding"].squeeze(0).half(),
                    "text_tokens": linguistic_out["token_features"].squeeze(0).half(),
                    "text_mask": linguistic_out["attention_mask"].squeeze(0).to(torch.uint8),
                    "vader": vader_out["tensor"].float(),
                    "label": LABEL_TO_INDEX[sample.label],
                },
                target,
            )
            index.append({"id": sample.id, "file": target.name, "label": sample.label})
        except Exception as exc:
            failures.append({"id": sample.id, "error": f"{type(exc).__name__}: {exc}"})

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.pt"))
    print(f"\n✅ {len(index)} cached to {out_dir}  ({total_bytes / 1e9:.2f} GB)")
    if skipped:
        print(f"   ({skipped} already present — rerun with --overwrite to redo)")
    if truncated:
        print(f"   {len(truncated)} clip(s) truncated to {MAX_AUDIO_SECONDS}s "
              f"(e.g. {', '.join(truncated[:3])})")
    if failures:
        print(f"⚠️  {len(failures)} clip(s) failed to encode:")
        for f in failures[:5]:
            print(f"     {f['id']}: {f['error']}")
    return {"cached": len(index), "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m emotion.training.precompute",
        description="Cache frozen-encoder features for fast training.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True, help="Cache subdirectory name, e.g. train.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--device", help="Torch device override (e.g. cpu, mps).")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        precompute_split(
            manifest=args.manifest,
            split=args.split,
            cache_dir=Path(args.cache_dir),
            device=args.device,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n❌ {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
