"""
Phase 7 — Training loop
=======================

Fine-tunes :class:`CrossAttentionFusion` + :class:`EmotionHead` on cached
encoder features.  Wav2Vec2 and BERT stay frozen — see ``precompute.py`` for
why that is the right trade on a laptop.

What actually gets trained
--------------------------
Only ``fusion.*`` and ``emotion_head.*`` (~5.5M parameters).  ``sarcasm_head``
is **not** trained: MELD carries no sarcasm annotation.  It is still written to
the checkpoint (so the file loads cleanly) but its weights stay random — see
the warning printed at the end of training.

Class imbalance
---------------
MELD train is 47% neutral and 2.7% fear.  Unweighted cross-entropy on this
converges to "always predict neutral", which scores well on accuracy and
terribly on macro F1.  Loss is therefore class-weighted by default.

Usage
-----
    python -m emotion.training.train \
        --train-split train --dev-split dev \
        --epochs 30 --out emotion/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..emotion_classifier import EMOTION_LABELS, MultimodalEmotionClassifier
from .metrics import compute_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "emotion" / "data" / "cache"
DEFAULT_CKPT = REPO_ROOT / "emotion" / "checkpoints" / "best.pt"


class CachedEmotionDataset(Dataset):
    """Reads the ``.pt`` files written by ``precompute.py``."""

    def __init__(self, cache_dir: Path, split: str):
        self.dir = Path(cache_dir) / split
        index_path = self.dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"No cache index at {index_path}.\n"
                f"   Run: python -m emotion.training.precompute "
                f"--manifest <manifest.csv> --split {split}"
            )
        self.entries = json.loads(index_path.read_text(encoding="utf-8"))
        if not self.entries:
            raise ValueError(f"Cache index {index_path} is empty.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        return torch.load(self.dir / self.entries[i]["file"], weights_only=True)

    def label_counts(self) -> list[int]:
        counts = [0] * len(EMOTION_LABELS)
        for entry in self.entries:
            counts[EMOTION_LABELS.index(entry["label"])] += 1
        return counts


def collate(batch: list[dict]) -> dict:
    """
    Pad variable-length audio to the batch maximum and build the frame mask.

    Without the mask the fusion layer would attend over padding — which is
    exactly the bug the ``key_padding_mask`` support in ``CrossAttentionFusion``
    exists to prevent.
    """
    max_frames = max(item["audio_frames"].shape[0] for item in batch)
    batch_size = len(batch)
    hidden = batch[0]["audio_frames"].shape[1]

    audio_frames = torch.zeros(batch_size, max_frames, hidden, dtype=torch.float32)
    audio_mask = torch.zeros(batch_size, max_frames, dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item["audio_frames"].shape[0]
        audio_frames[i, :n] = item["audio_frames"].float()
        audio_mask[i, :n] = 1.0

    return {
        "audio_pooled": torch.stack([b["audio_pooled"].float() for b in batch]),
        "audio_frames": audio_frames,
        "audio_mask": audio_mask,
        "text_cls": torch.stack([b["text_cls"].float() for b in batch]),
        "text_tokens": torch.stack([b["text_tokens"].float() for b in batch]),
        "text_mask": torch.stack([b["text_mask"].float() for b in batch]),
        "vader": torch.stack([b["vader"].float() for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def class_weights(counts: list[int], scheme: str) -> torch.Tensor | None:
    """Inverse-frequency loss weights.  ``sqrt`` tempers the correction."""
    if scheme == "none":
        return None
    total = sum(counts)
    k = len(counts)
    weights = []
    for c in counts:
        if c == 0:
            weights.append(0.0)          # class absent — never contributes
        elif scheme == "sqrt":
            weights.append(math.sqrt(total / (k * c)))
        else:                             # "balanced"
            weights.append(total / (k * c))
    return torch.tensor(weights, dtype=torch.float32)


def move(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def forward_batch(model: MultimodalEmotionClassifier, batch: dict) -> torch.Tensor:
    """Fusion + emotion head. Returns logits [B, 7]."""
    fused = model.fusion(
        batch["audio_pooled"], batch["audio_frames"],
        batch["text_cls"], batch["text_tokens"], batch["vader"],
        text_mask=batch["text_mask"], audio_mask=batch["audio_mask"],
    )
    return model.emotion_head(fused)


@torch.no_grad()
def evaluate_split(model: MultimodalEmotionClassifier, loader: DataLoader, device: str) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        batch = move(batch, device)
        logits = forward_batch(model, batch)
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        y_true.extend(batch["label"].cpu().tolist())
    return compute_metrics(y_true, y_pred)


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)

    train_ds = CachedEmotionDataset(cache_dir, args.train_split)
    dev_ds = CachedEmotionDataset(cache_dir, args.dev_split)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.workers, drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.workers,
    )

    counts = train_ds.label_counts()
    print(f"\n🧠 device={device}  train={len(train_ds)}  dev={len(dev_ds)}")
    print("   train distribution: " + ", ".join(
        f"{lab}={c}" for lab, c in zip(EMOTION_LABELS, counts)
    ))

    model = MultimodalEmotionClassifier(use_zero_shot=False).to(device)

    # Only the fusion and the emotion head learn. sarcasm_head has no labels in
    # MELD, so leaving it out of the optimizer keeps it at its init rather than
    # letting weight decay drift it somewhere arbitrary.
    trainable = list(model.fusion.parameters()) + list(model.emotion_head.parameters())
    for param in model.sarcasm_head.parameters():
        param.requires_grad = False
    n_trainable = sum(p.numel() for p in trainable)
    print(f"   trainable parameters: {n_trainable:,}")

    weights = class_weights(counts, args.class_weight)
    if weights is not None:
        print("   loss weights: " + ", ".join(
            f"{lab}={w:.2f}" for lab, w in zip(EMOTION_LABELS, weights.tolist())
        ))
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1 = -1.0
    best_epoch = -1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            optimizer.step()
            scheduler.step()
            running += loss.item() * batch["label"].size(0)
            seen += batch["label"].size(0)

        dev_metrics = evaluate_split(model, dev_loader, device)
        train_loss = running / max(1, seen)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_accuracy": dev_metrics["accuracy"],
        })

        marker = ""
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            # Save the FULL classifier state dict (including the untrained
            # sarcasm head) so the file loads into MultimodalEmotionClassifier
            # with every key matched.
            torch.save(model.state_dict(), out_path)
            marker = "  ← best, saved"

        print(
            f"   epoch {epoch:>3}/{args.epochs}  loss={train_loss:.4f}  "
            f"dev macro-F1={dev_metrics['macro_f1']:.4f}  "
            f"dev acc={dev_metrics['accuracy']:.4f}{marker}",
            flush=True,
        )

    print(f"\n✅ Best dev macro-F1 {best_f1:.4f} at epoch {best_epoch} → {out_path}")
    print(
        "\n⚠️  sarcasm_head was NOT trained (MELD has no sarcasm labels). Loading this\n"
        "   checkpoint switches EmotionAnalyzer to the trained path, where sarcasm comes\n"
        "   from that untrained head instead of the VADER heuristic — so `sarcasm` output\n"
        "   will be meaningless until it is trained on a sarcasm corpus (e.g. MUStARD)."
    )

    history_path = out_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"📈 Per-epoch history → {history_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m emotion.training.train",
        description="Train cross-attention fusion + emotion head on cached features.",
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--out", default=str(DEFAULT_CKPT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--class-weight", choices=["balanced", "sqrt", "none"], default="balanced",
        help="Inverse-frequency loss weighting (MELD is 47%% neutral).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    try:
        return train(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n❌ {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
