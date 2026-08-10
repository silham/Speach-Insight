"""
Phase 7 — Evaluation harness
============================

Measures the emotion system's accuracy on a labelled manifest, and diffs two
such measurements to answer "did fine-tuning actually help?".

This deliberately drives the real :class:`emotion.EmotionAnalyzer` façade
rather than re-implementing inference, so what it scores is exactly what
``api.py`` and ``pipeline/`` will serve.  Swapping ``--checkpoint`` in and out
is the *only* difference between the before and after runs.

Usage
-----
    # Validate a labelling sheet without loading any models
    python -m emotion.training.evaluate --manifest DATA.csv --dry-run

    # "Before" — current zero-shot ensemble
    python -m emotion.training.evaluate --manifest DATA.csv \
        --name "zero-shot baseline" --out emotion/eval_results/before.json

    # "After" — trained cross-attention fusion
    python -m emotion.training.evaluate --manifest DATA.csv \
        --checkpoint emotion/checkpoints/best.pt \
        --name "fused (trained)" --out emotion/eval_results/after.json

    # Paired comparison
    python -m emotion.training.evaluate --compare \
        emotion/eval_results/before.json emotion/eval_results/after.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..emotion_classifier import EMOTION_LABELS
from .dataset import LABEL_TO_INDEX, format_stats, load_manifest
from .metrics import compute_metrics, mcnemar, render_comparison, render_report

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "emotion" / "eval_results"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def verify_checkpoint(checkpoint_path: str) -> None:
    """
    Fail loudly on a checkpoint that would load as garbage.

    ``MultimodalEmotionClassifier`` is forgiving in two dangerous ways: a
    *missing* file leaves the randomly-initialised fusion weights in place with
    ``use_zero_shot=False``, and a present-but-mismatched file loads nothing
    because ``load_state_dict`` is called with ``strict=False``.  Either way
    inference proceeds and returns confident nonsense, which would land in the
    "after" column of a before/after table as a real result.
    """
    import torch

    from ..emotion_classifier import MultimodalEmotionClassifier

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "   Note: EmotionAnalyzer does NOT fall back to zero-shot here — it would "
            "score randomly-initialised fusion weights. Refusing to run."
        )

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} is not a state dict (got {type(state).__name__}).")

    # A training loop that saved {"model_state_dict": ..., "epoch": ...} would
    # load as zero matching keys.
    if "model_state_dict" in state or "state_dict" in state:
        raise ValueError(
            f"Checkpoint {path} looks wrapped (contains 'model_state_dict'/'state_dict'). "
            "EmotionAnalyzer expects a bare state dict — unwrap it before evaluating."
        )

    expected = set(MultimodalEmotionClassifier(use_zero_shot=True).state_dict())
    got = set(state)
    overlap = expected & got

    if not overlap:
        raise ValueError(
            f"Checkpoint {path} shares no parameter names with the classifier.\n"
            f"   expected e.g. {sorted(expected)[:3]}\n"
            f"   found e.g.    {sorted(got)[:3]}\n"
            "   strict=False would silently load nothing. Refusing to run."
        )
    if missing := expected - got:
        print(
            f"⚠️  Checkpoint is missing {len(missing)} of {len(expected)} parameters "
            f"(e.g. {sorted(missing)[:3]}); those stay randomly initialised."
        )
    print(f"✅ Checkpoint verified: {len(overlap)}/{len(expected)} parameters matched.")


def run_predictions(
    samples: list,
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run every sample through :class:`EmotionAnalyzer`.

    Returns ``(records, errors)``.  A record carries the gold label, the
    predicted label, the full 7-way distribution and the auxiliary sarcasm /
    ambiguity outputs, so the results file is self-contained enough to
    re-score later without touching audio again.
    """
    # Imported here, not at module scope, so --compare and --dry-run stay fast.
    from .. import EmotionAnalyzer

    if checkpoint_path:
        verify_checkpoint(checkpoint_path)

    analyzer = EmotionAnalyzer(checkpoint_path=checkpoint_path, device=device)

    # Belt-and-braces: a verified checkpoint must leave the analyzer in trained
    # mode, and no checkpoint must leave it in zero-shot mode.
    if checkpoint_path and analyzer.classifier.use_zero_shot:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path!r} verified but the classifier is still "
            "in zero-shot mode. Refusing to report this as a trained run."
        )
    if not checkpoint_path and not analyzer.classifier.use_zero_shot:
        raise RuntimeError(
            "No checkpoint given but the classifier is not in zero-shot mode — "
            "the baseline would be scored on random weights."
        )

    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc="Evaluating", unit="clip")
    except ImportError:
        iterator = samples

    records: list[dict] = []
    errors: list[dict] = []
    started = time.time()

    for sample in iterator:
        try:
            out = analyzer.analyze(sample.audio_path, sample.text)
        except Exception as exc:  # one bad clip must not lose the whole run
            errors.append({"id": sample.id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        records.append({
            "id": sample.id,
            "gold": sample.label,
            "gold_index": LABEL_TO_INDEX[sample.label],
            "pred": out["emotion"],
            "pred_index": EMOTION_LABELS.index(out["emotion"]),
            "confidence": out["confidence"],
            "all_emotions": out["all_emotions"],
            "sarcasm_score": out.get("sarcasm_score"),
            "ambiguity_score": out.get("ambiguity_score"),
            "text": sample.text,
            "audio_path": sample.audio_path,
        })

    elapsed = time.time() - started
    if records:
        print(f"\n⏱  {len(records)} clips in {elapsed:.1f}s "
              f"({elapsed / len(records):.2f}s per clip)")
    return records, errors


def score_records(records: list[dict]) -> dict:
    """Compute metrics from prediction records."""
    return compute_metrics(
        y_true=[r["gold_index"] for r in records],
        y_pred=[r["pred_index"] for r in records],
        confidences=[r["confidence"] for r in records],
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    samples, stats = load_manifest(args.manifest)

    print(f"\n📂 Manifest: {args.manifest}")
    print(format_stats(stats))

    if not samples:
        print(
            "\n❌ Nothing to score. Every row was skipped — most likely the "
            "`label` column is still blank.\n"
            "   Build a sheet with:  python -m emotion.training.build_manifest "
            "from-processed --out emotion/data/inhouse.csv"
        )
        return 1

    if len(stats["label_counts"]) < 2:
        print("\n⚠️  Only one class present — F1 and balanced accuracy will be degenerate.")

    if args.limit:
        samples = samples[: args.limit]
        print(f"\n⚠️  --limit {args.limit}: scoring a subset only, not a full evaluation.")

    if args.dry_run:
        print("\n✅ Manifest is valid. Re-run without --dry-run to score it.")
        return 0

    mode = "trained fusion" if args.checkpoint else "zero-shot ensemble"
    try:
        records, errors = run_predictions(samples, args.checkpoint, args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # Setup problems (bad checkpoint, unloadable models) — a traceback here
        # buries the actionable message.
        print(f"\n❌ {exc}")
        return 1

    if not records:
        print("\n❌ Every sample failed during inference. First error:")
        if errors:
            print(f"   {errors[0]['id']}: {errors[0]['error']}")
        return 1

    results = {
        "name": args.name or mode,
        "mode": mode,
        "checkpoint": args.checkpoint,
        "manifest": str(args.manifest),
        "manifest_stats": stats,
        "metrics": score_records(records),
        "predictions": records,
        "errors": errors,
    }

    print(render_report(results))

    out_path = Path(args.out) if args.out else (
        DEFAULT_RESULTS_DIR / f"{'after' if args.checkpoint else 'before'}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n💾 Saved to {out_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    before_path, after_path = args.compare
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))

    b_by_id = {r["id"]: r for r in before["predictions"]}
    a_by_id = {r["id"]: r for r in after["predictions"]}
    shared = sorted(set(b_by_id) & set(a_by_id))

    if not shared:
        print("❌ The two runs share no sample IDs — they cannot be compared.")
        return 1

    dropped = (len(b_by_id) - len(shared)) + (len(a_by_id) - len(shared))
    if dropped:
        print(
            f"⚠️  {dropped} sample(s) appear in only one run and were excluded; "
            f"comparing on the {len(shared)} shared samples."
        )

    # Re-score both on the shared subset so the comparison is strictly paired.
    b_records = [b_by_id[i] for i in shared]
    a_records = [a_by_id[i] for i in shared]
    before["metrics"] = score_records(b_records)
    after["metrics"] = score_records(a_records)

    paired = mcnemar(
        correct_before=[r["gold_index"] == r["pred_index"] for r in b_records],
        correct_after=[r["gold_index"] == r["pred_index"] for r in a_records],
    )

    print(render_comparison(before, after, paired))

    if args.show_flips:
        print("\n  REGRESSIONS  (before correct → after wrong)")
        shown = 0
        for b, a in zip(b_records, a_records):
            if b["gold_index"] == b["pred_index"] and a["gold_index"] != a["pred_index"]:
                print(f"    [{a['id']}] gold={a['gold']} now={a['pred']} "
                      f"({a['confidence']:.2f})  “{a['text'][:60]}”")
                shown += 1
                if shown >= 20:
                    print("    … truncated at 20")
                    break
        if not shown:
            print("    none 🎉")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m emotion.training.evaluate",
        description="Measure emotion-recognition accuracy, and diff two measurements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", help="Labelled manifest (.csv or .jsonl).")
    parser.add_argument(
        "--checkpoint",
        help="Trained fusion checkpoint. Omit to evaluate the zero-shot baseline.",
    )
    parser.add_argument("--name", help="Label for this run, shown in reports.")
    parser.add_argument("--out", help="Where to write the results JSON.")
    parser.add_argument("--device", help="Torch device override (e.g. cpu, mps).")
    parser.add_argument("--limit", type=int, help="Score only the first N samples (smoke test).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate the manifest and exit without loading models.",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE.json", "AFTER.json"),
        help="Diff two saved results files instead of running inference.",
    )
    parser.add_argument(
        "--show-flips", action="store_true",
        help="With --compare, list samples the 'after' run regressed on.",
    )

    args = parser.parse_args(argv)

    if args.compare:
        return cmd_compare(args)
    if not args.manifest:
        parser.error("one of --manifest or --compare is required")
    return cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
