"""
Phase 7 — Training & Evaluation
===============================

Evaluation-first: before any fine-tuning happens you need a number to beat.

    # 1. Build a labelling sheet from your own pipeline output
    python -m emotion.training.build_manifest from-processed \
        --out emotion/data/inhouse.csv

    # 2. ... fill in the `label` column by hand ...

    # 3. Measure the CURRENT (zero-shot) system
    python -m emotion.training.evaluate \
        --manifest emotion/data/inhouse.csv \
        --name "zero-shot baseline" \
        --out emotion/eval_results/before.json

    # 4. ... fine-tune, producing emotion/checkpoints/best.pt ...

    # 5. Measure the trained system on the SAME manifest
    python -m emotion.training.evaluate \
        --manifest emotion/data/inhouse.csv \
        --checkpoint emotion/checkpoints/best.pt \
        --name "fused (trained)" \
        --out emotion/eval_results/after.json

    # 6. Paired before/after comparison
    python -m emotion.training.evaluate --compare \
        emotion/eval_results/before.json emotion/eval_results/after.json
"""

from .dataset import (
    EmotionSample,
    LABEL_TO_INDEX,
    load_manifest,
    normalize_label,
    write_manifest,
)

__all__ = [
    "EmotionSample",
    "LABEL_TO_INDEX",
    "load_manifest",
    "normalize_label",
    "write_manifest",
]
