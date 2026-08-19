"""
Phase 7 — Metrics & reporting
=============================

Scoring lives here, separate from the inference loop in ``evaluate.py``, so
the numbers can be recomputed from a saved results file without re-running
the models.

Why these metrics
-----------------
Plain accuracy is misleading on emotion corpora: MELD is ~47% neutral and
in-house meeting audio is worse, so a model that answers "neutral" every
time already scores in the high forties.  The headline number to watch is
therefore **macro F1** (every class counts equally), with **balanced
accuracy** and a **non-neutral accuracy** cut as corroboration.
"""

from __future__ import annotations

from ..emotion_classifier import EMOTION_LABELS

NEUTRAL_INDEX = EMOTION_LABELS.index("neutral")


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float] | None = None,
) -> dict:
    """
    Score a set of predictions against gold labels.

    Parameters
    ----------
    y_true, y_pred :
        Parallel lists of class indices into :data:`EMOTION_LABELS`.
    confidences :
        Optional per-sample probability of the predicted class.  Used to
        report whether the model is confident when it is wrong.
    """
    import warnings

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    if not y_true:
        raise ValueError("No labelled samples to score.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} gold vs {len(y_pred)} predicted.")

    label_indices = list(range(len(EMOTION_LABELS)))

    # Classes that actually occur in the gold labels.  A hand-labelled sheet of
    # meeting audio may contain no `disgust` at all; averaging a forced 0.0 F1
    # for that class into macro F1 would penalise the model for a gap in the
    # test set. Gold support is identical across paired before/after runs (same
    # manifest), so restricting to these classes keeps the two comparable.
    # This is not a loophole: predicting an absent class still costs recall on
    # whatever the true class was.
    present_indices = sorted(set(y_true))

    with warnings.catch_warnings():
        # We deliberately score over all 7 columns; sklearn's "y_pred contains
        # classes not in y_true" note is expected, not a problem.
        warnings.filterwarnings("ignore", message=".*classes not in y_true.*")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=label_indices, zero_division=0,
        )
        macro_f1_present = float(f1_score(
            y_true, y_pred, labels=present_indices, average="macro", zero_division=0,
        ))
        macro_f1_all = float(f1_score(
            y_true, y_pred, labels=label_indices, average="macro", zero_division=0,
        ))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    per_class = {
        EMOTION_LABELS[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in label_indices
    }

    # Accuracy restricted to samples whose *gold* label is not neutral — this
    # is what collapses if the model has learned to answer "neutral".
    non_neutral = [(t, p) for t, p in zip(y_true, y_pred) if t != NEUTRAL_INDEX]
    if non_neutral:
        nn_correct = sum(1 for t, p in non_neutral if t == p)
        non_neutral_accuracy = nn_correct / len(non_neutral)
    else:
        non_neutral_accuracy = None

    # How often does the model simply say "neutral"?
    neutral_rate = sum(1 for p in y_pred if p == NEUTRAL_INDEX) / len(y_pred)

    result = {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1_present, 4),
        "macro_f1_all_classes": round(macro_f1_all, 4),
        "classes_in_gold": len(present_indices),
        "missing_from_gold": [
            EMOTION_LABELS[i] for i in label_indices if i not in present_indices
        ],
        "weighted_f1": round(weighted_f1, 4),
        "non_neutral_accuracy": (
            round(non_neutral_accuracy, 4) if non_neutral_accuracy is not None else None
        ),
        "non_neutral_n": len(non_neutral),
        "predicted_neutral_rate": round(neutral_rate, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_indices).tolist(),
        "labels": list(EMOTION_LABELS),
    }

    if confidences:
        correct_conf = [c for c, t, p in zip(confidences, y_true, y_pred) if t == p]
        wrong_conf = [c for c, t, p in zip(confidences, y_true, y_pred) if t != p]
        result["confidence"] = {
            "mean": round(sum(confidences) / len(confidences), 4),
            "mean_when_correct": (
                round(sum(correct_conf) / len(correct_conf), 4) if correct_conf else None
            ),
            "mean_when_wrong": (
                round(sum(wrong_conf) / len(wrong_conf), 4) if wrong_conf else None
            ),
        }

    return result


def mcnemar(correct_before: list[bool], correct_after: list[bool]) -> dict:
    """
    Exact McNemar test on paired predictions over the same samples.

    Answers the question a raw accuracy delta cannot: *is this improvement
    bigger than sampling noise?*  Only the disagreements carry information —
    samples both systems got right (or both wrong) tell you nothing about
    which is better.

    Returns counts plus a two-sided p-value.  ``p_value`` is ``None`` when the
    two systems never disagree.
    """
    if len(correct_before) != len(correct_after):
        raise ValueError("Paired test requires equal-length correctness vectors.")

    # b = before right / after wrong (regressions)
    # c = before wrong / after right (fixes)
    b = sum(1 for x, y in zip(correct_before, correct_after) if x and not y)
    c = sum(1 for x, y in zip(correct_before, correct_after) if not x and y)
    both_right = sum(1 for x, y in zip(correct_before, correct_after) if x and y)
    both_wrong = sum(1 for x, y in zip(correct_before, correct_after) if not x and not y)

    p_value = None
    if b + c > 0:
        try:
            from scipy.stats import binomtest
            p_value = float(binomtest(c, b + c, 0.5).pvalue)
        except ImportError:
            p_value = None

    return {
        "fixed_by_after": c,
        "broken_by_after": b,
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "n_disagreements": b + c,
        "p_value": round(p_value, 6) if p_value is not None else None,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(value, width: int = 8) -> str:
    if value is None:
        return "—".rjust(width)
    if isinstance(value, float):
        return f"{value:.4f}".rjust(width)
    return str(value).rjust(width)


def render_report(results: dict) -> str:
    """Format one evaluation run as a readable block."""
    m = results["metrics"]
    name = results.get("name") or "unnamed run"
    mode = results.get("mode", "?")

    lines = [
        "",
        "=" * 68,
        f"  {name}   [mode: {mode}]",
        "=" * 68,
        f"  samples scored     : {m['n']}",
        "",
        "  HEADLINE",
        f"    macro F1         : {m['macro_f1']:>6.2%}   <- the number to beat"
        f"  (over {m.get('classes_in_gold', len(m['labels']))} classes present in gold)",
        f"    balanced accuracy: {m['balanced_accuracy']:>6.2%}",
        f"    accuracy         : {m['accuracy']:>6.2%}"
        f"   ({round(m['accuracy'] * m['n'])}/{m['n']} correct)",
        f"    weighted F1      : {m['weighted_f1']:>6.2%}",
    ]

    if m.get("missing_from_gold"):
        lines.append(
            f"    ⚠️  no gold examples of: {', '.join(m['missing_from_gold'])}"
            " — these classes are untested."
        )

    if m["non_neutral_accuracy"] is not None:
        lines.append(
            f"    non-neutral acc  : {m['non_neutral_accuracy']:>6.2%}"
            f"   (on {m['non_neutral_n']} non-neutral samples)"
        )
    lines.append(f"    predicted neutral: {m['predicted_neutral_rate']:>6.2%} of the time")

    if "confidence" in m:
        c = m["confidence"]
        lines += [
            "",
            "  CALIBRATION",
            f"    mean confidence when correct: {_fmt(c['mean_when_correct'])}",
            f"    mean confidence when wrong  : {_fmt(c['mean_when_wrong'])}",
        ]

    lines += ["", "  PER-CLASS", f"    {'label':<10}{'prec':>9}{'recall':>9}{'F1':>9}{'support':>9}"]
    for label, stats in m["per_class"].items():
        # A zero-support class is excluded from macro F1 — flag it so the row
        # of zeros is not misread as "the model is bad at disgust".
        marker = "   (not in gold)" if stats["support"] == 0 else ""
        lines.append(
            f"    {label:<10}{stats['precision']:>9.2%}{stats['recall']:>9.2%}"
            f"{stats['f1']:>9.2%}{stats['support']:>9}{marker}"
        )

    lines += ["", "  CONFUSION MATRIX  (rows = gold, cols = predicted)"]
    labels = m["labels"]
    header = "    " + " " * 10 + "".join(f"{lab[:7]:>8}" for lab in labels)
    lines.append(header)
    for i, row in enumerate(m["confusion_matrix"]):
        lines.append(f"    {labels[i]:<10}" + "".join(f"{v:>8}" for v in row))

    errors = results.get("errors") or []
    if errors:
        lines += ["", f"  ⚠️  {len(errors)} sample(s) failed during inference and were excluded."]
        for err in errors[:3]:
            lines.append(f"      {err['id']}: {err['error'][:70]}")

    lines.append("=" * 68)
    return "\n".join(lines)


def render_comparison(before: dict, after: dict, paired: dict | None = None) -> str:
    """Format a before/after diff of two evaluation runs."""
    bm, am = before["metrics"], after["metrics"]
    b_name = before.get("name") or "before"
    a_name = after.get("name") or "after"

    def row(caption: str, key: str) -> str:
        bv, av = bm.get(key), am.get(key)
        if bv is None or av is None:
            return f"    {caption:<22}{_fmt(bv, 10)}{_fmt(av, 10)}{'—':>12}"
        delta = av - bv
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        # Deltas are in percentage POINTS, not percent — a move from 47.8% to
        # 52.6% is +4.8pp, not +10% relative. Labelling it "pp" keeps the two
        # readings from being confused.
        return (f"    {caption:<22}{bv:>10.2%}{av:>10.2%}"
                f"{arrow + f' {delta * 100:+.2f}pp':>13}")

    lines = [
        "",
        "=" * 68,
        "  BEFORE  →  AFTER",
        "=" * 68,
        f"    before : {b_name}  [{before.get('mode', '?')}]",
        f"    after  : {a_name}  [{after.get('mode', '?')}]",
        f"    scored on {am['n']} shared samples",
        "",
        f"    {'metric':<22}{'before':>10}{'after':>10}{'change':>13}",
        "    " + "-" * 55,
        row("macro F1", "macro_f1"),
        row("balanced accuracy", "balanced_accuracy"),
        row("accuracy", "accuracy"),
        row("weighted F1", "weighted_f1"),
        row("non-neutral accuracy", "non_neutral_accuracy"),
        row("predicted neutral rate", "predicted_neutral_rate"),
        "",
        "  PER-CLASS F1",
        f"    {'label':<12}{'before':>10}{'after':>10}{'change':>13}{'support':>10}",
        "    " + "-" * 55,
    ]

    for label in am["per_class"]:
        support = am["per_class"][label]["support"]
        if support == 0:
            lines.append(f"    {label:<12}{'—':>10}{'—':>10}{'—':>13}{0:>10}   (not in gold)")
            continue
        bf = bm["per_class"].get(label, {}).get("f1", 0.0)
        af = am["per_class"][label]["f1"]
        delta = af - bf
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        lines.append(
            f"    {label:<12}{bf:>10.2%}{af:>10.2%}"
            f"{arrow + f' {delta * 100:+.2f}pp':>13}{support:>10}"
        )

    if paired:
        lines += [
            "",
            "  PAIRED ANALYSIS  (same samples, both systems)",
            f"    fixed by 'after'   : {paired['fixed_by_after']}",
            f"    broken by 'after'  : {paired['broken_by_after']}",
            f"    both correct       : {paired['both_correct']}",
            f"    both wrong         : {paired['both_wrong']}",
        ]
        p = paired["p_value"]
        if p is None:
            note = (
                "no disagreements — the two runs are identical"
                if paired["n_disagreements"] == 0
                else "scipy not installed, p-value unavailable"
            )
            lines.append(f"    McNemar p-value    : — ({note})")
        else:
            verdict = "significant at p<0.05" if p < 0.05 else "NOT significant at p<0.05"
            lines.append(f"    McNemar p-value    : {p:.4g}  ({verdict})")
            if p >= 0.05:
                lines.append(
                    "    ↳ the change could plausibly be sampling noise; "
                    "collect more labels before trusting it."
                )

    lines.append("=" * 68)
    return "\n".join(lines)
