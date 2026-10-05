"""Identity-level evaluation metrics (Task 10).

Counts are built from per-observation pairs ``(expected, predicted)`` where
each value is a registration number or the literal ``"UNKNOWN"``.  A face
count is never treated as accuracy - only identity agreement is.

* ``CORRECT``             expected == predicted (known or UNKNOWN)
* ``WRONG``               both known but different students
* ``CORRECT REJECTION``   expected UNKNOWN, predicted UNKNOWN
* ``FALSE ACCEPT``        expected UNKNOWN, predicted a known student
* ``FALSE REJECT``        expected known, predicted UNKNOWN
"""

from __future__ import annotations

import csv
from pathlib import Path

__all__ = [
    "UNKNOWN",
    "identity_metrics",
    "classify_pair",
    "load_ground_truth",
    "ResultLabel",
]

UNKNOWN = "UNKNOWN"


class ResultLabel:
    CORRECT = "CORRECT"
    WRONG = "WRONG"
    CORRECT_REJECTION = "CORRECT REJECTION"
    FALSE_ACCEPT = "FALSE ACCEPT"
    FALSE_REJECT = "FALSE REJECT"


def classify_pair(expected: str, predicted: str) -> str:
    expected = (expected or UNKNOWN).strip() or UNKNOWN
    predicted = (predicted or UNKNOWN).strip() or UNKNOWN
    exp_known = expected != UNKNOWN
    pred_known = predicted != UNKNOWN

    if not exp_known and not pred_known:
        return ResultLabel.CORRECT_REJECTION
    if not exp_known and pred_known:
        return ResultLabel.FALSE_ACCEPT
    if exp_known and not pred_known:
        return ResultLabel.FALSE_REJECT
    if expected == predicted:
        return ResultLabel.CORRECT
    return ResultLabel.WRONG


def _safe(num: float, den: float) -> float:
    return num / den if den else 0.0


def identity_metrics(pairs: list[tuple[str, str]]) -> dict:
    """Accuracy / precision / recall / F1 / FAR / FRR / unknown rejection.

    Micro-averaged over observations: a wrong identity counts as a false
    positive for the predicted student *and* a false negative for the
    expected one, so swaps can never inflate the score.
    """
    counts = {
        "total": len(pairs),
        "correct": 0,
        "incorrect": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
        "false_accepts": 0,
        "false_rejects": 0,
        "correct_rejections": 0,
        "wrong_identities": 0,
        "expected_known": 0,
        "expected_unknown": 0,
    }

    for expected, predicted in pairs:
        label = classify_pair(expected, predicted)
        exp = (expected or UNKNOWN).strip() or UNKNOWN
        pred = (predicted or UNKNOWN).strip() or UNKNOWN
        exp_known = exp != UNKNOWN
        pred_known = pred != UNKNOWN

        counts["expected_known" if exp_known else "expected_unknown"] += 1

        if label == ResultLabel.CORRECT:
            counts["correct"] += 1
            counts["true_positive"] += 1
        elif label == ResultLabel.CORRECT_REJECTION:
            counts["correct"] += 1
            counts["true_negative"] += 1
            counts["correct_rejections"] += 1
        elif label == ResultLabel.FALSE_ACCEPT:
            counts["incorrect"] += 1
            counts["false_positive"] += 1
            counts["false_accepts"] += 1
        elif label == ResultLabel.FALSE_REJECT:
            counts["incorrect"] += 1
            counts["false_negative"] += 1
            counts["false_rejects"] += 1
        else:  # WRONG identity
            counts["incorrect"] += 1
            counts["false_positive"] += 1
            counts["false_negative"] += 1
            counts["wrong_identities"] += 1

    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    f1 = _safe(2 * precision * recall, precision + recall)
    far = _safe(counts["false_accepts"], counts["expected_unknown"])
    frr = _safe(counts["false_rejects"], counts["expected_known"])
    unknown_rejection_rate = _safe(
        counts["correct_rejections"], counts["expected_unknown"]
    )

    out = dict(counts)
    out.update(
        {
            "accuracy": _safe(counts["correct"], counts["total"]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "far": far,
            "frr": frr,
            "unknown_rejection_rate": unknown_rejection_rate,
        }
    )
    return out


def load_ground_truth(path: str | Path) -> dict[int, list[str]]:
    """Read ground truth ``frame,reg`` (``UNKNOWN`` allowed as the reg).

    Returns ``{frame_index: [registration_no_or_UNKNOWN, ...]}``.  A frame
    with several faces has one entry per face, ordered **left to right**
    (the evaluator pairs them with the detected boxes in that same order);
    a frame whose row count does not match the detected faces is skipped
    rather than guessed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ground-truth CSV not found: {path}")
    gt: dict[int, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"ground-truth CSV has no header: {path}")
        keys = {(k or "").strip().lower(): k for k in reader.fieldnames}
        frame_key = keys.get("frame") or keys.get("frame_index") or keys.get("frame_idx")
        reg_key = (
            keys.get("reg")
            or keys.get("reg_no")
            or keys.get("registration_no")
            or keys.get("registration_no.")
            or keys.get("expected")
        )
        if frame_key is None or reg_key is None:
            raise ValueError(
                f"ground-truth CSV must have 'frame' and 'reg' columns, "
                f"got {list(reader.fieldnames)}"
            )
        for row in reader:
            raw_frame = (row.get(frame_key) or "").strip()
            if not raw_frame:
                continue
            reg = (row.get(reg_key) or "").strip() or UNKNOWN
            gt.setdefault(int(float(raw_frame)), []).append(reg)
    if not gt:
        raise ValueError(f"ground-truth CSV has no rows: {path}")
    return gt
