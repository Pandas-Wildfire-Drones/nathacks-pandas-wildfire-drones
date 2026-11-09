"""Three-way triage of a score: FIRE / REVIEW / NOT_FIRE (pure NumPy; usable by perception, ground base and notebooks).

    score <  not_fire_below                       -> NOT_FIRE   (dropped: nobody looks again)
    not_fire_below <= score < fire_at_or_above    -> REVIEW     (a seeker revisit or a human decides)
    score >= fire_at_or_above                     -> FIRE       (treated as fire without further review)

The score can be the CNN's p(fire) (sigmoid of the ONNX logit, see detectors/classifiers.py) or a temperature in
deg C for the pure algorithmic approach: bands are just two cut points on whatever scale the score uses.

Default bands for a model come from its sidecar: `reject_threshold` (pooled OOF recall >= 0.99) and
`confirm_threshold` (precision >= 0.98), i.e. the close-range bands of decision 0004. The search threshold sits
between them.

Caveat: p(fire) is NOT a calibrated probability. Training uses BCE with pos_weight (training/train_cnn.py), which
pushes scores up, so "0.8" does not mean "80 % of such blobs are fires". Pick cut points from data (the sweep below,
or `suggest_bands`), not from intuition about probabilities.

The numbers that matter for night initial attack, in order:
  missed_fire_rate  fires that land in NOT_FIRE / all fires         (the dangerous error: nobody comes back)
  review_rate       REVIEW / all candidates                          (workload for seekers and the ground base)
  fire_precision    fires in FIRE / everything in FIRE               (false alarms that skip review)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .evaluation import choose_threshold_precision_first, choose_threshold_recall_first

__all__ = ["Triage", "TriageBands", "triage_table", "sweep_bands", "suggest_bands"]


class Triage(StrEnum):
    FIRE = "FIRE"
    REVIEW = "REVIEW"
    NOT_FIRE = "NOT_FIRE"


@dataclass(frozen=True)
class TriageBands:
    not_fire_below: float
    fire_at_or_above: float

    def __post_init__(self) -> None:
        if not self.not_fire_below <= self.fire_at_or_above:
            raise ValueError(f"not_fire_below ({self.not_fire_below}) must be <= fire_at_or_above "
                             f"({self.fire_at_or_above})")

    @classmethod
    def from_classifier(cls, clf) -> "TriageBands":
        """(reject_threshold, confirm_threshold) from the model sidecar; confirm is never below reject."""
        lo = float(getattr(clf, "reject_threshold", clf.threshold))
        hi = float(getattr(clf, "confirm_threshold", clf.threshold))
        return cls(lo, max(lo, hi))

    def classify(self, score: float) -> Triage:
        if score >= self.fire_at_or_above:
            return Triage.FIRE
        if score < self.not_fire_below:
            return Triage.NOT_FIRE
        return Triage.REVIEW

    def classify_many(self, scores) -> np.ndarray:
        """Array of Triage values (dtype object), same shape as `scores`."""
        s = np.asarray(scores, dtype=np.float64)
        out = np.full(s.shape, Triage.REVIEW, dtype=object)
        out[s >= self.fire_at_or_above] = Triage.FIRE
        out[s < self.not_fire_below] = Triage.NOT_FIRE
        return out


def triage_table(y_true, scores, bands: TriageBands) -> dict:
    """Counts per band x truth, plus the rates in the module docstring. y_true: 1 = fire, 0 = not fire."""
    y = np.asarray(y_true).astype(bool)
    t = bands.classify_many(scores)
    counts = {b.value: {"fire": int(((t == b) & y).sum()), "not_fire": int(((t == b) & ~y).sum())} for b in Triage}
    n, n_fire, n_not = len(y), int(y.sum()), int((~y).sum())
    fire_band = counts["FIRE"]["fire"] + counts["FIRE"]["not_fire"]
    auto = fire_band + counts["NOT_FIRE"]["fire"] + counts["NOT_FIRE"]["not_fire"]

    def rate(a: int, b: int) -> float:
        return a / b if b else float("nan")

    return {
        "bands": {"not_fire_below": bands.not_fire_below, "fire_at_or_above": bands.fire_at_or_above},
        "n": n, "n_fire": n_fire, "n_not_fire": n_not, "counts": counts,
        "missed_fire_rate": rate(counts["NOT_FIRE"]["fire"], n_fire),
        "fire_or_review_recall": rate(n_fire - counts["NOT_FIRE"]["fire"], n_fire),
        "review_rate": rate(counts["REVIEW"]["fire"] + counts["REVIEW"]["not_fire"], n),
        "fire_precision": rate(counts["FIRE"]["fire"], fire_band),
        "false_alarm_rate": rate(counts["FIRE"]["not_fire"], n_not),   # non-fires that skip review
        "auto_accuracy": rate(counts["FIRE"]["fire"] + counts["NOT_FIRE"]["not_fire"], auto),
    }


def sweep_bands(y_true, scores, lows, highs) -> dict[str, np.ndarray]:
    """Grid over (low, high) cut points. Returns 2D arrays [i_low, j_high] of missed_fire_rate, review_rate,
    fire_precision (NaN where low > high or the FIRE band is empty)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    lows, highs = np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)
    n_fire = max(int(y.sum()), 1)
    below = s[None, :] < lows[:, None]            # (L, N)
    above = s[None, :] >= highs[:, None]          # (H, N)
    missed = (below & y).sum(1) / n_fire          # (L,)
    n_below = below.sum(1)
    n_above = above.sum(1)
    tp_above = (above & y).sum(1)
    review = (len(s) - n_below[:, None] - n_above[None, :]) / max(len(s), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(n_above > 0, tp_above / n_above, np.nan)
    valid = lows[:, None] <= highs[None, :]
    return {"low": lows, "high": highs,
            "missed_fire_rate": np.where(valid, missed[:, None], np.nan),
            "review_rate": np.where(valid, review, np.nan),
            "fire_precision": np.where(valid, prec[None, :], np.nan)}


def suggest_bands(y_true, scores, max_missed_fire_rate: float = 0.01, min_fire_precision: float = 0.98) -> dict:
    """Cut points from labelled scores: NOT_FIRE below the highest threshold that keeps recall >= 1 - max_missed,
    FIRE above the lowest-review threshold with precision >= min_fire_precision. Choose them on TRAINING/CV scores and
    check them on a held-out set; choosing on the set you report is optimistic."""
    lo = choose_threshold_recall_first(y_true, scores, min_recall=1.0 - max_missed_fire_rate)
    hi = choose_threshold_precision_first(y_true, scores, min_precision=min_fire_precision)
    bands = TriageBands(lo["threshold"], max(lo["threshold"], hi["threshold"]))
    return {"bands": bands, "low": lo, "high": hi}
