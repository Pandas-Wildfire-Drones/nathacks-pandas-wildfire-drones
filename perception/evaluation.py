"""Metrics shared by training (cross-validation) and pipeline evaluation (test split). NumPy only.

Which number matters? F1 = harmonic mean of
  precision = TP / (TP + FP)   "of the alarms we raised, how many were real fires"
  recall    = TP / (TP + FN)   "of the real fires, how many did we catch"
For night initial attack RECALL matters more: a missed spot fire can grow overnight, while a false alarm only
costs a seeker an INVESTIGATE pass. So thresholds are chosen recall-first (recall >= target, then the best
precision), and we still report F1 for the >90% F1 design criterion. Accuracy is reported but is misleading
here: most patches/frames are not fire, so "always say no fire" scores high accuracy and zero recall.

Three evaluation levels (report them separately, never mix):
  * patch   : verifier only, on threshold proposals (what the CNN/GBM is trained on)
  * object  : pipeline detections matched to ground-truth fire objects (connected components of the mask)
  * frame   : "does this frame contain fire" (the only level available when a dataset has image labels only)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = [
    "binary_metrics", "metrics_from_counts", "choose_threshold_recall_first", "choose_threshold_precision_first", "average_precision", "roc_auc",
    "GtObject", "gt_objects_from_mask", "match_detections", "blob_label", "BLOB_IGNORE", "drop_ignored", "bootstrap_ci",
    "summarize_folds",
]


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int | None = None) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    out = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall, "f1": f1, "f2": f2}
    if tn is not None:
        total = tp + fp + fn + tn
        out.update(tn=int(tn), accuracy=(tp + tn) / total if total else 0.0,
                   specificity=tn / (tn + fp) if tn + fp else 0.0)
    return out


def binary_metrics(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return metrics_from_counts(int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((~y & ~p).sum()))


def choose_threshold_recall_first(y_true, scores, min_recall: float = 0.95) -> dict:
    """Highest-precision threshold whose recall >= min_recall (ties -> higher threshold).

    If no threshold reaches the target, falls back to the threshold with the best recall and flags it.
    Returns {"threshold", "precision", "recall", "f1", "met_target"}.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0:
        raise ValueError("no positives: cannot choose a recall-first threshold")
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(~y_sorted)
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]  # last index of each distinct score
    thr, tp, fp = s_sorted[last], tps[last], fps[last]
    recall = tp / y.sum()
    precision = tp / np.maximum(tp + fp, 1)
    ok = recall >= min_recall - 1e-12
    if ok.any():
        idx = np.nonzero(ok)[0]
        best = idx[np.lexsort((-thr[idx], -precision[idx]))[0]]  # max precision, then max threshold
        met = True
    else:
        best, met = int(np.argmax(recall)), False
    p, r = float(precision[best]), float(recall[best])
    return {"threshold": float(thr[best]), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "met_target": met, "min_recall": min_recall}


def _pr_curve(y_true, scores):
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tps, fps = np.cumsum(y_sorted), np.cumsum(~y_sorted)
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    thr, tp, fp = s_sorted[last], tps[last], fps[last]
    return thr, tp / max(int(y.sum()), 1), tp / np.maximum(tp + fp, 1)


def choose_threshold_precision_first(y_true, scores, min_precision: float = 0.98) -> dict:
    """CONFIRM operating point: highest-recall threshold whose precision >= min_precision (ties -> higher threshold).
    Falls back to the highest-precision threshold (flagged) if the target is unreachable."""
    if np.asarray(y_true).astype(bool).sum() == 0:
        raise ValueError("no positives: cannot choose a precision-first threshold")
    thr, recall, precision = _pr_curve(y_true, scores)
    ok = precision >= min_precision - 1e-12
    if ok.any():
        idx = np.nonzero(ok)[0]
        best, met = idx[np.lexsort((-thr[idx], -recall[idx]))[0]], True
    else:
        best, met = int(np.argmax(precision)), False
    p, r = float(precision[best]), float(recall[best])
    return {"threshold": float(thr[best]), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "met_target": met, "min_precision": min_precision}


def average_precision(y_true, scores) -> float:
    """Area under the precision-recall curve (step interpolation, like sklearn's average_precision_score)."""
    y = np.asarray(y_true).astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def roc_auc(y_true, scores) -> float:
    y = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class GtObject:
    u: float
    v: float
    bbox: tuple[int, int, int, int]  # (u0, v0, u1, v1), exclusive end
    area_px: int


def gt_objects_from_mask(mask: np.ndarray, min_area_px: int = 1) -> list[GtObject]:
    """Ground-truth fire objects = 8-connected components of the binary fire mask."""
    labels, _ = ndimage.label(np.asarray(mask).astype(bool), structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        m = labels[sl] == i
        area = int(m.sum())
        if area < min_area_px:
            continue
        rows, cols = np.nonzero(m)
        out.append(GtObject(u=float(cols.mean() + sl[1].start + 0.5), v=float(rows.mean() + sl[0].start + 0.5),
                            bbox=(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop), area_px=area))
    return out


def match_detections(dets_uv: list[tuple[float, float]], gts: list[GtObject], tol_px: float = 3.0) -> dict:
    """Greedy object matching. A detection matches a GT object if its centroid lies inside the GT bbox grown by
    tol_px. Each GT is matched once; extra detections on an already-matched GT count as `duplicates` (the
    tracker merges those), not as false positives. Returns counts tp/fp/fn/duplicates + matched index pairs."""
    matched_gt: set[int] = set()
    tp = fp = dup = 0
    pairs = []
    for di, (u, v) in enumerate(dets_uv):
        hits = [gi for gi, g in enumerate(gts)
                if g.bbox[0] - tol_px <= u <= g.bbox[2] + tol_px and g.bbox[1] - tol_px <= v <= g.bbox[3] + tol_px]
        if not hits:
            fp += 1
            continue
        free = [gi for gi in hits if gi not in matched_gt]
        if not free:
            dup += 1
            continue
        gi = min(free, key=lambda k: (gts[k].u - u) ** 2 + (gts[k].v - v) ** 2)
        matched_gt.add(gi)
        pairs.append((di, gi))
        tp += 1
    return {"tp": tp, "fp": fp, "fn": len(gts) - len(matched_gt), "duplicates": dup, "pairs": pairs}


BLOB_IGNORE = -1


def blob_label(bbox: tuple[float, float, float, float], mask: np.ndarray, tol_px: int = 1,
               ignore: np.ndarray | None = None) -> int:
    """1 if the fire mask has any pixel inside the blob bbox grown by tol_px. Used to label proposals for
    training AND for patch-level evaluation, so both use the same definition of a 'fire' proposal.
    With an `ignore` mask: no fire but touching an ignore pixel -> BLOB_IGNORE (-1): skip it in training and scoring."""
    u0, v0, u1, v1 = (int(x) for x in bbox)
    h, w = mask.shape
    win = (slice(max(v0 - tol_px, 0), min(v1 + tol_px, h)), slice(max(u0 - tol_px, 0), min(u1 + tol_px, w)))
    if mask[win].any():
        return 1
    if ignore is not None and ignore[win].any():
        return BLOB_IGNORE
    return 0


def drop_ignored(dets_uv: list[tuple[float, float]], ignore: np.ndarray | None, tol_px: int = 1):
    """Remove detections whose centroid falls in (or within tol_px of) an ignore region: they are neither TP nor FP."""
    if ignore is None or not dets_uv:
        return list(dets_uv)
    grown = ndimage.binary_dilation(ignore, iterations=tol_px) if tol_px > 0 else ignore
    h, w = grown.shape
    return [(u, v) for u, v in dets_uv if not grown[min(max(int(v), 0), h - 1), min(max(int(u), 0), w - 1)]]


def bootstrap_ci(per_unit_counts: list[tuple[int, int, int]], metric: str = "f1", n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI by resampling units (frames, or better whole flights) with replacement."""
    if not per_unit_counts:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    c = np.asarray(per_unit_counts, dtype=np.int64)
    vals = []
    for _ in range(n_boot):
        s = c[rng.integers(0, len(c), len(c))].sum(0)
        vals.append(metrics_from_counts(int(s[0]), int(s[1]), int(s[2]))[metric])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize_folds(fold_metrics: list[dict], keys=("precision", "recall", "f1", "f2", "accuracy", "ap", "roc_auc")) -> dict:
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m and m[k] == m[k]]  # skip NaN
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out
