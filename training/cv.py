"""Cross-validation shared by the CNN (primary) and the GBM ([STRETCH]) so both see identical folds.

* StratifiedGroupKFold: a flight/burn never appears in both train and validation (neighbouring frames leak).
* Out-of-fold (OOF) probabilities give every training patch a prediction from a model that never saw its group.
* Threshold, recall-first: for fold k the threshold is chosen on the OTHER folds' OOF scores and then applied
  to fold k, so per-fold numbers are honest. The final shipped threshold is chosen on all OOF scores pooled.
"""
from __future__ import annotations

import numpy as np

from perception.evaluation import (
    average_precision, binary_metrics, choose_threshold_precision_first, choose_threshold_recall_first, roc_auc,
    summarize_folds,
)

__all__ = ["make_folds", "evaluate_oof"]


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    from sklearn.model_selection import StratifiedGroupKFold  # training-only dependency

    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        raise ValueError(
            "only one group in the training split: CV would leak frames between folds. Give each flight/burn its "
            "own group, or for a single-burn dataset (FLAME 3 Kaggle subset) use time blocks: "
            "build_manifest_flame3.py --group-by sequence")
    k = min(n_splits, n_groups)
    if k < n_splits:
        print(f"[cv] only {n_groups} groups: using {k} folds instead of {n_splits}")
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y, groups))


def evaluate_oof(y: np.ndarray, oof: np.ndarray, folds, min_recall: float = 0.95, min_precision: float = 0.98,
                 reject_recall: float = 0.99) -> dict:
    y = np.asarray(y).astype(int)
    per_fold = []
    for k, (_, va) in enumerate(folds):
        others = np.concatenate([f[1] for j, f in enumerate(folds) if j != k])
        if y[others].sum() == 0 or y[va].sum() == 0:
            continue
        thr = choose_threshold_recall_first(y[others], oof[others], min_recall)["threshold"]
        m = binary_metrics(y[va], oof[va] >= thr)
        m.update(fold=k, threshold=thr, ap=average_precision(y[va], oof[va]), roc_auc=roc_auc(y[va], oof[va]),
                 n=int(len(va)), positives=int(y[va].sum()))
        per_fold.append(m)
    final = choose_threshold_recall_first(y, oof, min_recall)
    pooled = binary_metrics(y, oof >= final["threshold"])
    pooled.update(ap=average_precision(y, oof), roc_auc=roc_auc(y, oof))
    at_half = binary_metrics(y, oof >= 0.5)
    # CONFIRM operating point (close-range INVESTIGATE, docs/decisions/0004). Chosen on single-patch OOF scores, so
    # it is conservative for the multi-frame average the confirmer actually thresholds.
    # Confirming must never be a LOWER bar than searching: on easy data "max recall at precision >= target" can pick a
    # threshold below the search one, which would invert the stages. Raise it to the search threshold if so.
    confirm = choose_threshold_precision_first(y, oof, min_precision)
    if confirm["threshold"] < final["threshold"]:
        confirm = {**confirm, "threshold": final["threshold"], "raised_to_search_threshold": True}
    confirm_m = binary_metrics(y, oof >= confirm["threshold"])
    # REJECT operating point (close range): only call a candidate FALSE_POSITIVE below a threshold that still keeps
    # recall >= reject_recall. Between reject and confirm the confirmer says "inconclusive -> escalate to a human".
    reject = choose_threshold_recall_first(y, oof, reject_recall)
    reject = {**reject, "threshold": min(reject["threshold"], final["threshold"])}
    return {"reject_threshold": reject,"per_fold": per_fold, "summary": summarize_folds(per_fold), "final_threshold": final,
            "pooled_oof_at_final_threshold": pooled, "pooled_oof_at_0.5": at_half, "min_recall": min_recall,
            "confirm_threshold": confirm, "pooled_oof_at_confirm_threshold": confirm_m}
