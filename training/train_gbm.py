"""[STRETCH GOAL, not the primary model] Gradient-boosting verifier on hand-made blob features.

Purpose: a comparison point for the judges ("does the CNN beat an explainable feature model?"). Same patch set,
same folds and same recall-first threshold rule as train_cnn.py, so the numbers are directly comparable.
Run it only after the CNN pipeline is working.

    python -m training.train_gbm --patch-cache exclude/runs/cnn_flame3/patches.npz \
        --out-dir exclude/runs/gbm_flame3 --export-to perception/models/fire_gbm.onnx --tag flame3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np

from perception.datasets import load_manifest
from perception.features import FEATURE_NAMES, FEATURE_VERSION, patch_features
from perception.patches import PATCH_SPEC
from perception.results_doc import fmt_metric_table, update_section

from .cv import evaluate_oof, make_folds
from .patch_data import PatchSet, build_patch_set

REPO = Path(__file__).resolve().parents[1]


def make_model(seed: int):
    # Classic GradientBoostingClassifier, not HistGradientBoosting: skl2onnx 1.20 fails to convert sklearn 1.9's
    # HistGradientBoosting trees (bool tree attribute). Class balance via sample weights in fit_balanced().
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                      random_state=seed)


def fit_balanced(model, X: np.ndarray, y: np.ndarray):
    pos = max(int(y.sum()), 1)
    w = np.where(y == 1, (len(y) - pos) / pos, 1.0)
    return model.fit(X, y, sample_weight=w)


def export_onnx(model, path: Path) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onx = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, len(FEATURE_NAMES)]))],
                          options={id(model): {"zipmap": False}}, target_opset=17)
    path.write_bytes(onx.SerializeToString())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest")
    ap.add_argument("--patch-cache", help="patches.npz from a train_cnn.py run (recommended: identical patches)")
    ap.add_argument("--out-dir", default=str(REPO / "exclude" / "runs" / f"gbm_{dt.datetime.now():%Y%m%d_%H%M}"))
    ap.add_argument("--export-to")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-doc", default=str(REPO / "perception" / "RESULTS.md"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-recall", type=float, default=0.95)
    ap.add_argument("--min-precision", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.patch_cache:
        ps = PatchSet.load(args.patch_cache)
    elif args.manifest:
        ps = build_patch_set(load_manifest(args.manifest, split="train"), seed=args.seed)
    else:
        raise SystemExit("give --patch-cache or --manifest")
    tag = args.tag or (Path(args.manifest).parent.name if args.manifest else Path(args.patch_cache).parent.name)
    X = patch_features(ps.patches_c)
    folds = make_folds(ps.y, ps.groups, args.folds, args.seed)  # same seed + same patches => same folds as the CNN
    oof = np.full(len(ps.y), np.nan, dtype=np.float32)
    for k, (tr, va) in enumerate(folds):
        m = fit_balanced(make_model(args.seed + k), X[tr], ps.y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        print(f"[gbm fold {k}] done")
    cv = evaluate_oof(ps.y, oof, folds, args.min_recall, args.min_precision)
    final = fit_balanced(make_model(args.seed + 100), X, ps.y)
    onnx_path = out / "fire_gbm.onnx"
    export_onnx(final, onnx_path)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_p = sess.run(None, {"features": X[:64]})[1][:, 1]
    parity = float(np.max(np.abs(onnx_p - final.predict_proba(X[:64])[:, 1])))
    print(f"[export] {onnx_path} max |sklearn - onnx| prob diff = {parity:.2e}")

    meta = {"model_kind": "gbm", "stretch_goal": True, "patch_spec": PATCH_SPEC.to_dict(),
            "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES), "input_name": "features",
            "scale_normalized": ps.scale_normalized,
            "threshold": cv["final_threshold"]["threshold"], "confirm_threshold": cv["confirm_threshold"]["threshold"],
            "reject_threshold": cv["reject_threshold"]["threshold"],
            "threshold_rule": f"recall>={args.min_recall} then max precision (pooled OOF)",
            "cv_summary": cv["summary"], "onnx_parity_max_abs": parity, "tag": tag,
            "created": dt.datetime.now().isoformat(timespec="seconds")}
    (out / "fire_gbm.json").write_text(json.dumps(meta, indent=2, default=float))
    if args.export_to:
        dst = Path(args.export_to)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx_path, dst)
        shutil.copy(out / "fire_gbm.json", dst.with_suffix(".json"))

    rows = [(f"fold {m['fold']} (thr={m['threshold']:.3f})", m) for m in cv["per_fold"]]
    rows += [("**mean ± std over folds**", cv["summary"]),
             (f"pooled OOF @ final thr {cv['final_threshold']['threshold']:.3f}", cv["pooled_oof_at_final_threshold"])]
    md = "\n".join([f"### [STRETCH] GBM verifier: {len(cv['per_fold'])}-fold group CV · {tag}", "",
                    "Stretch-goal comparison against the primary CNN: same patches, folds and threshold rule.", "",
                    fmt_metric_table(rows, ("precision", "recall", "f1", "f2", "accuracy", "ap", "roc_auc"))])
    update_section(args.results_doc, f"gbm_cv_{tag}", md, {"cv": cv, "meta": meta})
    print(md)


if __name__ == "__main__":
    main()
