"""Evaluate the full mixed algorithmic + ML pipeline on a held-out split and write the results doc.

Run this ONCE per final model on the TEST split (tuning on test = leakage). Training never reads the test split.

    python -m perception.evaluate --manifest exclude/data/flame3/manifest_split.csv --split test \
        --cnn perception/models/fire_cnn.onnx [--gbm perception/models/fire_gbm.onnx] \
        --label "FLAME 3 test" --key pipeline_test_flame3 --results-doc perception/RESULTS.md

Compares, on the same frames:
  1. THRESHOLD only: blob peak >= abs_threshold_c (plain classical baseline) -> what ML adds
  2. classical proposal recall = upper bound on fused recall    -> what the proposer misses, ML can't recover
  3. FUSED + CNN (PRIMARY), with and without the hot-blob bypass (decision 0003), with how often it fired
  4. FUSED + GBM ([STRETCH], only if --gbm is given)
At three levels: patch (verifier on proposals), object (needs masks), frame (always).
Also: 95% bootstrap CI on object F1 (resampling whole groups/flights), and CPU latency per frame.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .datasets import FrameRecord, frame_label, load_frame, load_manifest, load_mask
from .detectors import FusedDetector, HotBypassRule, ThresholdDetector, load_classifier
from .evaluation import (
    average_precision, binary_metrics, blob_label, bootstrap_ci, gt_objects_from_mask, match_detections, metrics_from_counts,
    roc_auc,
)
from .results_doc import fmt_metric_table, update_section

__all__ = ["evaluate_pipeline"]


class _Acc:
    """Accumulates object counts per group and frame predictions for one pipeline variant."""

    def __init__(self):
        self.obj = defaultdict(lambda: [0, 0, 0])
        self.dups = 0
        self.frame_true: list[int] = []
        self.frame_pred: list[int] = []

    def add(self, group, dets_uv, gts, frame_true):
        if gts is not None:
            m = match_detections(dets_uv, gts)
            c = self.obj[group]
            c[0] += m["tp"]; c[1] += m["fp"]; c[2] += m["fn"]
            self.dups += m["duplicates"]
        self.frame_true.append(frame_true)
        self.frame_pred.append(int(len(dets_uv) > 0))

    def result(self) -> dict:
        out = {"frame": binary_metrics(self.frame_true, self.frame_pred)}
        if self.obj:
            tot = np.sum(list(self.obj.values()), axis=0)
            om = metrics_from_counts(int(tot[0]), int(tot[1]), int(tot[2]))
            om["duplicates"] = self.dups
            om["f1_ci95_by_group"] = bootstrap_ci([tuple(v) for v in self.obj.values()], "f1")
            om["recall_ci95_by_group"] = bootstrap_ci([tuple(v) for v in self.obj.values()], "recall")
            out["object"] = om
        return out


def evaluate_pipeline(records: list[FrameRecord], classifiers: dict[str, object],
                      bypass: HotBypassRule = HotBypassRule()) -> dict:
    """Frames use their manifest gsd_m (scale-normalized patches, bypass 'oversize' window) when present."""
    proposer = ThresholdDetector()
    fused = {name: FusedDetector(clf, proposer, bypass=bypass) for name, clf in classifiers.items()}
    fused_nobypass = {name: FusedDetector(clf, proposer, bypass=HotBypassRule(enabled=False))
                      for name, clf in classifiers.items()}
    bypass_stats = {name: defaultdict(lambda: {"fired": 0, "on_non_fire": 0}) for name in classifiers}
    acc = {"threshold": _Acc(), "proposals": _Acc()}
    for name in classifiers:
        acc[f"fused:{name}"] = _Acc()
        acc[f"fused:{name}:no_bypass"] = _Acc()
    patch_y: list[int] = []
    patch_scores = defaultdict(list)
    lat = defaultdict(list)
    has_masks = all(r.has_mask for r in records)

    for rec in records:
        temp = load_frame(rec)
        mask = load_mask(rec)
        gts = gt_objects_from_mask(mask) if mask is not None else None
        y_frame = frame_label(rec, mask)

        t0 = time.perf_counter()
        blobs = proposer.blobs(temp)
        lat["threshold"].append(time.perf_counter() - t0)
        uv_all = [(b.u, b.v) for b in blobs]
        acc["proposals"].add(rec.group, uv_all, gts, y_frame)
        peaks = np.array([b.peak_temp_c for b in blobs])
        abs_thr = proposer.params.abs_threshold_c
        acc["threshold"].add(rec.group, [uv for uv, t in zip(uv_all, peaks) if t >= abs_thr], gts, y_frame)
        if mask is not None:
            patch_y.extend(blob_label(b.bbox, mask) for b in blobs)
            patch_scores["threshold"].extend(peaks.tolist())

        for name, det in fused.items():
            t0 = time.perf_counter()
            _, probs = det.score_blobs(temp, rec.gsd_m)
            dets = det.decide(temp, blobs, probs, rec.gsd_m)
            lat[f"fused:{name}"].append(time.perf_counter() - t0)
            acc[f"fused:{name}"].add(rec.group, [(d.u, d.v) for d in dets], gts, y_frame)
            for b, why in zip(blobs, det.last_bypass_reasons):
                if why in ("saturated", "oversize"):
                    bs = bypass_stats[name][why]
                    bs["fired"] += 1
                    if mask is not None and not blob_label(b.bbox, mask):
                        bs["on_non_fire"] += 1
            nb = fused_nobypass[name].decide(temp, blobs, probs, rec.gsd_m)
            acc[f"fused:{name}:no_bypass"].add(rec.group, [(d.u, d.v) for d in nb], gts, y_frame)
            if mask is not None:
                patch_scores[name].extend(np.asarray(probs).tolist())

    results = {"n_frames": len(records), "n_groups": len({r.group for r in records}), "has_masks": has_masks,
               "variants": {k: a.result() for k, a in acc.items()},
               "latency_ms_per_frame": {k: 1000 * float(np.mean(v)) for k, v in lat.items()},
               "bypass": {"trigger_c": bypass.trigger_c, "saturation_c": bypass.saturation_c,
                          "min_hot_px": bypass.min_hot_px,
                          "by_classifier": {n: dict(v) for n, v in bypass_stats.items()}},
               "scale_normalized_frames": int(sum(r.gsd_m is not None for r in records))}
    if patch_y:
        y = np.array(patch_y)
        patch = {"n": int(len(y)), "positives": int(y.sum())}
        for name, s in patch_scores.items():
            s = np.array(s)
            thr = proposer.params.abs_threshold_c if name == "threshold" else classifiers[name].threshold
            m = binary_metrics(y, s >= thr)
            m.update(ap=average_precision(y, s), roc_auc=roc_auc(y, s), threshold=float(thr))
            patch[name] = m
        results["patch"] = patch
    return results


def _markdown(res: dict, label: str, classifiers: dict) -> str:
    v = res["variants"]
    lines = [f"### Pipeline evaluation: {label}", "",
             f"{res['n_frames']} frames, {res['n_groups']} groups (flights/burns). "
             f"Thresholds were chosen recall-first on training CV and never tuned on this split.", ""]
    names = {"threshold": "THRESHOLD only (peak >= 60 C, classical)", "proposals": "all classical proposals (recall ceiling)"}
    for n in classifiers:
        tag = " [STRETCH]" if n == "gbm" else " (PRIMARY)" if n == "cnn" else ""
        names[f"fused:{n}"] = f"FUSED + {n.upper()}{tag}"
        names[f"fused:{n}:no_bypass"] = f"FUSED + {n.upper()}, no hot-bypass{tag}"
    if res["has_masks"]:
        lines += ["**Object level** (detections matched to mask components):", "",
                  fmt_metric_table([(names[k], v[k]["object"]) for k in v if "object" in v[k]],
                                   ("precision", "recall", "f1", "f2")), "",
                  "95% CI on F1 (bootstrap over groups): " + ", ".join(
                      f"{names[k]} [{v[k]['object']['f1_ci95_by_group'][0]:.3f}, {v[k]['object']['f1_ci95_by_group'][1]:.3f}]"
                      for k in v if "object" in v[k] and k != "proposals"), ""]
    lines += ["**Frame level** (does the frame contain fire):", "",
              fmt_metric_table([(names[k], v[k]["frame"]) for k in v], ("precision", "recall", "f1", "accuracy")), ""]
    if "patch" in res:
        p = res["patch"]
        rows = [(("peak temperature" if k == "threshold" else names[f"fused:{k}"].replace("FUSED + ", "")) + " verifier",
                 p[k]) for k in p if isinstance(p[k], dict)]
        lines += [f"**Patch level** (verifier on {p['n']} proposals, {p['positives']} fire):", "",
                  fmt_metric_table(rows, ("precision", "recall", "f1", "ap", "roc_auc", "threshold")), ""]
    bp = res.get("bypass")
    if bp:
        parts = []
        for n, reasons in bp["by_classifier"].items():
            for why, c in reasons.items():
                parts.append(f"{n.upper()} {why}: fired {c['fired']}x, {c['on_non_fire']} on non-fire")
        lines += [f"**Hot-blob bypass** (trigger ≥ {bp['min_hot_px']} px at ≥ {bp['trigger_c']:.0f} °C = 0.9 × "
                  f"saturation {bp['saturation_c']:.0f} °C, or blob larger than the patch): "
                  + ("; ".join(parts) if parts else "never fired") + ".", ""]
    lines += ["CPU latency per frame (ms, this machine): " +
              ", ".join(f"{k} {ms:.1f}" for k, ms in res["latency_ms_per_frame"].items()),
              "Budget at 9 Hz is 111 ms/frame on the Jetson (context.md 6.5)."]
    return "\n".join(lines)


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", help="manifest split to evaluate ('' = all rows)")
    ap.add_argument("--cnn", help="PRIMARY verifier .onnx (sidecar .json next to it)")
    ap.add_argument("--gbm", help="[STRETCH] GBM verifier .onnx")
    ap.add_argument("--saturation-c", type=float, default=550.0,
                    help="camera saturation for the active gain mode; bypass triggers at 0.9x this (decision 0003)")
    ap.add_argument("--label", default="test split")
    ap.add_argument("--key", default="pipeline_test")
    ap.add_argument("--results-doc", default=str(Path(__file__).with_name("RESULTS.md")))
    ap.add_argument("--json-out", help="also write the raw results JSON here")
    a = ap.parse_args()

    recs = load_manifest(a.manifest, split=a.split or None)
    if not recs:
        raise SystemExit(f"no rows with split={a.split!r} in {a.manifest} (run training/make_split.py first)")
    clfs = {}
    if a.cnn:
        clfs["cnn"] = load_classifier(a.cnn)
    if a.gbm:
        clfs["gbm"] = load_classifier(a.gbm)
    res = evaluate_pipeline(recs, clfs, HotBypassRule(saturation_c=a.saturation_c))
    res.update(label=a.label, manifest=str(a.manifest), split=a.split,
               models={k: getattr(c, "meta", {}) for k, c in clfs.items()})
    md = _markdown(res, a.label, clfs)
    update_section(a.results_doc, a.key, md, res)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2, default=float))
    print(md)


if __name__ == "__main__":
    _main()
