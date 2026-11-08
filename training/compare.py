"""Side-by-side scores for the three detection approaches on the SAME candidates, with FIRE / REVIEW / NOT_FIRE triage.

    algorithmic  ThresholdDetector proposal, decided by its peak temperature (deg C). No ML.
    model        the CNN verifier's p(fire) at the proposal centre. No temperature gate, no hot-blob bypass:
                 isolates what the network itself decides.
    pipeline     what FusedDetector emits: p(fire), raised to `bypass_confidence` when the hot-blob bypass fires
                 (decision 0003). Same number the tracker and the confirm stage see.

Why candidates and not whole frames: the CNN classifies a patch around a given point; it does not search the frame.
Every approach is therefore scored on the classical proposals (the drone's real input to the CNN). Fires the
proposer never finds are counted separately (`missed_objects`): no verifier can recover them.

Two data sources:
  handlabels  training/label_tool.py output (real FLAME 3 blobs judged from the RGB photo). TEST ONLY.
  manifest    any manifest with masks (synthetic, composites): each proposal is labelled by mask overlap.

    python -m training.compare handlabels --candidates exclude/labels/candidates.csv \
        --labels exclude/labels/labels_DoqCA.csv --cnn perception/models/fire_cnn_flame3comp.onnx
    python -m training.compare manifest --manifest exclude/data/flame3_composite/manifest_split.csv --split test \
        --cnn perception/models/fire_cnn_flame3comp.onnx
    options: --model-bands LO HI  --algo-bands LO_C HI_C  --key K (write a RESULTS.md section)  --json-out PATH

Interactive version with plots and threshold sliders: training/compare_approaches.ipynb.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from perception.datasets import FrameRecord, load_frame, load_ignore, load_manifest, load_mask
from perception.detectors import FusedDetector, HotBypassRule, ThresholdDetector, load_classifier
from perception.evaluation import (
    BLOB_IGNORE, average_precision, binary_metrics, blob_label, roc_auc,
)
from perception.results_doc import update_section
from perception.triage import TriageBands, triage_table

from .label_tool import labelled_targets, load_candidates, load_labels

__all__ = ["APPROACHES", "TEMP_BANDS", "ScoredCandidates", "score_handlabels", "score_manifest", "score_points",
           "default_bands",
           "default_search_thresholds", "compare", "group_bootstrap_auc", "markdown"]

APPROACHES = ("algorithmic", "model", "pipeline")
# Peak-temperature bands for per-band reporting (context.md section 10). Reporting bins only: never decision rules.
TEMP_BANDS = (("<80C", -np.inf, 80.0), ("80-200C", 80.0, 200.0), (">=200C", 200.0, np.inf))


@dataclass
class ScoredCandidates:
    source: str
    y: np.ndarray                    # 1 = fire, 0 = not fire
    peak_c: np.ndarray
    area_px: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p_model: np.ndarray
    bypass: list                     # 'saturated' | 'oversize' | 'excluded:...' | None, per candidate
    group: np.ndarray
    frame_path: list
    gsd_m: list
    saturation_c: list
    label_text: list                 # hand label ('active', 'smoulder', 'not_fire') or 'fire' / 'not_fire'
    rgb_path: list = field(default_factory=list)
    rgb_box: list = field(default_factory=list)
    bypass_confidence: float = 0.9
    n_objects: int = 0               # ground-truth objects (manifest source only)
    missed_objects: int = 0          # objects with no proposal at all: lost before any verifier runs

    def __len__(self) -> int:
        return len(self.y)

    @property
    def bypassed(self) -> np.ndarray:
        return np.array([b in ("saturated", "oversize") for b in self.bypass], dtype=bool)

    @property
    def p_pipeline(self) -> np.ndarray:
        return np.where(self.bypassed, np.maximum(self.p_model, self.bypass_confidence), self.p_model)

    def scores(self, approach: str) -> np.ndarray:
        return {"algorithmic": self.peak_c, "model": self.p_model, "pipeline": self.p_pipeline}[approach]

    def subset(self, keep) -> "ScoredCandidates":
        keep = np.asarray(keep, dtype=bool)
        idx = np.flatnonzero(keep)

        def pick(x):
            return x[keep] if isinstance(x, np.ndarray) else [x[i] for i in idx] if x else x

        return ScoredCandidates(self.source, *(pick(getattr(self, f)) for f in (
            "y", "peak_c", "area_px", "u", "v", "p_model", "bypass", "group", "frame_path", "gsd_m", "saturation_c",
            "label_text", "rgb_path", "rgb_box")), bypass_confidence=self.bypass_confidence)


def _nearest_blob(blobs, u: float, v: float, tol_px: float = 1.5):
    if not blobs:
        return None
    d = [(b.u - u) ** 2 + (b.v - v) ** 2 for b in blobs]
    k = int(np.argmin(d))
    return blobs[k] if d[k] <= tol_px ** 2 else None


def score_handlabels(candidates_csv, labels_csv, classifier, positive=("active", "smoulder"),
                     bypass: HotBypassRule = HotBypassRule(), proposer: ThresholdDetector | None = None,
                     bypass_confidence: float = 0.9) -> ScoredCandidates:
    """Hand-labelled candidates ('unsure' dropped). The bypass is evaluated on the proposer blob at the candidate
    (candidates come from the same ThresholdDetector, so the blob is reproduced exactly)."""
    cands = load_candidates(candidates_csv)
    labels = load_labels(labels_csv)
    kept, y = labelled_targets(cands, labels, positive)
    proposer = proposer or ThresholdDetector()
    det = FusedDetector(classifier, proposer, bypass=bypass)
    p = np.zeros(len(kept), dtype=np.float32)
    reasons: list = [None] * len(kept)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for k, c in enumerate(kept):
        by_frame[c.frame_path].append(k)
    for path, ks in by_frame.items():
        temp = load_frame(FrameRecord(frame_path=path, group="", label=0))
        gsds = [kept[k].gsd_m for k in ks]
        gsd = None if any(g is None for g in gsds) else gsds
        p[ks] = classifier.predict_proba(temp, [(kept[k].u, kept[k].v) for k in ks], gsd_m=gsd)
        blobs = proposer.blobs(temp)
        for k in ks:
            c = kept[k]
            b = _nearest_blob(blobs, c.u, c.v)
            if b is not None:
                reasons[k] = det.bypass_reasons(temp, [b], c.gsd_m, saturation_c=c.saturation_c)[0]
    return ScoredCandidates(
        source=f"handlabels:{Path(labels_csv).stem}", y=y.astype(int),
        peak_c=np.array([c.peak_c for c in kept], dtype=np.float64),
        area_px=np.array([c.area_px for c in kept], dtype=np.float64),
        u=np.array([c.u for c in kept]), v=np.array([c.v for c in kept]), p_model=p, bypass=reasons,
        group=np.array([c.group for c in kept]), frame_path=[c.frame_path for c in kept],
        gsd_m=[c.gsd_m for c in kept], saturation_c=[c.saturation_c for c in kept],
        label_text=[labels[c.cand_id]["label"] for c in kept], rgb_path=[c.rgb_path for c in kept],
        rgb_box=[c.rgb_box for c in kept], bypass_confidence=bypass_confidence)


def _frame_gsd(rec: FrameRecord, height_px: int):
    rows = rec.gsd_rows(height_px) if rec.has_geometry else None
    if rows is not None:
        g = rows[0]
        return lambda v: float(g[min(max(int(v), 0), len(g) - 1)])
    return rec.gsd_m


def score_manifest(records: list[FrameRecord], classifier, bypass: HotBypassRule = HotBypassRule(),
                   proposer: ThresholdDetector | None = None, bypass_confidence: float = 0.9) -> ScoredCandidates:
    """Every proposal in every masked frame, labelled by mask overlap (perception.evaluation.blob_label)."""
    recs = [r for r in records if r.has_mask]
    if not recs:
        raise ValueError("no frames with masks: proposal-level labels need masks (use handlabels for real frames)")
    proposer = proposer or ThresholdDetector()
    det = FusedDetector(classifier, proposer, bypass=bypass)
    cols = defaultdict(list)
    n_obj = missed = 0
    for rec in recs:
        temp, mask, ignore = load_frame(rec), load_mask(rec), load_ignore(rec)
        gsd = _frame_gsd(rec, temp.shape[0])
        blobs, probs = det.score_blobs(temp, gsd)
        reasons = det.bypass_reasons(temp, blobs, gsd, saturation_c=rec.saturation_c)
        n_c, n_miss = _uncovered_components(mask, blobs)
        n_obj += n_c
        missed += n_miss
        for b, p, why in zip(blobs, probs, reasons):
            y = blob_label(b.bbox, mask, ignore=ignore)
            if y == BLOB_IGNORE:
                continue
            for k, val in (("y", y), ("peak_c", b.peak_temp_c), ("area_px", b.area_px), ("u", b.u), ("v", b.v),
                           ("p", float(p)), ("bypass", why), ("group", rec.group), ("frame_path", rec.frame_path),
                           ("gsd_m", gsd(b.v) if callable(gsd) else gsd), ("saturation_c", rec.saturation_c),
                           ("label_text", "fire" if y else "not_fire"), ("rgb_path", rec.rgb_path),
                           ("rgb_box", (0.0, 0.0, 1.0, 1.0))):
                cols[k].append(val)
    return ScoredCandidates(
        source="manifest", y=np.array(cols["y"], dtype=int), peak_c=np.array(cols["peak_c"], dtype=np.float64),
        area_px=np.array(cols["area_px"], dtype=np.float64), u=np.array(cols["u"]), v=np.array(cols["v"]),
        p_model=np.array(cols["p"], dtype=np.float32), bypass=cols["bypass"], group=np.array(cols["group"]),
        frame_path=cols["frame_path"], gsd_m=cols["gsd_m"], saturation_c=cols["saturation_c"],
        label_text=cols["label_text"], rgb_path=cols["rgb_path"], rgb_box=cols["rgb_box"],
        bypass_confidence=bypass_confidence, n_objects=n_obj, missed_objects=missed)


def score_points(records: list[FrameRecord], classifier, n_neg: int = 20, n_pos: int = 20,
                 bypass: HotBypassRule = HotBypassRule(), proposer: ThresholdDetector | None = None,
                 bypass_confidence: float = 0.9, warm_dc: float = 8.0, clear_m: float = 1.2, peak_radius_m: float = 0.5,
                 seed: int = 0) -> ScoredCandidates:
    """Balanced HARD view straight from the masks (not from proposals), per frame:
      n_neg  warm not-fire pixels (>= warm_dc above the frame median, >= clear_m from fire): char, warm ground, rocks
      n_pos  fire pixels, half from the cooler half (residual heat), half from all fire pixels
    'algorithmic' score = hottest pixel within peak_radius_m (what a blob's peak would read). The bypass is checked on
    that same neighbourhood (a pseudo-blob), not on whatever big proposer blob contains the point: the drone would
    score that blob at its own centroid, never at this point. Use it where proposals are nearly all fire (NADIR)."""
    from perception.detectors.threshold import Blob

    from .patch_data import sample_hard_points

    proposer = proposer or ThresholdDetector()
    det = FusedDetector(classifier, proposer, bypass=bypass)
    cols = defaultdict(list)
    for k, rec in enumerate(records):
        temp, mask, ignore = load_frame(rec), load_mask(rec), load_ignore(rec)
        if mask is None:
            continue
        rng = np.random.default_rng(seed + k)
        pts = sample_hard_points(temp, mask, ignore, rec, n_neg, n_pos - n_pos // 2, rng, warm_dc, clear_m)
        if n_pos // 2 and mask.any():
            vs, us = np.nonzero(mask)
            for i in rng.choice(len(vs), min(n_pos // 2, len(vs)), replace=False):
                pts.append(((us[i] + 0.5, vs[i] + 0.5), 1))
        if not pts:
            continue
        gsd = _frame_gsd(rec, temp.shape[0])
        gs = [gsd(v) if callable(gsd) else gsd for (_, v), _ in pts]
        p = classifier.predict_proba(temp, [uv for uv, _ in pts], gsd_m=None if any(g is None for g in gs) else gs)
        h, w = temp.shape
        for ((u, v), lab), prob, g in zip(pts, p, gs):
            r = max(int(round(peak_radius_m / (g or 0.1))), 1)
            v0, v1, u0, u1 = max(int(v) - r, 0), min(int(v) + r + 1, h), max(int(u) - r, 0), min(int(u) + r + 1, w)
            win = temp[v0:v1, u0:u1]
            peak = float(win.max())
            pseudo = Blob(u, v, int(win.size), peak, float(win.mean()), float(np.median(temp)), (u0, v0, u1, v1))
            why = det.bypass_reasons(temp, [pseudo], g, saturation_c=rec.saturation_c)[0]
            kind = "warm_not_fire" if not lab else ("fire<120C" if temp[int(v), int(u)] < 120 else "fire>=120C")
            for key, val in (("y", lab), ("peak_c", peak), ("area_px", 0.0), ("u", u), ("v", v), ("p", float(prob)),
                             ("bypass", why), ("group", rec.group), ("frame_path", rec.frame_path), ("gsd_m", g),
                             ("saturation_c", rec.saturation_c), ("label_text", kind)):
                cols[key].append(val)
    return ScoredCandidates(
        source="points", y=np.array(cols["y"], dtype=int), peak_c=np.array(cols["peak_c"], dtype=np.float64),
        area_px=np.array(cols["area_px"]), u=np.array(cols["u"]), v=np.array(cols["v"]),
        p_model=np.array(cols["p"], dtype=np.float32), bypass=cols["bypass"], group=np.array(cols["group"]),
        frame_path=cols["frame_path"], gsd_m=cols["gsd_m"], saturation_c=cols["saturation_c"],
        label_text=cols["label_text"], bypass_confidence=bypass_confidence)


def _uncovered_components(mask: np.ndarray, blobs) -> tuple[int, int]:
    """(fire components, components no proposal bbox (+-1 px) touches). Real masks split into many small
    components that one blob covers together, so one-to-one matching would call most of them missed."""
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return 0, 0
    cover = np.zeros(mask.shape, bool)
    h, w = mask.shape
    for b in blobs:
        u0, v0, u1, v1 = (int(x) for x in b.bbox)
        cover[max(v0 - 1, 0):min(v1 + 1, h), max(u0 - 1, 0):min(u1 + 1, w)] = True
    touched = np.unique(lab[cover & (lab > 0)])
    return n, n - len(touched)


def default_bands(classifier, proposer: ThresholdDetector | None = None, algo_fire_c: float = 150.0
                  ) -> dict[str, TriageBands]:
    """model/pipeline: the sidecar's (reject, confirm). algorithmic: NOT_FIRE below the proposer's absolute threshold
    (60 C), FIRE at >= algo_fire_c. [ASSUMPTION] 150 C = HotBypassRule.oversize_min_c, above night-time decoys."""
    proposer = proposer or ThresholdDetector()
    model = TriageBands.from_classifier(classifier)
    return {"algorithmic": TriageBands(proposer.params.abs_threshold_c, algo_fire_c), "model": model,
            "pipeline": model}


def default_search_thresholds(classifier, proposer: ThresholdDetector | None = None) -> dict[str, float]:
    """The single yes/no threshold each approach uses on the drone today."""
    proposer = proposer or ThresholdDetector()
    return {"algorithmic": proposer.params.abs_threshold_c, "model": float(classifier.threshold),
            "pipeline": float(classifier.threshold)}


def group_bootstrap_auc(y, s, groups, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % CI of ROC AUC, resampling whole groups (neighbouring frames are not independent)."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def compare(sc: ScoredCandidates, bands: dict[str, TriageBands], search: dict[str, float],
            n_boot: int = 1000) -> dict:
    """Per approach: ROC AUC (+ group-bootstrap CI), AP, the single-threshold decision, and the 3-band triage.
    Also the same triage split by peak-temperature band."""
    out: dict = {"source": sc.source, "n": len(sc), "n_fire": int(sc.y.sum()), "n_groups": len(set(sc.group)),
                 "bypass_fired": int(sc.bypassed.sum()), "n_objects": sc.n_objects,
                 "missed_objects": sc.missed_objects, "approaches": {}, "by_temp_band": {}}
    for a in APPROACHES:
        s = sc.scores(a)
        decide = s >= search[a]
        if a == "pipeline":
            decide = decide | sc.bypassed
        out["approaches"][a] = {
            "roc_auc": roc_auc(sc.y, s), "roc_auc_ci95": group_bootstrap_auc(sc.y, s, sc.group, n_boot),
            "ap": average_precision(sc.y, s), "search_threshold": search[a],
            "search": binary_metrics(sc.y, decide), "triage": triage_table(sc.y, s, bands[a])}
    for name, lo, hi in TEMP_BANDS:
        sel = (sc.peak_c >= lo) & (sc.peak_c < hi)
        if sel.any():
            out["by_temp_band"][name] = {a: triage_table(sc.y[sel], sc.scores(a)[sel], bands[a]) for a in APPROACHES}
    return out


def _f(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}" if isinstance(x, float) else str(x)


def markdown(res: dict, label: str) -> str:
    names = {"algorithmic": "algorithmic (peak °C)", "model": "CNN alone (no bypass)",
             "pipeline": "full pipeline (CNN + bypass)"}
    lines = [f"### Approach comparison: {label}", "",
             f"{res['n']} candidates ({res['n_fire']} fire), {res['n_groups']} groups. "
             f"Hot-blob bypass fired on {res['bypass_fired']}."]
    if res["n_objects"]:
        lines.append(f"Proposer never touched {res['missed_objects']} of {res['n_objects']} fire mask components "
                     "(no approach can recover those).")
    lines += ["", "| approach | ROC AUC [95% CI, groups] | AP | search thr | precision | recall |",
              "|---|---|---|---|---|---|"]
    for a, r in res["approaches"].items():
        lo, hi = r["roc_auc_ci95"]
        lines.append(f"| {names[a]} | {_f(r['roc_auc'])} [{_f(lo)}, {_f(hi)}] | {_f(r['ap'])} | "
                     f"{_f(r['search_threshold'])} | {_f(r['search']['precision'])} | {_f(r['search']['recall'])} |")
    lines += ["", "Triage (NOT_FIRE < low ≤ REVIEW < high ≤ FIRE):", "",
              "| approach | low | high | fires missed (NOT_FIRE) | review rate | FIRE precision | non-fires auto-FIRE |",
              "|---|---|---|---|---|---|---|"]
    for a, r in res["approaches"].items():
        t = r["triage"]
        lines.append(f"| {names[a]} | {_f(t['bands']['not_fire_below'])} | {_f(t['bands']['fire_at_or_above'])} | "
                     f"{t['counts']['NOT_FIRE']['fire']}/{t['n_fire']} ({_f(t['missed_fire_rate'])}) | "
                     f"{_f(t['review_rate'])} | {_f(t['fire_precision'])} | "
                     f"{t['counts']['FIRE']['not_fire']}/{t['n_not_fire']} |")
    return "\n".join(lines)


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("handlabels")
    h.add_argument("--candidates", required=True)
    h.add_argument("--labels", required=True)
    h.add_argument("--positive", nargs="+", default=["active", "smoulder"])
    m = sub.add_parser("manifest")
    m.add_argument("--manifest", required=True)
    m.add_argument("--split", default="test")
    for p in (h, m):
        p.add_argument("--cnn", required=True)
        p.add_argument("--saturation-c", type=float, default=550.0, help="default gain-mode saturation for the bypass")
        p.add_argument("--model-bands", type=float, nargs=2, metavar=("LO", "HI"))
        p.add_argument("--algo-bands", type=float, nargs=2, metavar=("LO_C", "HI_C"))
        p.add_argument("--label")
        p.add_argument("--key", help="write this section into --results-doc")
        p.add_argument("--results-doc", default=str(Path(__file__).resolve().parents[1] / "perception" / "RESULTS.md"))
        p.add_argument("--json-out")
    a = ap.parse_args()
    clf = load_classifier(a.cnn)
    bypass = HotBypassRule(saturation_c=a.saturation_c)
    if a.cmd == "handlabels":
        sc = score_handlabels(a.candidates, a.labels, clf, tuple(a.positive), bypass)
    else:
        sc = score_manifest(load_manifest(a.manifest, split=a.split or None), clf, bypass)
    bands = default_bands(clf)
    if a.model_bands:
        bands["model"] = bands["pipeline"] = TriageBands(*a.model_bands)
    if a.algo_bands:
        bands["algorithmic"] = TriageBands(*a.algo_bands)
    res = compare(sc, bands, default_search_thresholds(clf))
    res["model"] = Path(a.cnn).name
    label = a.label or f"{sc.source} · {Path(a.cnn).stem}"
    md = markdown(res, label)
    if a.key:
        update_section(a.results_doc, a.key, md, res)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2, default=float))
    print(md)


if __name__ == "__main__":
    _main()
