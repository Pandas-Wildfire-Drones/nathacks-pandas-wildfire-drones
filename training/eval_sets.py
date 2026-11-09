"""The datasets behind the "mix" model and the six test views in training/compare_approaches.ipynb.

TRAINING (split == "train" rows only, merged by `write_mix_manifest`):
  nadir      FLAME 3 NADIR plot 2 (both parts), history-labelled masks           training/build_nadir.py
  composite  synthetic fires + decoys on real Sycan No-Fire frames               training/composite.py
  cvsubset   Sycan CV subset, weak masks: clipped/>=200 C pixels in Fire frames = fire, No-Fire frames = not fire,
             everything else ignored; frames holding a hand-labelled candidate are never trained on
                                                                                 training/build_sycan_weak.py
TEST VIEWS (never trained on):
  nadir           NADIR plot 1 (a different plot of the same burn), history labels, POINT-level hard view: per frame
                  20 warm not-fire points (char, warm ground) + 20 fire points (half residual heat), sampled from the
                  masks (training/compare.py::score_points). Proposal-level NADIR is ~99 % fire (the proposer keeps
                  the 64 hottest blobs), so it only measures recall: that's view `nadir_proposals`.
  handlabels      your RGB-judged Sycan blobs, 80-200 C ('unsure' dropped)
  composite       composite test split (4 held-out ground cells)
  cvsubset        Sycan test groups with the weak masks above
  all_with_hand   nadir + composite + cvsubset + handlabels, pooled
  all_no_hand     nadir + composite + cvsubset, pooled
Groups are prefixed with the view name when pooling, so the group bootstrap never merges two datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from perception.datasets import FrameRecord, load_manifest, write_manifest
from perception.detectors import HotBypassRule

from .compare import ScoredCandidates, score_handlabels, score_manifest, score_points

__all__ = ["Paths", "SINGLE_VIEWS", "EXTRA_VIEWS", "POOLED_VIEWS", "write_mix_manifest", "test_records", "concat", "score_views"]

SINGLE_VIEWS = ("nadir", "handlabels", "composite", "cvsubset")
EXTRA_VIEWS = ("nadir_proposals",)
POOLED_VIEWS = {"all_with_hand": ("nadir", "composite", "cvsubset", "handlabels"),
                "all_no_hand": ("nadir", "composite", "cvsubset")}


@dataclass(frozen=True)
class Paths:
    nadir: str = "exclude/data/flame3_nadir_prepared/manifest.csv"
    composite: str = "exclude/data/flame3_composite/manifest_split.csv"
    cvsubset: str = "exclude/data/flame3_cvsubset_weak/manifest_split.csv"
    candidates: str = "exclude/labels/candidates.csv"
    labels: str = "exclude/labels/labels_DoqCA.csv"
    mix: str = "exclude/data/mix_v1/manifest_split.csv"
    extra: dict = field(default_factory=dict)


def write_mix_manifest(paths: Paths = Paths(), out: str | None = None, sources=("nadir", "composite", "cvsubset")
                       ) -> list[FrameRecord]:
    """One manifest with every row of the chosen sources (train AND test rows; training reads split == 'train').
    A frame listed by two sources (Sycan No-Fire frames are composite backgrounds AND cvsubset negatives) is kept once."""
    seen, recs = set(), []
    for s in sources:
        for r in load_manifest(getattr(paths, s)):
            key = str(Path(r.frame_path).resolve())
            if key not in seen:
                seen.add(key)
                recs.append(r)
    write_manifest(out or paths.mix, recs)
    return recs


def test_records(view: str, paths: Paths = Paths(), every: int = 1) -> list[FrameRecord]:
    recs = load_manifest(getattr(paths, "nadir" if view == "nadir_proposals" else view), split="test")
    return recs[::every]


def concat(parts: dict[str, ScoredCandidates]) -> ScoredCandidates:
    """Pool several views; groups get a 'view:' prefix. A blob already present in an earlier view (same frame and
    pixel) is dropped from later ones, so shared Sycan No-Fire frames and hand-labelled blobs count once."""
    seen: set = set()
    items = []
    for name, sc in parts.items():
        keep = []
        for i in range(len(sc)):
            key = (str(Path(sc.frame_path[i]).resolve()), int(round(sc.u[i])), int(round(sc.v[i])))
            keep.append(key not in seen)
            seen.add(key)
        items.append((name, sc if all(keep) else sc.subset(np.array(keep))))
    first = items[0][1]

    def cat(attr):
        vals = [getattr(sc, attr) for _, sc in items]
        return np.concatenate(vals) if isinstance(vals[0], np.ndarray) else [x for v in vals for x in v]

    group = np.concatenate([np.array([f"{name}:{g}" for g in sc.group]) for name, sc in items])
    label_text = [f"{name}:{t}" for name, sc in items for t in sc.label_text]
    out = ScoredCandidates(
        "+".join(n for n, _ in items), cat("y"), cat("peak_c"), cat("area_px"), cat("u"), cat("v"), cat("p_model"),
        cat("bypass"), group, cat("frame_path"), cat("gsd_m"), cat("saturation_c"), label_text,
        [p for _, sc in items for p in (sc.rgb_path or [""] * len(sc))],
        [b for _, sc in items for b in (sc.rgb_box or [(0.0, 0.0, 1.0, 1.0)] * len(sc))],
        bypass_confidence=first.bypass_confidence,
        n_objects=sum(sc.n_objects for sc in parts.values()), missed_objects=sum(sc.missed_objects for sc in parts.values()))
    out.view = np.concatenate([np.full(len(sc), name) for name, sc in items])
    return out


def score_views(classifier, views=SINGLE_VIEWS + tuple(POOLED_VIEWS), paths: Paths = Paths(),
                bypass: HotBypassRule = HotBypassRule(), positive=("active", "smoulder"), every: int = 1,
                cache: dict | None = None) -> dict[str, ScoredCandidates]:
    """ScoredCandidates per requested view (single views are scored once and reused by the pooled ones)."""
    cache = {} if cache is None else cache
    need = {v for v in views if v in SINGLE_VIEWS + EXTRA_VIEWS}
    for v in views:
        need.update(POOLED_VIEWS.get(v, ()))
    for v in sorted(need):
        if v in cache:
            continue
        if v == "handlabels":
            if not (Path(paths.candidates).exists() and Path(paths.labels).exists()):
                continue
            cache[v] = score_handlabels(paths.candidates, paths.labels, classifier, positive, bypass)
        else:
            if not Path(getattr(paths, "nadir" if v == "nadir_proposals" else v)).exists():
                continue
            recs = test_records(v, paths, every)
            cache[v] = (score_points(recs, classifier, bypass=bypass) if v == "nadir"
                        else score_manifest(recs, classifier, bypass))
            cache[v].source = v
    out = {}
    for v in views:
        if v in POOLED_VIEWS:
            parts = {p: cache[p] for p in POOLED_VIEWS[v] if p in cache}
            if parts:
                out[v] = concat(parts)
        elif v in cache:
            out[v] = cache[v]
    return out


def main() -> None:
    """python -m training.eval_sets --model "name=path.onnx" [--model ...] [--key K]  -> matrix in perception/RESULTS.md"""
    import argparse
    import json

    from perception.results_doc import update_section

    from .compare import compare, default_bands, default_search_thresholds
    from .compare_plots import matrix_markdown
    from perception.detectors import load_classifier

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name=path.onnx (repeatable)")
    ap.add_argument("--views", nargs="+", default=list(SINGLE_VIEWS + tuple(POOLED_VIEWS) + EXTRA_VIEWS))
    ap.add_argument("--labels", default=Paths().labels)
    ap.add_argument("--key", help="RESULTS.md section key")
    ap.add_argument("--results-doc", default=str(Path(__file__).resolve().parents[1] / "perception" / "RESULTS.md"))
    ap.add_argument("--json-out")
    a = ap.parse_args()
    paths = Paths(labels=a.labels)
    results = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        clf = load_classifier(path)
        views = score_views(clf, tuple(a.views), paths)
        results[name] = {v: compare(sc, default_bands(clf), default_search_thresholds(clf), n_boot=500)
                         for v, sc in views.items()}
    md = "### Models × test views (three approaches)\n\n" + matrix_markdown(results)
    if a.key:
        update_section(a.results_doc, a.key, md, results)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(results, indent=2, default=float))
    print(md)


if __name__ == "__main__":
    main()
