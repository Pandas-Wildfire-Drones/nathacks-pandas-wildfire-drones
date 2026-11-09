"""FLAME 3 Sycan CV subset -> weak pixel masks that only say what the dataset's own labels make certain.

FLAME 3 labels whole frames with a max-temperature rule (< ~80 C no fire, > ~200 C fire; context.md section 10).
At blob level that supports exactly two statements, so the mask says only those:
  No-Fire frame              every pixel 0 (not fire): real negatives, incl. sun-warmed rocks/logs the proposer finds
  Fire frame, T >= hot_c     1 (fire). hot_c = min(200 C, gain-mode clip - 0.5 C): high-gain frames clip at 187.4 C,
                             so a clipped pixel is "at least 187 C", certainly burning
  Fire frame, everything else  2 (ignore): the 80-200 C band is exactly what nobody can label from temperature;
                             that's what the hand labels (training/label_tool.py) are for
Split (by group = day x 250 m cell): No-Fire groups keep the composite split (their frames are composite
backgrounds, so they must stay on the same side); fire-only groups get their own seeded group split. Frames holding a
hand-labelled candidate are marked split="holdout" if their group is in train, so they're never trained on.

    python -m training.build_sycan_weak --manifest exclude/data/flame3/manifest.csv \
        --composite exclude/data/flame3_composite/manifest_split.csv --candidates exclude/labels/candidates.csv \
        --out exclude/data/flame3_cvsubset_weak
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from perception.datasets import MASK_IGNORE, FrameRecord, load_frame, load_manifest, write_manifest

__all__ = ["weak_mask", "assign_splits", "build"]


def weak_mask(temp: np.ndarray, is_fire_frame: bool, saturation_c: float | None, hot_c: float = 200.0) -> np.ndarray:
    if not is_fire_frame:
        return np.zeros(temp.shape, np.uint8)
    thr = min(hot_c, saturation_c - 0.5) if saturation_c else hot_c
    m = np.full(temp.shape, MASK_IGNORE, np.uint8)
    m[temp >= thr] = 1
    return m


def assign_splits(records: list[FrameRecord], composite_split: dict[str, str], test_frac: float = 0.25,
                  seed: int = 0) -> dict[str, str]:
    groups = sorted({r.group for r in records})
    out = {g: composite_split[g] for g in groups if g in composite_split}
    rest = [g for g in groups if g not in out]
    rng = np.random.default_rng(seed)
    rng.shuffle(rest)
    k = max(1, int(round(test_frac * len(rest)))) if rest else 0
    out.update({g: ("test" if i < k else "train") for i, g in enumerate(rest)})
    return out


def build(manifest: str | Path, composite: str | Path, candidates: str | Path | None, out: str | Path,
          hot_c: float = 200.0, test_frac: float = 0.25, seed: int = 0) -> list[FrameRecord]:
    out = Path(out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    recs = load_manifest(manifest)
    comp = {}
    for r in load_manifest(composite):
        comp.setdefault(r.group, r.split)
    splits = assign_splits(recs, comp, test_frac, seed)
    held = set()
    if candidates and Path(candidates).exists():
        with open(candidates, newline="") as fh:
            held = {Path(row["frame_path"]).resolve() for row in csv.DictReader(fh)}
    new = []
    for r in recs:
        temp = load_frame(r)
        m = weak_mask(temp, bool(r.label), r.saturation_c, hot_c)
        stem = f"{'fire' if r.label else 'nofire'}_{Path(r.frame_path).stem}"
        np.save(out / "masks" / f"{stem}.npy", m)
        split = splits[r.group]
        if split == "train" and Path(r.frame_path).resolve() in held:
            split = "holdout"
        new.append(replace(r, mask_path=str(out / "masks" / f"{stem}.npy"), split=split, source="flame3_sycan_weak"))
    write_manifest(out / "manifest_split.csv", new)
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="exclude/data/flame3/manifest.csv")
    ap.add_argument("--composite", default="exclude/data/flame3_composite/manifest_split.csv")
    ap.add_argument("--candidates", default="exclude/labels/candidates.csv")
    ap.add_argument("--out", default="exclude/data/flame3_cvsubset_weak")
    ap.add_argument("--hot-c", type=float, default=200.0)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    recs = build(a.manifest, a.composite, a.candidates, a.out, a.hot_c, a.test_frac, a.seed)
    from collections import Counter
    print(Counter((r.split, r.label) for r in recs), "->", Path(a.out) / "manifest_split.csv")


if __name__ == "__main__":
    main()
