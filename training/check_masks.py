"""Audit a dataset's fire masks: were they made by thresholding the temperature image?

If mask == (temperature >= T) for some T, then scoring a "threshold + CNN" detector against those masks is
partly circular: the labels ARE a threshold, so they reward reproducing it, and they label every warm decoy
above T as "fire". Run this on FLAME 3 before trusting its masks for evaluation.

    python -m training.check_masks --manifest exclude/data/flame3/manifest.csv --out exclude/runs/mask_audit \
        [--overlays 40] [--rgb-dir exclude/data/flame3_raw/rgb_corrected] [--tag flame3]

Per frame (written to per_frame.csv):
  best_iou       IoU between the mask and (T >= t) at the best single threshold t. ~1.0 => mask is a threshold.
  best_t_c       that threshold.
  gap_c          min(T inside mask) - max(T outside mask). > 0 => a threshold separates them perfectly.
  hole_frac      area of holes inside the mask / mask area. Threshold masks have holes; hand-drawn polygons don't.
  min_in_c       coldest masked pixel (hand masks often include cool pixels at fire edges; threshold masks can't).
Summary verdict:
  THRESHOLD-GENERATED   most frames best_iou >= 0.98 and best_t nearly constant -> one global threshold
  PER-FRAME THRESHOLD   most frames best_iou >= 0.98 but best_t varies -> per-frame threshold (e.g. Otsu)
  NOT A PLAIN THRESHOLD best_iou typically < 0.9 -> likely hand-drawn/corrected; still eyeball the overlays
Overlays (--overlays N): thermal in grey, mask outline in red, RGB alongside if --rgb-dir has the same file stem.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage

from perception.datasets import load_frame, load_manifest, load_mask
from perception.results_doc import update_section

REPO = Path(__file__).resolve().parents[1]


def threshold_fit(temp: np.ndarray, mask: np.ndarray, n_candidates: int = 256) -> dict:
    """Best IoU of mask vs (temp >= t) over candidate t, computed with sorted arrays (fast)."""
    t_all = np.sort(temp.ravel())
    t_in = np.sort(temp[mask])
    n_mask = len(t_in)
    cand = np.unique(np.quantile(t_in, np.linspace(0, 1, n_candidates)))
    above_all = len(t_all) - np.searchsorted(t_all, cand, side="left")
    above_in = n_mask - np.searchsorted(t_in, cand, side="left")
    iou = above_in / (n_mask + above_all - above_in)
    i = int(np.argmax(iou))
    out_max = float(temp[~mask].max()) if (~mask).any() else float("-inf")
    filled = ndimage.binary_fill_holes(mask)
    return {"best_iou": float(iou[i]), "best_t_c": float(cand[i]), "gap_c": float(t_in[0] - out_max),
            "max_out_c": out_max,
            "hole_frac": float((filled.sum() - mask.sum()) / max(mask.sum(), 1)), "min_in_c": float(t_in[0]),
            "mask_px": int(n_mask)}


def global_threshold_consistency(rows: list[dict]) -> tuple[float, float]:
    """Is ONE temperature T (same for every frame) consistent with the masks? Frame i agrees with T when
    max(T outside mask) < T <= min(T inside mask). Returns the T agreeing with the most frames, and that fraction."""
    lo = np.array([r["max_out_c"] for r in rows])
    hi = np.array([r["min_in_c"] for r in rows])
    cand = np.unique(hi)
    best_t, best_n = float(cand[0]), -1
    for i in range(0, len(cand), 512):  # chunked broadcast: fine for tens of thousands of frames
        c = cand[i:i + 512][:, None]
        n = ((lo[None, :] < c) & (c <= hi[None, :])).sum(1)
        j = int(np.argmax(n))
        if n[j] > best_n:
            best_t, best_n = float(c[j, 0]), int(n[j])
    return best_t, best_n / len(rows)


def verdict(rows: list[dict]) -> str:
    """High IoU alone is NOT proof: a small, hot, sharp-edged fire is naturally separable by temperature.
    The fingerprint of a threshold-made mask is the SAME cutoff in every frame (or a perfect cutoff in every frame)."""
    if not rows:
        return "NO MASKED FRAMES"
    ious = np.array([r["best_iou"] for r in rows])
    t_star, frac_global = global_threshold_consistency(rows)
    if frac_global >= 0.9:
        return f"THRESHOLD-GENERATED (one global cutoff ~{t_star:.1f} C)"
    if float((ious >= 0.98).mean()) >= 0.9:
        return "PER-FRAME THRESHOLD LIKELY (e.g. Otsu per image), or exceptionally sharp fires: check overlays"
    if np.median(ious) >= 0.9:
        return "INCONCLUSIVE: temperature-separable but no constant cutoff; check overlays against RGB"
    return "NOT A PLAIN THRESHOLD"


def _overlay(temp, mask, path: Path, rgb_path: Path | None) -> None:
    from PIL import Image  # lazy

    lo, hi = np.percentile(temp, [1, 99.5])
    g = (np.clip((temp - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    img = np.stack([g, g, g], -1)
    edge = mask & ~ndimage.binary_erosion(mask)
    img[edge] = (255, 0, 0)
    out = Image.fromarray(img)
    if rgb_path is not None and rgb_path.exists():
        rgb = Image.open(rgb_path).convert("RGB").resize(out.size)
        both = Image.new("RGB", (out.width * 2, out.height))
        both.paste(out, (0, 0))
        both.paste(rgb, (out.width, 0))
        out = both
    out.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlays", type=int, default=0, help="save N random overlay PNGs")
    ap.add_argument("--rgb-dir", help="folder of RGB images with the same file stem as the thermal frames")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-doc", default=str(REPO / "perception" / "RESULTS.md"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = [r for r in load_manifest(a.manifest) if r.has_mask]
    rows = []
    for r in recs:
        temp, mask = load_frame(r), load_mask(r)
        if mask is None or not mask.any():
            continue
        rows.append({"frame": Path(r.frame_path).name, "group": r.group, **threshold_fit(temp, mask)})
    if not rows:
        raise SystemExit("no frames with non-empty masks in the manifest")
    with (out / "per_frame.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.overlays:
        rng = np.random.default_rng(a.seed)
        rgb_dir = Path(a.rgb_dir) if a.rgb_dir else None
        (out / "overlays").mkdir(exist_ok=True)
        masked = [r for r in recs if load_mask(r).any()]
        for r in rng.choice(masked, size=min(a.overlays, len(masked)), replace=False):
            stem = Path(r.frame_path).stem
            rgb = next(iter(rgb_dir.glob(f"{stem}.*")), None) if rgb_dir else None
            _overlay(load_frame(r), load_mask(r), out / "overlays" / f"{stem}.png", rgb)

    ious = np.array([x["best_iou"] for x in rows])
    ts = np.array([x["best_t_c"] for x in rows])
    v = verdict(rows)
    t_star, frac_global = global_threshold_consistency(rows)
    summary = {
        "verdict": v, "global_cutoff_c": t_star, "frac_frames_consistent_with_global_cutoff": frac_global, "frames": len(rows), "frac_iou_ge_0.98": float((ious >= 0.98).mean()),
        "median_best_iou": float(np.median(ious)), "best_t_c_p10_p50_p90": [float(x) for x in np.percentile(ts, [10, 50, 90])],
        "frac_perfectly_separable": float(np.mean([x["gap_c"] > 0 for x in rows])),
        "median_hole_frac": float(np.median([x["hole_frac"] for x in rows])),
        "median_min_in_c": float(np.median([x["min_in_c"] for x in rows])),
    }
    md = "\n".join([
        f"### Mask audit · {a.tag or Path(a.manifest).parent.name}", "",
        f"**Verdict: {v}** over {len(rows)} masked frames.", "",
        "| check | value | threshold-made masks look like |", "|---|---|---|",
        f"| **frames consistent with ONE cutoff ({t_star:.1f} °C) for all frames** | {frac_global:.1%} | ~100% (key test) |",
        f"| frames where one threshold reproduces the mask (IoU ≥ 0.98) | {summary['frac_iou_ge_0.98']:.1%} | ~100% |",
        f"| median best-threshold IoU | {summary['median_best_iou']:.3f} | ~1.0 |",
        f"| best threshold p10 / p50 / p90 (°C) | {' / '.join(f'{x:.1f}' for x in summary['best_t_c_p10_p50_p90'])} | constant if global |",
        f"| frames perfectly separable by temperature | {summary['frac_perfectly_separable']:.1%} | ~100% |",
        f"| median hole fraction | {summary['median_hole_frac']:.3f} | > 0 (hand polygons ≈ 0) |",
        f"| median coldest masked pixel (°C) | {summary['median_min_in_c']:.1f} | = the threshold |",
        "", "If threshold-made: hand-check the TEST split masks against RGB, and report frame-level results from "
        "human Fire/No-Fire labels too (exclude/DATA_SOURCES.md).",
    ])
    update_section(a.results_doc, f"mask_audit_{a.tag or Path(a.manifest).parent.name}", md, summary)
    print(md)
    print(f"per-frame: {out / 'per_frame.csv'}")


if __name__ == "__main__":
    main()
