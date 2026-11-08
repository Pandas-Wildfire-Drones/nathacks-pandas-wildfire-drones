"""Composite training data: synthetic fires (and optional decoys) inserted into REAL No-Fire radiometric frames.

Why (training/FLAME3.md): FLAME 3's labels come from a max-temperature heuristic. Fire frames don't say which blob is
the fire, and the 80-200 C band, where the CNN verifier matters most, is under-represented. The trustworthy part is
the No-Fire frames (max < ~80 C): real terrain, real sensor noise, real clutter. Inserting fires with known geometry
gives EXACT masks at any size and temperature we choose, including sub-pixel smouldering spots whose observed peak
lands in 80-200 C. Because the backgrounds are real, the CNN can't win by learning our synthetic-background quirks.

Physics: the same renderer as perception.synth (sub-pixel fill, hot core / cooler rim). In each pixel the object
replaces fraction f of the background radiance, L = L_bg (1 - f) + L_obj. That change is blurred with the PSF (the
real frame is already blurred, so only the inserted part needs it) and converted back to C. Pixels the object
doesn't reach stay bit-identical to the real frame.

Inputs: a manifest (perception.datasets format). Only rows with label 0, an empty or absent mask, finite values and
max(T) < --max-bg-c are used as backgrounds.
GSD: per-row gsd_m, else --gsd-m (one of the two is required: object areas are in m^2).
  [ASSUMPTION] nadir view: on oblique FLAME 3 frames the GSD varies across the image, so object sizes are approximate.
  [VERIFY] FLAME 3 backgrounds come from prescribed burns, many in daytime or at dusk, so they are warmer and more
  sun-heated than our night scenes. Report results on the night synthetic test set as well.

Outputs (in --out):
  frames/*.npy, masks/*.npy   composites (float32 C, uint8 mask = pixels with any fire fill, as in perception.synth)
  manifest.csv                composites + (unless --no-backgrounds) the original backgrounds as negatives. The GROUP
                              IS THE BACKGROUND'S GROUP, so a split never puts a background in train and its
                              composite in test.
  objects.csv                 one row per inserted object: true area and core temperature, OBSERVED peak, and band
                              (<80 / 80-200 / >200 C). Use it for per-band F1.

CLI:
  python -m training.composite --manifest exclude/data/flame3/manifest.csv --out exclude/data/flame3_composite \
      --per-background 3 --gsd-m 0.2 [--decoys] [--seed 0]
  then: python -m training.make_split --manifest exclude/data/flame3_composite/manifest.csv, and train as usual.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import ndimage

from perception.datasets import FrameRecord, frame_label, load_frame, load_manifest, load_mask, write_manifest
from perception.synth import SynthObject, SynthParams, _render_object, planck_radiance, radiance_to_temp, random_objects

__all__ = ["BANDS", "band_of", "composite", "sample_fires", "build_composites"]

# Max-temperature bands from the FLAME 3 labeling heuristic (Hopkins et al. 2024, sec III.A.1).
BANDS = (("<80", -math.inf, 80.0), ("80-200", 80.0, 200.0), (">200", 200.0, math.inf))


def band_of(temp_c: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= temp_c < hi:
            return name
    return "nan"


def composite(bg_c: np.ndarray, objects: list[SynthObject], gsd_m: float, psf_sigma_px: float = 0.6,
              saturation_c: float = 550.0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Insert `objects` into a real radiometric frame.

    Returns (temp_c float32 HxW, fire_mask bool HxW, observed peak C per object, NaN if it fell off the frame).
    Overlapping objects are not handled exactly (each replaces a fraction of the BACKGROUND), so keep them apart.
    """
    h, w = bg_c.shape
    p = SynthParams(height_px=h, width_px=w, gsd_m=gsd_m, psf_sigma_px=psf_sigma_px, saturation_c=saturation_c)
    bg = bg_c.astype(np.float64)
    rad_bg = planck_radiance(bg)
    delta = np.zeros_like(rad_bg)
    mask = np.zeros((h, w), dtype=bool)
    windows: list[tuple[slice, slice, np.ndarray] | None] = []
    for obj in objects:
        r0, c0, fill, rad_in = _render_object(obj, p)
        n = fill.shape[0]
        rs, cs = slice(max(r0, 0), min(r0 + n, h)), slice(max(c0, 0), min(c0 + n, w))
        if rs.start >= rs.stop or cs.start >= cs.stop:
            windows.append(None)
            continue
        f = fill[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0]
        ri = rad_in[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0]
        delta[rs, cs] += ri - f * rad_bg[rs, cs]   # L_bg (1 - f) + L_obj, written as a change to L_bg
        if obj.is_fire:
            mask[rs, cs] |= f > 0
        windows.append((rs, cs, f > 0))

    touched = delta != 0
    if psf_sigma_px > 0 and touched.any():
        delta = ndimage.gaussian_filter(delta, psf_sigma_px)            # scipy default truncate = 4 sigma
        touched = ndimage.binary_dilation(touched, iterations=int(4.0 * psf_sigma_px + 0.5))
    out = bg.copy()
    out[touched] = np.clip(radiance_to_temp(rad_bg[touched] + delta[touched]), -40.0, saturation_c)

    peaks: list[float] = []
    for win in windows:
        if win is None or not win[2].any():
            peaks.append(math.nan)
        else:
            rs, cs, footprint = win
            peaks.append(float(out[rs, cs][footprint].max()))
    return out.astype(np.float32), mask, peaks


def sample_fires(rng: np.random.Generator, h: int, w: int, area_range_m2: tuple[float, float] = (0.01, 4.0),
                 core_range_c: tuple[float, float] = (150.0, 700.0), n_range: tuple[int, int] = (1, 3),
                 gsd_m: float = 0.1, margin_px: int = 12) -> list[SynthObject]:
    """1-3 fires, log-uniform area (embers to ~4 m^2 spot fires), uniform core temperature (perception.synth
    defaults). Small areas get diluted below the pixel size, so many observed peaks land in 80-200 C."""
    fires: list[SynthObject] = []
    radii: list[float] = []
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        area = float(np.exp(rng.uniform(np.log(area_range_m2[0]), np.log(area_range_m2[1]))))
        aspect = float(rng.uniform(1, 2.5))
        r = math.sqrt(area / gsd_m ** 2 * aspect / math.pi) + 2   # semi-major axis in px + render margin
        for _try in range(20):                                     # keep fires apart (see composite())
            u, v = float(rng.uniform(margin_px, w - margin_px)), float(rng.uniform(margin_px, h - margin_px))
            if all(math.hypot(u - f.u, v - f.v) > r + rf + 2 for f, rf in zip(fires, radii)):
                fires.append(SynthObject("fire", u, v, area, float(rng.uniform(*core_range_c)), aspect,
                                         float(rng.uniform(0, 180))))
                radii.append(r)
                break
    return fires


def _usable_background(rec: FrameRecord, max_bg_c: float) -> tuple[np.ndarray | None, str]:
    if frame_label(rec) != 0:
        return None, "labelled_fire"
    mask = load_mask(rec)
    if mask is not None and mask.any():
        return None, "mask_not_empty"
    temp = load_frame(rec)
    if not np.isfinite(temp).all():
        return None, "non_finite"
    if float(temp.max()) >= max_bg_c:
        return None, "too_hot"
    return temp, "used"


def build_composites(manifest: str | Path, out_dir: str | Path, per_background: int = 3, gsd_m: float | None = None,
                     max_bg_c: float = 80.0, decoys: bool = False, fire_prob: float = 0.8, keep_backgrounds: bool = True,
                     area_range_m2: tuple[float, float] = (0.01, 4.0), core_range_c: tuple[float, float] = (150.0, 700.0),
                     psf_sigma_px: float = 0.6, saturation_c: float = 550.0, seed: int = 0) -> dict:
    """Write composites + manifest.csv + objects.csv into `out_dir`. Returns a summary dict (also printed by the CLI)."""
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    if not decoys:
        fire_prob = 1.0   # a composite with nothing inserted would just duplicate its background

    recs_out: list[FrameRecord] = []
    obj_rows: list[dict] = []
    reasons: Counter = Counter()
    bands: Counter = Counter()
    for idx, rec in enumerate(load_manifest(manifest)):
        gsd = rec.gsd_m if rec.gsd_m is not None else gsd_m
        if gsd is None:
            raise ValueError(f"{rec.frame_path}: no gsd_m in the manifest; pass --gsd-m (object areas are in m^2)")
        bg, why = _usable_background(rec, max_bg_c)
        reasons[why] += 1
        if bg is None:
            continue
        if keep_backgrounds:
            recs_out.append(replace(rec, gsd_m=gsd, split=""))
        h, w = bg.shape
        p = SynthParams(height_px=h, width_px=w, gsd_m=gsd)
        stem = Path(rec.frame_path).stem
        for k in range(per_background):
            objs = sample_fires(rng, h, w, area_range_m2, core_range_c, gsd_m=gsd) if rng.random() < fire_prob else []
            if decoys:
                objs += random_objects(rng, p, fire_prob=0.0)   # rocks / vehicles / animals only
            temp, mask, peaks = composite(bg, objs, gsd, psf_sigma_px, saturation_c)
            name = f"{idx:05d}_{stem}_c{k}"
            fp, mp = out / "frames" / f"{name}.npy", out / "masks" / f"{name}.npy"
            np.save(fp, temp)
            np.save(mp, mask.astype(np.uint8))
            recs_out.append(FrameRecord(frame_path=str(fp), group=rec.group, mask_path=str(mp), label=int(mask.any()),
                                        source=f"composite:{rec.source or 'real'}", gsd_m=gsd))
            for o, pk in zip(objs, peaks):
                b = band_of(pk) if o.is_fire else ""
                if o.is_fire:
                    bands[b] += 1
                obj_rows.append({"frame_path": f"frames/{name}.npy", "group": rec.group, "kind": o.kind,
                                 "u": round(o.u, 2), "v": round(o.v, 2), "area_m2": round(o.area_m2, 5),
                                 "core_temp_c": round(o.temp_c, 1), "observed_peak_c": round(pk, 2), "band": b})

    if reasons["used"] == 0:
        raise ValueError(f"no usable backgrounds in {manifest}: {dict(reasons)}")
    write_manifest(out / "manifest.csv", recs_out)
    cols = ["frame_path", "group", "kind", "u", "v", "area_m2", "core_temp_c", "observed_peak_c", "band"]
    with (out / "objects.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(obj_rows)
    return {"backgrounds": dict(reasons), "composites": reasons["used"] * per_background,
            "frames_in_manifest": len(recs_out), "fires_by_observed_band": dict(bands),
            "manifest": str(out / "manifest.csv")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="input manifest (e.g. FLAME 3 from build_manifest_flame3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-background", type=int, default=3)
    ap.add_argument("--gsd-m", type=float, help="metres/pixel for rows without gsd_m")
    ap.add_argument("--max-bg-c", type=float, default=80.0, help="skip backgrounds at/above this max temperature")
    ap.add_argument("--decoys", action="store_true", help="also insert synthetic rocks/vehicles/animals")
    ap.add_argument("--fire-prob", type=float, default=0.8, help="P(composite has fires) when --decoys is on")
    ap.add_argument("--no-backgrounds", action="store_true", help="don't list the original backgrounds as negatives")
    ap.add_argument("--area-range", type=float, nargs=2, default=(0.01, 4.0), metavar=("MIN_M2", "MAX_M2"))
    ap.add_argument("--core-range", type=float, nargs=2, default=(150.0, 700.0), metavar=("MIN_C", "MAX_C"))
    ap.add_argument("--psf-sigma-px", type=float, default=0.6)
    ap.add_argument("--saturation-c", type=float, default=550.0, help="[VERIFY] set to the real camera's gain mode")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    s = build_composites(a.manifest, a.out, a.per_background, a.gsd_m, a.max_bg_c, a.decoys, a.fire_prob,
                         not a.no_backgrounds, tuple(a.area_range), tuple(a.core_range), a.psf_sigma_px,
                         a.saturation_c, a.seed)
    for k, v in s.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
