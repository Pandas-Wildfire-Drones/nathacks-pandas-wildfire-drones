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
GSD: view geometry (agl_m, pitch_deg, focal_px: oblique frames), else the row's gsd_m, else --gsd-m. Object areas are
  in m^2 of GROUND, so one of these is required.
OBLIQUE FRAMES (all FLAME 3 Sycan frames: pitch -10..-41 deg for the No-Fire backgrounds). Per object at row v:
  * size uses the cross-track GSD of that row, and the ellipse is FORESHORTENED vertically by sin(depression):
    a 1 m^2 round spot seen at 15 deg depression covers ~4x fewer pixels, squashed into a flat ellipse;
  * objects are only placed on GROUND: rows at >= --min-depression-deg below the horizon and pixels warmer than
    --sky-c (the M30T floors sky at -40.6 / -22.8 C);
  * CLIP (--clip-mode). "camera" (default): every composite clips at --saturation-c, i.e. what OUR seeker camera
    would read (ThermalCameraSpec.saturation_temp_c, [VERIFY] context.md section 12 item 11). Clipping is a property
    of the camera setting, not of the scene, and 108 of the 116 FLAME 3 Sycan No-Fire backgrounds were shot in HIGH
    gain (clip 187.4 C): with "frame" nearly every inserted fire would read 187.4 C. "frame": use the manifest's
    per-frame saturation_c, to mimic FLAME 3's own frames (e.g. to compare with its real high-gain fire frames).
  Assumptions: flat ground (the LRF agrees with pitch x height, but hills break it); no atmospheric loss over the
  100-400 m slant range; emissivity independent of view angle.
  [VERIFY] FLAME 3 backgrounds come from prescribed burns in daytime or at dusk (15:00-19:20 local), so they are
  warmer and more sun-heated than our night scenes, and the view is oblique where our seeker looks straight down.
  Report results on the night synthetic test set as well.

Outputs (in --out):
  frames/*.npy, masks/*.npy   composites (float32 C, uint8 mask = pixels with any fire fill, as in perception.synth)
  manifest.csv                composites + (unless --no-backgrounds) the original backgrounds as negatives. The GROUP
                              IS THE BACKGROUND'S GROUP, so a split never puts a background in train and its
                              composite in test.
  objects.csv                 one row per inserted object: true area and core temperature, OBSERVED peak, and band
                              (<80 / 80-200 / >200 C). Use it for per-band F1.

CLI:
  python -m training.composite --manifest exclude/data/flame3/manifest.csv --out exclude/data/flame3_composite \
      --per-background 3 [--decoys] [--seed 0]          # FLAME 3: geometry + gain come from the manifest
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

from perception.datasets import (FrameRecord, depression_rows, frame_label, load_frame, load_manifest, load_mask,
                                 write_manifest)
from perception.synth import SynthObject, SynthParams, _render_object, planck_radiance, radiance_to_temp, random_objects

__all__ = ["BANDS", "band_of", "composite", "sample_fires", "to_pixel_equivalent", "ground_pixels", "build_composites"]

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


def _sample_uv(rng: np.random.Generator, h: int, w: int, margin_px: int, valid: np.ndarray | None) -> tuple[float, float]:
    """Uniform position inside the margins, restricted to `valid` pixels (ground) when given."""
    if valid is None:
        return float(rng.uniform(margin_px, w - margin_px)), float(rng.uniform(margin_px, h - margin_px))
    k = int(rng.integers(0, len(valid)))
    v, u = divmod(int(valid[k]), w)
    return float(u + rng.uniform(0, 1)), float(v + rng.uniform(0, 1))


def sample_fires(rng: np.random.Generator, h: int, w: int, area_range_m2: tuple[float, float] = (0.01, 4.0),
                 core_range_c: tuple[float, float] = (150.0, 700.0), n_range: tuple[int, int] = (1, 3),
                 gsd_m: float = 0.1, margin_px: int = 12, valid: np.ndarray | None = None,
                 gsd_rows: np.ndarray | None = None) -> list[SynthObject]:
    """1-3 fires, log-uniform area (embers to ~4 m^2 spot fires), uniform core temperature (perception.synth
    defaults). Small areas get diluted below the pixel size, so many observed peaks land in 80-200 C.
    Returned objects are in GROUND units (true area/aspect/angle); oblique frames: see to_pixel_equivalent().
    valid: flat indices of allowed pixels (ground); gsd_rows: cross-track GSD per row (else gsd_m everywhere)."""
    fires: list[SynthObject] = []
    radii: list[float] = []
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        area = float(np.exp(rng.uniform(np.log(area_range_m2[0]), np.log(area_range_m2[1]))))
        aspect = float(rng.uniform(1, 2.5))
        for _try in range(20):                                     # keep fires apart (see composite())
            u, v = _sample_uv(rng, h, w, margin_px, valid)
            g = gsd_m if gsd_rows is None else float(gsd_rows[min(int(v), h - 1)])
            r = math.sqrt(area / g ** 2 * aspect / math.pi) + 2   # semi-major axis in px + render margin
            if all(math.hypot(u - f.u, v - f.v) > r + rf + 2 for f, rf in zip(fires, radii)):
                fires.append(SynthObject("fire", u, v, area, float(rng.uniform(*core_range_c)), aspect,
                                         float(rng.uniform(0, 180))))
                radii.append(r)
                break
    return fires


def to_pixel_equivalent(obj: SynthObject, gsd_v: float, sin_dep: float, gsd_ref: float) -> SynthObject:
    """The object that, rendered NADIR at gsd_ref, covers the same pixels as `obj` seen obliquely at a row with
    cross-track GSD gsd_v and depression sin_dep: scale by gsd_v, then squash the image-vertical axis by sin_dep.
    Ellipse x^T Q x <= 1 in pixels; squashing y by s maps Q -> S^-1 Q S^-1 with S = diag(1, s)."""
    if abs(sin_dep - 1.0) < 1e-9 and abs(gsd_v - gsd_ref) < 1e-12:
        return obj
    area_px = obj.area_m2 / gsd_v ** 2
    a = math.sqrt(area_px * obj.aspect / math.pi)
    b = math.sqrt(area_px / (obj.aspect * math.pi))
    th = math.radians(obj.angle_deg)                  # major axis along (cos th, sin th) in (u, v): perception.synth
    e1, e2 = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    q = np.outer(e1, e1) / a ** 2 + np.outer(e2, e2) / b ** 2
    s_inv = np.diag([1.0, 1.0 / max(sin_dep, 1e-3)])
    lam, vec = np.linalg.eigh(s_inv @ q @ s_inv)      # ascending: lam[0] -> major axis
    a2, b2 = 1.0 / math.sqrt(lam[0]), 1.0 / math.sqrt(lam[1])
    ang = math.degrees(math.atan2(vec[1, 0], vec[0, 0])) % 180.0
    return SynthObject(obj.kind, obj.u, obj.v, math.pi * a2 * b2 * gsd_ref ** 2, obj.temp_c, a2 / b2, ang)


def ground_pixels(bg_c: np.ndarray, rec: FrameRecord, margin_px: int = 12, min_depression_deg: float = 10.0,
                  sky_c: float = -35.0) -> np.ndarray:
    """Flat indices of pixels where an object may be placed: inside the margins, not sky, and (oblique frames)
    at least min_depression_deg below the horizon."""
    h, w = bg_c.shape
    ok = np.zeros((h, w), dtype=bool)
    ok[margin_px:h - margin_px, margin_px:w - margin_px] = True
    ok &= bg_c > sky_c
    if rec.has_geometry:
        dep = np.degrees(depression_rows(h, rec.pitch_deg, rec.focal_px))
        ok &= (dep >= min_depression_deg)[:, None]
    return np.flatnonzero(ok)


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
                     psf_sigma_px: float = 0.6, saturation_c: float = 550.0, seed: int = 0,
                     min_depression_deg: float = 10.0, sky_c: float = -35.0, min_ground_frac: float = 0.05,
                     clip_mode: str = "camera") -> dict:
    """clip_mode "camera": clip at saturation_c; "frame": the row's saturation_c (gain mode), else saturation_c."""
    if clip_mode not in ("camera", "frame"):
        raise ValueError(f"clip_mode must be 'camera' or 'frame', not {clip_mode!r}")
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
        if not rec.has_gsd and gsd_m is None:
            raise ValueError(f"{rec.frame_path}: no gsd_m/view geometry in the manifest; pass --gsd-m "
                             "(object areas are in m^2)")
        if not rec.has_gsd:
            rec = replace(rec, gsd_m=gsd_m)
        bg, why = _usable_background(rec, max_bg_c)
        if bg is not None:
            h, w = bg.shape
            valid = ground_pixels(bg, rec, 12, min_depression_deg, sky_c)
            if len(valid) < min_ground_frac * h * w:
                bg, why = None, "too_little_ground"
        reasons[why] += 1
        if bg is None:
            continue
        if keep_backgrounds:
            recs_out.append(replace(rec, split=""))
        gsd_rows, sin_rows = rec.gsd_rows(h)
        gsd_ref = float(rec.gsd_m if rec.gsd_m is not None else np.median(gsd_rows))
        sat = rec.saturation_c if clip_mode == "frame" and rec.saturation_c is not None else saturation_c
        p = SynthParams(height_px=h, width_px=w, gsd_m=gsd_ref)
        stem = Path(rec.frame_path).stem
        for k in range(per_background):
            objs = sample_fires(rng, h, w, area_range_m2, core_range_c, gsd_m=gsd_ref, valid=valid,
                                gsd_rows=gsd_rows) if rng.random() < fire_prob else []
            if decoys:   # rocks / vehicles / animals only, moved onto the ground
                for o in random_objects(rng, p, fire_prob=0.0):
                    u, v = _sample_uv(rng, h, w, 12, valid)
                    objs.append(replace(o, u=u, v=v))
            row = [min(int(o.v), h - 1) for o in objs]
            drawn = [to_pixel_equivalent(o, float(gsd_rows[r]), float(sin_rows[r]), gsd_ref) for o, r in zip(objs, row)]
            temp, mask, peaks = composite(bg, drawn, gsd_ref, psf_sigma_px, sat)
            name = f"{idx:05d}_{stem}_c{k}"
            fp, mp = out / "frames" / f"{name}.npy", out / "masks" / f"{name}.npy"
            np.save(fp, temp)
            np.save(mp, mask.astype(np.uint8))
            recs_out.append(replace(rec, frame_path=str(fp), mask_path=str(mp), label=int(mask.any()), split="",
                                    source=f"composite:{rec.source or 'real'}", rgb_path="", saturation_c=sat))
            for o, pk, r in zip(objs, peaks, row):
                b = band_of(pk) if o.is_fire else ""
                if o.is_fire:
                    bands[b] += 1
                obj_rows.append({"frame_path": f"frames/{name}.npy", "group": rec.group, "kind": o.kind,
                                 "u": round(o.u, 2), "v": round(o.v, 2), "area_m2": round(o.area_m2, 5),
                                 "core_temp_c": round(o.temp_c, 1), "observed_peak_c": round(pk, 2), "band": b,
                                 "gsd_m": round(float(gsd_rows[r]), 4), "sin_depression": round(float(sin_rows[r]), 4),
                                 "saturation_c": sat})

    if reasons["used"] == 0:
        raise ValueError(f"no usable backgrounds in {manifest}: {dict(reasons)}")
    write_manifest(out / "manifest.csv", recs_out)
    cols = ["frame_path", "group", "kind", "u", "v", "area_m2", "core_temp_c", "observed_peak_c", "band", "gsd_m",
            "sin_depression", "saturation_c"]
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
    ap.add_argument("--saturation-c", type=float, default=550.0,
                    help="clip for rows without saturation_c in the manifest ([VERIFY] the camera's gain mode)")
    ap.add_argument("--min-depression-deg", type=float, default=10.0, help="oblique frames: no objects nearer the horizon")
    ap.add_argument("--sky-c", type=float, default=-35.0, help="pixels at/below this are sky: no objects there")
    ap.add_argument("--clip-mode", choices=["camera", "frame"], default="camera",
                    help="camera: clip at --saturation-c; frame: each background's own gain clip (FLAME 3 high gain 187.4 C)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    s = build_composites(a.manifest, a.out, a.per_background, a.gsd_m, a.max_bg_c, a.decoys, a.fire_prob,
                         not a.no_backgrounds, tuple(a.area_range), tuple(a.core_range), a.psf_sigma_px,
                         a.saturation_c, a.seed, a.min_depression_deg, a.sky_c, clip_mode=a.clip_mode)
    for k, v in s.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
