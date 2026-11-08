"""Offline dataset plumbing: one manifest format for every dataset (FLAME 3, synthetic, anything else).

NOT used on the drone. File readers (tifffile, Pillow) are imported lazily.

Manifest = CSV, one row per frame. Paths are relative to the CSV's folder (or absolute).
    frame_path   radiometric frame in degrees C: .npy (float32) or .tif/.tiff (Celsius TIFF)   REQUIRED
    group        flight / burn / sequence id. Splits and CV folds never cut a group (neighbouring
                 frames leak, context.md section 10).                                         REQUIRED
    mask_path    binary fire mask, same HxW (.npy / .png / .tif). Empty = no mask.            optional
    label        1 = frame contains fire, 0 = not. Derived from the mask if empty.            optional*
    split        "train" or "test" (written by training/make_split.py). Empty = unassigned.   optional
    source       dataset name, e.g. "flame3", "synth"                                         optional
    gsd_m        ground sample distance, metres/pixel, if known (nadir; image centre if oblique) optional
    --- view geometry for OBLIQUE frames (all three, or none) -------------------------------------------------
    agl_m        camera height above the ground at the image centre, metres                      optional
    pitch_deg    camera pitch vs the horizon: -90 = nadir, -15 = shallow oblique, > 0 = above horizon
    focal_px     focal length in pixels (thermal)
    --- other per-frame metadata ---------------------------------------------------------------------------
    saturation_c clip temperature of the frame's gain mode (e.g. FLAME 3 M30T: 187.4 high / 608.7 low)  optional
    rgb_path     paired visible image (labelling tool only; never fed to the detector)              optional
    timestamp    capture time, seconds (any epoch, only differences matter: ordering, prev/next)   optional
(*) at least one of mask_path or label is required.

GSD with view geometry (FrameRecord.gsd_rows): pixel row v looks down at depression
    dep(v) = -pitch_deg + atan((v + 0.5 - H/2) / focal_px)
and, over FLAT ground, sees it at slant range agl_m / sin(dep). Cross-track GSD = range / focal_px; along-track
(image vertical) GSD is that / sin(dep), i.e. objects are foreshortened vertically by sin(dep). Rows at or above
the horizon are clamped to `min_depression_deg` so every row has a (large) finite GSD.

Add a dataset by writing a tiny adapter that emits this CSV (see training/build_manifest_flame3.py).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

__all__ = ["FrameRecord", "load_manifest", "write_manifest", "load_frame", "load_mask", "load_ignore", "MASK_IGNORE",
           "assign_group_split", "depression_rows"]

# Mask value meaning "unknown": excluded from training labels AND from scoring (neither fire nor background).
# Used where labels come from rules that can't decide (e.g. FLAME 3 NADIR pixels heating ahead of the front).
# Any other non-zero value is fire, so existing 0/1 and 0/255 masks keep their meaning.
MASK_IGNORE = 2


@dataclass(frozen=True)
class FrameRecord:
    frame_path: str
    group: str
    mask_path: str = ""
    label: int | None = None
    split: str = ""
    source: str = ""
    gsd_m: float | None = None
    agl_m: float | None = None
    pitch_deg: float | None = None
    focal_px: float | None = None
    saturation_c: float | None = None
    rgb_path: str = ""
    timestamp: float | None = None

    @property
    def has_mask(self) -> bool:
        return bool(self.mask_path)

    @property
    def has_geometry(self) -> bool:
        return None not in (self.agl_m, self.pitch_deg, self.focal_px)

    @property
    def has_gsd(self) -> bool:
        return self.gsd_m is not None or self.has_geometry

    def gsd_rows(self, height_px: int, min_depression_deg: float = 2.0) -> tuple[np.ndarray, np.ndarray] | None:
        """(cross-track GSD m/px, sin(depression)) per image row, shape (H,) each. None if no GSD is known.
        Without view geometry: constant gsd_m and sin = 1 (nadir)."""
        if self.has_geometry:
            dep = np.maximum(depression_rows(height_px, self.pitch_deg, self.focal_px), math.radians(min_depression_deg))
            sin = np.sin(dep)
            return (self.agl_m / sin / self.focal_px).astype(np.float64), sin
        if self.gsd_m is not None:
            return np.full(height_px, self.gsd_m, dtype=np.float64), np.ones(height_px)
        return None

    def gsd_at(self, v: float, height_px: int) -> float | None:
        """Cross-track GSD at (fractional) row v; the value the scale-normalized patch uses."""
        rows = self.gsd_rows(height_px)
        if rows is None:
            return None
        return float(rows[0][min(max(int(v), 0), height_px - 1)])


def depression_rows(height_px: int, pitch_deg: float, focal_px: float) -> np.ndarray:
    """Depression angle below the horizon (radians) of each row's centre; row 0 = top of the image."""
    off = (np.arange(height_px) + 0.5 - height_px / 2) / focal_px
    return math.radians(-pitch_deg) + np.arctan(off)


def load_manifest(csv_path: str | Path, split: str | None = None) -> list[FrameRecord]:
    csv_path = Path(csv_path)
    root = csv_path.parent
    out = []
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            def res(p: str) -> str:
                return "" if not p else str(p if Path(p).is_absolute() else (root / p))
            def num(k: str) -> float | None:
                x = row.get(k, "")
                return float(x) if x not in ("", None) else None

            lab = row.get("label", "")
            rec = FrameRecord(frame_path=res(row["frame_path"]), group=row["group"], mask_path=res(row.get("mask_path", "")),
                              label=int(lab) if lab not in ("", None) else None, split=row.get("split", "") or "",
                              source=row.get("source", "") or "", gsd_m=num("gsd_m"), agl_m=num("agl_m"),
                              pitch_deg=num("pitch_deg"), focal_px=num("focal_px"), saturation_c=num("saturation_c"),
                              rgb_path=res(row.get("rgb_path", "") or ""), timestamp=num("timestamp"))
            if not rec.has_mask and rec.label is None:
                raise ValueError(f"{rec.frame_path}: needs mask_path or label")
            if split is None or rec.split == split:
                out.append(rec)
    return out


def write_manifest(csv_path: str | Path, records: list[FrameRecord], relative_to: str | Path | None = None) -> None:
    csv_path = Path(csv_path)
    base = Path(relative_to or csv_path.parent).resolve()
    names = [f.name for f in fields(FrameRecord)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        for r in records:
            d = {n: getattr(r, n) for n in names}
            for k in ("frame_path", "mask_path", "rgb_path"):
                if d[k]:
                    p = Path(d[k]).resolve()
                    d[k] = str(p.relative_to(base)) if p.is_relative_to(base) else str(p)
            w.writerow({k: ("" if v is None else v) for k, v in d.items()})


def _read_array(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in (".tif", ".tiff"):
        import tifffile  # lazy

        return tifffile.imread(path)
    from PIL import Image  # lazy

    return np.asarray(Image.open(path))


def load_frame(rec: FrameRecord) -> np.ndarray:
    """Float32 degrees C, shape (H, W). Multi-channel TIFFs take channel 0 (check your dataset!)."""
    a = _read_array(rec.frame_path).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] < a.shape[0] else a[0]
    return a


def _raw_mask(rec: FrameRecord) -> np.ndarray | None:
    if not rec.has_mask:
        return None
    m = _read_array(rec.mask_path)
    return m[..., 0] if m.ndim == 3 else m


def load_mask(rec: FrameRecord) -> np.ndarray | None:
    """Boolean FIRE mask (MASK_IGNORE pixels are not fire)."""
    m = _raw_mask(rec)
    return None if m is None else (m > 0) & (m != MASK_IGNORE)


def load_ignore(rec: FrameRecord) -> np.ndarray | None:
    """Boolean 'unknown' mask, or None if the frame has no ignore pixels."""
    m = _raw_mask(rec)
    if m is None:
        return None
    ig = m == MASK_IGNORE
    return ig if ig.any() else None


def frame_label(rec: FrameRecord, mask: np.ndarray | None = None) -> int:
    if rec.label is not None:
        return int(rec.label)
    m = load_mask(rec) if mask is None else mask
    return int(bool(m is not None and m.any()))


def assign_group_split(records: list[FrameRecord], test_frac: float = 0.2, seed: int = 0) -> list[FrameRecord]:
    """Hold out whole groups for test, stratified on "group contains fire" so both classes appear in test."""
    rng = np.random.default_rng(seed)
    groups: dict[str, bool] = {}
    for r in records:
        groups[r.group] = groups.get(r.group, False) or bool(frame_label(r))
    test: set[str] = set()
    for has_fire in (True, False):
        g = sorted(k for k, v in groups.items() if v is has_fire)
        rng.shuffle(g)
        k = int(round(test_frac * len(g)))
        if len(g) >= 2:
            k = min(max(k, 1), len(g) - 1)  # keep at least one group of each class on both sides
        test.update(g[:k])
    return [replace(r, split="test" if r.group in test else "train") for r in records]
