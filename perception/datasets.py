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
    gsd_m        ground sample distance, metres/pixel, if known                               optional
(*) at least one of mask_path or label is required.

Add a dataset by writing a tiny adapter that emits this CSV (see training/build_manifest_flame3.py).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

__all__ = ["FrameRecord", "load_manifest", "write_manifest", "load_frame", "load_mask", "assign_group_split"]


@dataclass(frozen=True)
class FrameRecord:
    frame_path: str
    group: str
    mask_path: str = ""
    label: int | None = None
    split: str = ""
    source: str = ""
    gsd_m: float | None = None

    @property
    def has_mask(self) -> bool:
        return bool(self.mask_path)


def load_manifest(csv_path: str | Path, split: str | None = None) -> list[FrameRecord]:
    csv_path = Path(csv_path)
    root = csv_path.parent
    out = []
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            def res(p: str) -> str:
                return "" if not p else str(p if Path(p).is_absolute() else (root / p))
            lab = row.get("label", "")
            gsd = row.get("gsd_m", "")
            rec = FrameRecord(frame_path=res(row["frame_path"]), group=row["group"], mask_path=res(row.get("mask_path", "")),
                              label=int(lab) if lab not in ("", None) else None, split=row.get("split", "") or "",
                              source=row.get("source", "") or "", gsd_m=float(gsd) if gsd not in ("", None) else None)
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
            for k in ("frame_path", "mask_path"):
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


def load_mask(rec: FrameRecord) -> np.ndarray | None:
    if not rec.has_mask:
        return None
    m = _read_array(rec.mask_path)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0


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
