"""Classical detector (DetectorKind.THRESHOLD): temperature threshold + connected components.

Role in the mixed approach: the RECALL stage. It proposes every blob that could be a fire; the ML verifier
(fused.py) removes false alarms. So these thresholds are deliberately permissive.

A pixel is "hot" if either
  * T >= abs_threshold_c                                          (obviously burning), or
  * T - local_background >= contrast_threshold_c and T >= min_temp_c (warm against a cold night background;
    catches small/sub-pixel spots whose reading is diluted by the cooler ground in the same pixel)

All numbers are [ASSUMPTION] defaults; tune them on validation data (never on the test split).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from contracts import Detection, DetectorKind, ThermalFrame

__all__ = ["ThresholdParams", "ThresholdDetector", "local_background", "Blob", "find_blobs"]


@dataclass(frozen=True)
class ThresholdParams:
    abs_threshold_c: float = 60.0        # [ASSUMPTION] smouldering ground is often 100-300 C; 60 keeps margin
    contrast_threshold_c: float = 8.0    # [ASSUMPTION] over local background (night ground ~ ambient +- few C)
    min_temp_c: float = 20.0             # [ASSUMPTION] contrast hits below this are ignored (cold water, shadows)
    background_block_px: int = 32        # block size for the coarse local-background median
    min_area_px: int = 1                 # sub-pixel hotspots are 1-2 px blobs; do not filter them out
    max_blobs: int = 64                  # keep the hottest N per frame (bounds ML cost on the Jetson)
    connectivity: int = 8


@dataclass(frozen=True)
class Blob:
    u: float                 # excess-temperature-weighted centroid column
    v: float                 # centroid row
    area_px: int
    peak_temp_c: float
    mean_temp_c: float
    background_c: float
    bbox: tuple[float, float, float, float]  # (u0, v0, u1, v1), u1/v1 exclusive


def local_background(temp_c: np.ndarray, block_px: int = 32) -> np.ndarray:
    """Coarse local background: median per block, bilinearly upsampled. Much cheaper than a big median filter."""
    h, w = temp_c.shape
    bh, bw = -(-h // block_px), -(-w // block_px)
    padded = np.pad(temp_c, ((0, bh * block_px - h), (0, bw * block_px - w)), mode="edge")
    blocks = padded.reshape(bh, block_px, bw, block_px).transpose(0, 2, 1, 3).reshape(bh, bw, -1)
    med = np.median(blocks, axis=2).astype(np.float32)
    if bh == 1 and bw == 1:
        return np.full_like(temp_c, med[0, 0], dtype=np.float32)
    up = ndimage.zoom(med, (h / bh, w / bw), order=1, mode="nearest", grid_mode=True)
    return up[:h, :w].astype(np.float32)


def find_blobs(temp_c: np.ndarray, params: ThresholdParams = ThresholdParams()) -> list[Blob]:
    temp = np.asarray(temp_c, dtype=np.float32)
    bg = local_background(temp, params.background_block_px)
    excess = temp - bg
    hot = (temp >= params.abs_threshold_c) | ((excess >= params.contrast_threshold_c) & (temp >= params.min_temp_c))
    if not hot.any():
        return []
    structure = np.ones((3, 3)) if params.connectivity == 8 else None
    labels, n = ndimage.label(hot, structure=structure)
    blobs: list[Blob] = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        m = labels[sl] == i
        area = int(m.sum())
        if area < params.min_area_px:
            continue
        t = temp[sl][m]
        wts = np.maximum(excess[sl][m], 1e-3)
        rows, cols = np.nonzero(m)
        v = float((rows * wts).sum() / wts.sum()) + sl[0].start + 0.5  # +0.5: pixel-centre convention (georef.py)
        u = float((cols * wts).sum() / wts.sum()) + sl[1].start + 0.5
        blobs.append(Blob(
            u=u, v=v, area_px=area, peak_temp_c=float(t.max()), mean_temp_c=float(t.mean()),
            background_c=float(np.median(bg[sl][m])),
            bbox=(float(sl[1].start), float(sl[0].start), float(sl[1].stop), float(sl[0].stop)),
        ))
    blobs.sort(key=lambda b: b.peak_temp_c, reverse=True)
    return blobs[: params.max_blobs]


def classical_confidence(blob: Blob, params: ThresholdParams) -> float:
    """Heuristic 0..1 score: how far the blob is above the contrast threshold (logistic, 10 C scale)."""
    margin = (blob.peak_temp_c - blob.background_c) - params.contrast_threshold_c
    return float(1.0 / (1.0 + np.exp(-margin / 10.0)))


class ThresholdDetector:
    """Detector protocol implementation (contracts.interfaces.Detector)."""

    kind = DetectorKind.THRESHOLD

    def __init__(self, params: ThresholdParams = ThresholdParams()):
        self.params = params

    def blobs(self, temp_c: np.ndarray) -> list[Blob]:
        return find_blobs(temp_c, self.params)

    def detect(self, frame: ThermalFrame) -> list[Detection]:
        return [
            Detection(u=b.u, v=b.v, area_px=b.area_px, peak_temp_c=b.peak_temp_c, mean_temp_c=b.mean_temp_c,
                      confidence=classical_confidence(b, self.params), source=DetectorKind.THRESHOLD, bbox=b.bbox)
            for b in self.blobs(frame.temp_c)
        ]
