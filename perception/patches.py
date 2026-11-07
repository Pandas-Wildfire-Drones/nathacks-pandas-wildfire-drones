"""Patch extraction + normalization: the ONE place that defines what the classifier sees.

Training (training/) and inference (perception/detectors/, perception/confirm.py) both call these functions, so
the preprocessing cannot drift. If you change anything here, bump PatchSpec.version: the ONNX metadata stores the
version it was trained with, and the classifier refuses to load a model trained on a different one.

Input is radiometric degrees Celsius (float32). Never feed 8-bit palette images (FLAME 1/2) through this.

SCALE NORMALIZATION (v2). A drone sees the same ember at 7.2 cm/px when searching (80 m AGL) and 2.7 cm/px when
it descends to confirm (30 m AGL). A fixed 32-pixel crop would cover 2.3 m of ground in one case and 0.9 m in the
other, so the CNN would see a different-looking object at every altitude. When the ground sample distance (GSD)
is known, the patch is a fixed GROUND window (ground_size_m) resampled to size_px IN RADIANCE, which approximates
what the camera would have read at the reference GSD. One model then serves search and confirm. Without a GSD
the patch is a plain pixel crop (v1 behaviour), and the model's sidecar says which one it was trained with.

Channels (all float32, shape (C, P, P)):
  0  absolute temperature  : clip(T, abs_min_c, abs_max_c) scaled to roughly [0, 1]
                             -> lets the model tell a 35 C warm rock from a 300 C ember
  1  local contrast        : (T - background) / contrast_scale_c, background = median of the patch border
                             -> robust to ambient differences between nights / sites

Pixel convention (same as georef.py): pixel (row r, col c) covers [c, c+1) x [r, r+1); its centre is (c+0.5, r+0.5).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .radiometry import planck_radiance, radiance_to_temp

__all__ = ["PatchSpec", "PATCH_SPEC", "extract_patch", "extract_patch_scaled", "extract_patches",
           "normalize_patch", "normalize_patches", "patch_window_px"]


@dataclass(frozen=True)
class PatchSpec:
    version: str = "2"
    size_px: int = 32              # P (network input side)
    ground_size_m: float = 2.3     # ground side of the patch when GSD is known (= 32 px at 80 m AGL, 7.2 cm/px)
    abs_min_c: float = -20.0
    abs_max_c: float = 600.0       # above the camera saturation
    contrast_scale_c: float = 50.0
    border_px: int = 3             # ring used to estimate local background
    channels: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


PATCH_SPEC = PatchSpec()


def patch_window_px(gsd_m: float | None, spec: PatchSpec = PATCH_SPEC) -> float:
    """Source pixels spanned by one patch side (size_px when the GSD is unknown)."""
    return spec.size_px if gsd_m is None else spec.ground_size_m / gsd_m


def extract_patch(temp_c: np.ndarray, u: float, v: float, size_px: int = PATCH_SPEC.size_px) -> np.ndarray:
    """Plain P x P crop of degrees C around the pixel containing (u, v). Edges are padded by replication."""
    half = size_px // 2
    cu, cv = int(math.floor(u)), int(math.floor(v))
    padded = np.pad(temp_c, half + 1, mode="edge")
    r0, c0 = cv + 1, cu + 1  # original (cv, cu) sits at (cv+half+1, cu+half+1) in `padded`
    return padded[r0:r0 + size_px, c0:c0 + size_px].astype(np.float32, copy=False)


def extract_patch_scaled(temp_c: np.ndarray, u: float, v: float, gsd_m: float | None,
                         spec: PatchSpec = PATCH_SPEC) -> np.ndarray:
    """Fixed-ground-size patch resampled to P x P in radiance space. Falls back to a plain crop if gsd_m is None."""
    if gsd_m is None:
        return extract_patch(temp_c, u, v, spec.size_px)
    p = spec.size_px
    scale = patch_window_px(gsd_m, spec) / p          # source pixels per output pixel
    if abs(scale - 1.0) < 0.02:
        return extract_patch(temp_c, u, v, p)
    sigma = 0.4 * scale if scale > 1 else 0.0         # anti-alias before downsampling (~box average)
    half = int(math.ceil(p * scale / 2 + 3 * sigma + 2))
    crop = extract_patch(temp_c, u, v, 2 * half)      # crop index `half` = pixel floor(u), floor(v)
    rad = planck_radiance(crop)
    if sigma > 0:
        rad = ndimage.gaussian_filter(rad, sigma, mode="nearest")
    # continuous offset of the output sample from the crop's centre pixel (index half, centre at floor(u)+0.5)
    off = (np.arange(p) + 0.5 - p / 2) * scale
    du, dv = u - (math.floor(u) + 0.5), v - (math.floor(v) + 0.5)
    rows = half + dv + off
    cols = half + du + off
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    out = ndimage.map_coordinates(rad, [rr, cc], order=1, mode="nearest")
    return radiance_to_temp(out).astype(np.float32)


def normalize_patches(patches_c: np.ndarray, spec: PatchSpec = PATCH_SPEC) -> np.ndarray:
    """(N, P, P) degrees C -> (N, C, P, P) float32 network input. Vectorized; used by training AND inference."""
    p = np.asarray(patches_c, dtype=np.float32)
    n, b = p.shape[0], spec.border_px
    ring = np.concatenate([p[:, :b].reshape(n, -1), p[:, -b:].reshape(n, -1),
                           p[:, b:-b, :b].reshape(n, -1), p[:, b:-b, -b:].reshape(n, -1)], axis=1)
    background = np.median(ring, axis=1)[:, None, None]
    absolute = (np.clip(p, spec.abs_min_c, spec.abs_max_c) - spec.abs_min_c) / (spec.abs_max_c - spec.abs_min_c)
    contrast = np.clip((p - background) / spec.contrast_scale_c, -5.0, 20.0)
    return np.stack([absolute, contrast], axis=1).astype(np.float32)


def normalize_patch(patch_c: np.ndarray, spec: PatchSpec = PATCH_SPEC) -> np.ndarray:
    """P x P degrees C -> (C, P, P) float32 network input."""
    return normalize_patches(np.asarray(patch_c)[None], spec)[0]


def extract_patches(temp_c: np.ndarray, centers_uv: list[tuple[float, float]], spec: PatchSpec = PATCH_SPEC,
                    gsd_m: float | None = None) -> np.ndarray:
    """Batch of normalized patches, shape (N, C, P, P). gsd_m given -> scale-normalized patches."""
    if not centers_uv:
        return np.zeros((0, spec.channels, spec.size_px, spec.size_px), dtype=np.float32)
    return normalize_patches(np.stack([extract_patch_scaled(temp_c, u, v, gsd_m, spec) for u, v in centers_uv]), spec)
