"""Patch extraction + normalization: the ONE place that defines what the classifier sees.

Training (training/) and inference (perception/detectors/fused.py) both call these functions, so the
preprocessing cannot drift. If you change anything here, bump PatchSpec.version: the ONNX metadata stores
the version it was trained with, and the classifier refuses to load a model trained on a different one.

Input is radiometric degrees Celsius (float32). Never feed 8-bit palette images (FLAME 1/2) through this.

Channels (all float32, shape (C, P, P)):
  0  absolute temperature  : clip(T, abs_min_c, abs_max_c) scaled to roughly [0, 1]
                             -> lets the model tell a 35 C warm rock from a 300 C ember
  1  local contrast        : (T - background) / contrast_scale_c, background = median of the patch border
                             -> robust to ambient differences between nights / sites
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["PatchSpec", "PATCH_SPEC", "extract_patch", "extract_patches", "normalize_patch", "normalize_patches"]


@dataclass(frozen=True)
class PatchSpec:
    version: str = "1"
    size_px: int = 32              # P. At 80 m AGL / 7.2 cm GSD a 32 px patch spans ~2.3 m of ground
    abs_min_c: float = -20.0
    abs_max_c: float = 600.0       # above the 550 C low-gain saturation in ThermalCameraSpec
    contrast_scale_c: float = 50.0
    border_px: int = 3             # ring used to estimate local background
    channels: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


PATCH_SPEC = PatchSpec()


def extract_patch(temp_c: np.ndarray, u: float, v: float, size_px: int = PATCH_SPEC.size_px) -> np.ndarray:
    """Square P x P crop of degrees C centred on pixel column u, row v. Edges are padded by replication."""
    half = size_px // 2
    cu, cv = int(round(u)), int(round(v))
    padded = np.pad(temp_c, half + 1, mode="edge")
    r0, c0 = cv + 1, cu + 1  # shift for the padding: original (cv, cu) is now at (cv+half+1, cu+half+1)
    return padded[r0:r0 + size_px, c0:c0 + size_px].astype(np.float32, copy=False)


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


def extract_patches(temp_c: np.ndarray, centers_uv: list[tuple[float, float]],
                    spec: PatchSpec = PATCH_SPEC) -> np.ndarray:
    """Batch of normalized patches, shape (N, C, P, P). N = 0 gives an empty (0, C, P, P) array."""
    if not centers_uv:
        return np.zeros((0, spec.channels, spec.size_px, spec.size_px), dtype=np.float32)
    return normalize_patches(np.stack([extract_patch(temp_c, u, v, spec.size_px) for u, v in centers_uv]), spec)
