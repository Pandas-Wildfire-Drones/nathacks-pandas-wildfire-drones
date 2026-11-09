"""[STRETCH, GBM only] Hand-made blob features for the gradient-boosting verifier.

The PRIMARY verifier is the CNN, which reads the normalized patch directly (perception/patches.py).
This module exists only for the stretch-goal GBM (training/train_gbm.py, classifiers.OnnxGbmClassifier).
Like patches.py, training and inference share this exact code; bump FEATURE_VERSION on any change.
"""
from __future__ import annotations

import numpy as np

from .patches import PATCH_SPEC, PatchSpec, extract_patch_scaled, per_centre_gsd

__all__ = ["FEATURE_VERSION", "FEATURE_NAMES", "patch_features", "features_at"]

FEATURE_VERSION = "2"  # v2: features computed on scale-normalized patches (PatchSpec v2)
FEATURE_NAMES = (
    "peak_c", "mean_c", "std_c", "p90_c", "background_c", "peak_minus_bg", "mean_minus_bg",
    "hot5_frac", "hot10_frac", "hot50_frac", "center_minus_ring", "grad_mean", "grad_max",
)


def patch_features(patches_c: np.ndarray, spec: PatchSpec = PATCH_SPEC) -> np.ndarray:
    """(N, P, P) degrees C -> (N, len(FEATURE_NAMES)) float32."""
    x = np.asarray(patches_c, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    n, p, _ = x.shape
    flat = x.reshape(n, -1)
    b = spec.border_px
    ring = np.concatenate([x[:, :b].reshape(n, -1), x[:, -b:].reshape(n, -1),
                           x[:, b:-b, :b].reshape(n, -1), x[:, b:-b, -b:].reshape(n, -1)], axis=1)
    bg = np.median(ring, axis=1)
    peak, mean = flat.max(1), flat.mean(1)
    excess = flat - bg[:, None]
    c0, c1 = p // 2 - 2, p // 2 + 2
    center = x[:, c0:c1, c0:c1].reshape(n, -1).mean(1)
    gy, gx = np.gradient(x, axis=(1, 2))
    grad = np.hypot(gx, gy).reshape(n, -1)
    feats = np.stack([
        peak, mean, flat.std(1), np.percentile(flat, 90, axis=1), bg, peak - bg, mean - bg,
        (excess > 5).mean(1), (excess > 10).mean(1), (excess > 50).mean(1), center - bg,
        grad.mean(1), grad.max(1),
    ], axis=1)
    return feats.astype(np.float32)


def features_at(temp_c: np.ndarray, centers_uv: list[tuple[float, float]], spec: PatchSpec = PATCH_SPEC,
                gsd_m=None) -> np.ndarray:
    """gsd_m: scalar, one value per centre, or None (see perception.patches.GsdArg)."""
    if not centers_uv:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    gs = per_centre_gsd(gsd_m, len(centers_uv))
    return patch_features(np.stack([extract_patch_scaled(temp_c, u, v, g, spec)
                                    for (u, v), g in zip(centers_uv, gs)]), spec)
