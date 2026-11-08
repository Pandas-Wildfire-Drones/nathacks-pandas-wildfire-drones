"""Lens distortion (Brown-Conrady: radial k1, k2 + tangential p1, p2) for georeferencing.

contracts.CameraModel is an ideal pinhole. Real thermal lenses, especially the extinguisher's 50 deg lens, bend
straight lines near the image edge. This model is PERCEPTION-LOCAL on purpose (docs/decisions/0002): adding
coefficients to CameraModel is a contract change, which we defer until real cameras are calibrated (heated
checkerboard). The simulator can pass the same LensDistortion to FakeDetector (to "distort") and to the
georeferencer (to "undistort") to show the error with and without calibration.

Coordinates: normalized x = (u - cx) / f, y = (v - cy) / f, with f the pinhole focal length in pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from contracts import CameraModel

__all__ = ["LensDistortion"]


def _focal(camera: CameraModel) -> float:
    return (camera.width_px / 2.0) / math.tan(math.radians(camera.hfov_deg) / 2.0)


@dataclass(frozen=True)
class LensDistortion:
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.k1 == self.k2 == self.p1 == self.p2 == 0.0

    def _apply(self, x: float, y: float) -> tuple[float, float]:
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2
        xd = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return xd, yd

    def distort_uv(self, u: float, v: float, camera: CameraModel) -> tuple[float, float]:
        """Ideal (pinhole) pixel -> where the real lens puts it."""
        if self.is_identity:
            return u, v
        f, cx, cy = _focal(camera), camera.width_px / 2.0, camera.height_px / 2.0
        xd, yd = self._apply((u - cx) / f, (v - cy) / f)
        return cx + f * xd, cy + f * yd

    def undistort_uv(self, u: float, v: float, camera: CameraModel, iterations: int = 20) -> tuple[float, float]:
        """Observed pixel -> ideal pinhole pixel (fixed-point iteration; accurate for mild distortion)."""
        if self.is_identity:
            return u, v
        f, cx, cy = _focal(camera), camera.width_px / 2.0, camera.height_px / 2.0
        xd, yd = (u - cx) / f, (v - cy) / f
        x, y = xd, yd
        for _ in range(iterations):
            ex, ey = self._apply(x, y)
            x, y = x + (xd - ex), y + (yd - ey)
        return cx + f * x, cy + f * y
