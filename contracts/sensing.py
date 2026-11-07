"""Perception-facing types: camera model, thermal frames, detections, obstacles.

PRINCIPLE: raw frames and point clouds NEVER travel on the bus. They stay on the drone.
Only small results (Detection -> Hotspot, Obstacle) leave the perception layer.
ThermalFrame is therefore a plain dataclass holding a numpy array, not a Pydantic model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field

from .base import FrozenContract
from .enums import DetectorKind
from .geometry import Pose, Position

if TYPE_CHECKING:  # contracts depend only on pydantic at runtime
    import numpy as np

__all__ = ["CameraModel", "ThermalFrame", "Detection", "Obstacle"]


class CameraModel(FrozenContract):
    """Pinhole camera. Footprint math below assumes a nadir (straight-down) mount."""

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    hfov_deg: float = Field(gt=0, lt=180)
    mount_pitch_deg: float = Field(default=-90.0, ge=-90, le=90)  # -90 = looking straight down

    @property
    def vfov_deg(self) -> float:
        half_h = math.tan(math.radians(self.hfov_deg) / 2)
        return math.degrees(2 * math.atan(half_h * self.height_px / self.width_px))

    def swath_m(self, agl_m: float) -> float:
        """Ground width covered by the horizontal FOV at height `agl_m` (nadir)."""
        return 2 * agl_m * math.tan(math.radians(self.hfov_deg) / 2)

    def ground_sample_distance_m(self, agl_m: float) -> float:
        """Meters of ground per pixel at height `agl_m` (nadir)."""
        return self.swath_m(agl_m) / self.width_px


@dataclass
class ThermalFrame:
    """One radiometric thermal image. NOT serialized onto the bus.

    temp_c: float32 array, shape (height_px, width_px), temperature in degrees Celsius.
            Keep per-pixel temperatures; do not convert to 8-bit before thresholding.
    """

    drone_id: str
    timestamp: float
    temp_c: "np.ndarray"
    pose: Pose
    camera: CameraModel


class Detection(FrozenContract):
    """Output of a Detector, in IMAGE coordinates. A Georeferencer turns it into a Hotspot position."""

    u: float  # centroid column (pixels)
    v: float  # centroid row (pixels)
    area_px: float = Field(ge=0)
    peak_temp_c: float
    mean_temp_c: float | None = None
    confidence: float = Field(ge=0, le=1)
    source: DetectorKind
    bbox: tuple[float, float, float, float] | None = None  # (u0, v0, u1, v1)


class Obstacle(FrozenContract):
    """Obstacle reported by an obstacle sensor (lidar/radar) or the simulator."""

    position: Position
    radius_m: float = Field(gt=0)
    height_m: float | None = Field(default=None, ge=0)
    kind: str = "unknown"  # e.g. "tree", "pole", "terrain"
