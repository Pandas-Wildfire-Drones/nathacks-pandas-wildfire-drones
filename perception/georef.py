"""Pixel <-> world projection for a pinhole thermal camera (contracts.interfaces.Georeferencer).

Conventions (contracts/geometry.py): ENU meters; z = height above scenario datum (NOT AGL); yaw = compass
degrees clockwise from north; pitch + = nose up; roll + = right side down. Image: u = column (right),
v = row (down). CameraModel.mount_pitch_deg: -90 = nadir, 0 = looking forward.

For a nadir camera at yaw 0: image top (v = 0) points NORTH, image right (u = W) points EAST.

Error budget at 80 m AGL (why the tracker gates at ~5 m): 1 deg of attitude error ~ 1.4 m on the ground;
RTK ~ 0.02 m; flat-ground assumption on a 10% slope ~ several metres off-nadir. Use TerrainGeoreferencer with
a DEM height function when the scenario has terrain (perception.terrain.DemSampler).

Lens distortion: pass a LensDistortion to the georeferencers / world_to_pixel; observed pixels are undistorted
before the pinhole ray is built (docs/decisions/0002). None = ideal pinhole.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from contracts import CameraModel, Pose, Position

from .distortion import LensDistortion

__all__ = ["camera_axes", "pixel_ray", "FlatGroundGeoreferencer", "TerrainGeoreferencer", "world_to_pixel",
           "footprint_corners"]


def _yaw_rotate(v: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a level-frame vector clockwise by a compass yaw (north -> east for +90)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c, v[2]])


def camera_axes(pose: Pose, camera: CameraModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(optical axis, image-right, image-up) as ENU unit vectors."""
    p, r = math.radians(pose.pitch_deg), math.radians(pose.roll_deg)
    fwd, right, up = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    # body pitch (about right axis, nose up +)
    fwd, up = math.cos(p) * fwd + math.sin(p) * up, -math.sin(p) * fwd + math.cos(p) * up
    # body roll (about forward axis, right side down +)
    right, up = math.cos(r) * right - math.sin(r) * up, math.sin(r) * right + math.cos(r) * up
    fwd, right, up = (_yaw_rotate(x, pose.yaw_deg) for x in (fwd, right, up))
    m = math.radians(camera.mount_pitch_deg)
    axis = math.cos(m) * fwd + math.sin(m) * up
    img_up = -math.sin(m) * fwd + math.cos(m) * up
    return axis, right, img_up


def _focal_px(camera: CameraModel) -> float:
    return (camera.width_px / 2.0) / math.tan(math.radians(camera.hfov_deg) / 2.0)


def pixel_ray(u: float, v: float, pose: Pose, camera: CameraModel,
              distortion: LensDistortion | None = None) -> np.ndarray:
    """Unnormalized ENU direction of the ray through observed pixel (u, v). Pixel centres are at integer + 0.5."""
    if distortion is not None:
        u, v = distortion.undistort_uv(u, v, camera)
    axis, right, img_up = camera_axes(pose, camera)
    f = _focal_px(camera)
    cx, cy = camera.width_px / 2.0, camera.height_px / 2.0
    return axis + ((u - cx) / f) * right - ((v - cy) / f) * img_up


class FlatGroundGeoreferencer:
    """Intersect the pixel ray with the plane z = ground_z (default: the scenario datum)."""

    def __init__(self, ground_z: float = 0.0, distortion: LensDistortion | None = None):
        self.ground_z = ground_z
        self.distortion = distortion

    def pixel_to_world(self, u: float, v: float, pose: Pose, camera: CameraModel) -> Position | None:
        ray = pixel_ray(u, v, pose, camera, self.distortion)
        c = pose.position
        if ray[2] >= -1e-9:
            return None  # at or above the horizon
        t = (self.ground_z - c.z) / ray[2]
        if t <= 0:
            return None  # ground plane is above the camera
        return Position(x=c.x + t * ray[0], y=c.y + t * ray[1], z=self.ground_z)


class TerrainGeoreferencer:
    """Ray-march against a height function z = ground_z(x, y) (e.g. a DEM sampler). Step = step_m along the ray."""

    def __init__(self, ground_z: Callable[[float, float], float], step_m: float = 1.0, max_range_m: float = 2000.0,
                 distortion: LensDistortion | None = None):
        self.ground_z, self.step_m, self.max_range_m = ground_z, step_m, max_range_m
        self.distortion = distortion

    def pixel_to_world(self, u: float, v: float, pose: Pose, camera: CameraModel) -> Position | None:
        ray = pixel_ray(u, v, pose, camera, self.distortion)
        ray = ray / np.linalg.norm(ray)
        c = np.array([pose.position.x, pose.position.y, pose.position.z])
        prev_t, prev_h = 0.0, c[2] - self.ground_z(c[0], c[1])
        if prev_h < 0:
            return None
        t = self.step_m
        while t <= self.max_range_m:
            p = c + t * ray
            h = p[2] - self.ground_z(p[0], p[1])
            if h <= 0:  # crossed the surface between prev_t and t: interpolate
                tt = prev_t + (t - prev_t) * prev_h / (prev_h - h)
                q = c + tt * ray
                return Position(x=float(q[0]), y=float(q[1]), z=float(self.ground_z(q[0], q[1])))
            prev_t, prev_h, t = t, h, t + self.step_m
        return None


def world_to_pixel(point: Position, pose: Pose, camera: CameraModel,
                   distortion: LensDistortion | None = None) -> tuple[float, float] | None:
    """Project a world point into the (distorted, if given) image. None if behind the camera or outside the frame."""
    axis, right, img_up = camera_axes(pose, camera)
    w = np.array([point.x - pose.position.x, point.y - pose.position.y, point.z - pose.position.z])
    zc = float(w @ axis)
    if zc <= 1e-9:
        return None
    f = _focal_px(camera)
    u = camera.width_px / 2.0 + f * float(w @ right) / zc
    v = camera.height_px / 2.0 - f * float(w @ img_up) / zc
    if distortion is not None:
        u, v = distortion.distort_uv(u, v, camera)
    if 0 <= u < camera.width_px and 0 <= v < camera.height_px:
        return u, v
    return None


def footprint_corners(pose: Pose, camera: CameraModel, georef) -> list[Position] | None:
    """Ground polygon of the frame (image corners, clockwise from top-left). None if any corner misses the ground."""
    w, h = camera.width_px, camera.height_px
    pts = [georef.pixel_to_world(u, v, pose, camera) for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
    return None if any(p is None for p in pts) else pts
