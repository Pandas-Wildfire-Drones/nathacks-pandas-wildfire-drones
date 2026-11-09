"""Render a radiometric ThermalFrame (°C) for a camera pose from the world rasters + current fire state.

Pixel -> ground uses perception.georef.camera_axes (the same math the FlatGroundGeoreferencer inverts), so a
detection at pixel (u, v) georeferences back to where the sim put the heat.

Composition per pixel (all temperatures from real FLAME 3 data on the real maps):
  ground = bg + (char - bg) * burnt_fraction - wet cooling
  flame  = ground + max(fire_tex - ground, 0) * temp_scale(fire type)
  pixel  = Planck^-1( (1 - heat) * L(ground) + heat * L(flame) )      mixing in RADIANCE, not °C
then NETD noise and gain-mode saturation (ThermalCameraSpec).

Split in two so the expensive part can run in a worker thread: `prepare()` (main thread) copies the small fire-grid
crops under the footprint; `render()` (any thread) does the sampling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from contracts import CameraModel, Pose
from perception.georef import camera_axes
from perception.radiometry import planck_radiance, radiance_to_temp
from sim.fire import FireModel
from sim.static import WorldStatic

__all__ = ["FrameJob", "ground_coords", "prepare", "render"]


@dataclass
class FrameJob:
    pose: Pose
    camera: CameraModel
    fine_box: tuple[int, int, int, int]
    coarse_box: tuple[int, int, int, int]
    heat: np.ndarray
    burnt: np.ndarray
    tscale: np.ndarray
    wet: np.ndarray
    noise_seed: int
    netd_c: float
    saturation_c: float


def ground_coords(pose: Pose, camera: CameraModel, ground_z: float = 0.0, us: np.ndarray | None = None,
                  vs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ground (x, y) of pixel centres (u + 0.5, v + 0.5); `us`/`vs` select pixel columns/rows (default: all)."""
    axis, right, img_up = camera_axes(pose, camera)
    f = (camera.width_px / 2.0) / math.tan(math.radians(camera.hfov_deg) / 2.0)
    us = np.arange(camera.width_px) if us is None else np.asarray(us, dtype=np.float64)
    vs = np.arange(camera.height_px) if vs is None else np.asarray(vs, dtype=np.float64)
    u = (us + 0.5 - camera.width_px / 2.0) / f
    v = (vs + 0.5 - camera.height_px / 2.0) / f
    ray = axis[None, None, :] + u[None, :, None] * right[None, None, :] - v[:, None, None] * img_up[None, None, :]
    c = pose.position
    t = (ground_z - c.z) / np.minimum(ray[..., 2], -1e-6)
    return c.x + t * ray[..., 0], c.y + t * ray[..., 1]


def _box(rows: np.ndarray, cols: np.ndarray, n: int, pad: int = 2) -> tuple[int, int, int, int]:
    r0 = int(max(math.floor(rows.min()) - pad, 0))
    r1 = int(min(math.ceil(rows.max()) + pad + 1, n))
    c0 = int(max(math.floor(cols.min()) - pad, 0))
    c1 = int(min(math.ceil(cols.max()) + pad + 1, n))
    return r0, max(r1, r0 + 1), c0, max(c1, c0 + 1)


def prepare(static: WorldStatic, fire: FireModel, pose: Pose, camera: CameraModel, noise_seed: int,
            netd_c: float = 0.05, saturation_c: float = 550.0) -> FrameJob:
    xs, ys = ground_coords(pose, camera, us=[0, camera.width_px - 1], vs=[0, camera.height_px - 1])  # corners
    R = static.radius
    fr, fc = (ys + R) / static.res - 0.5, (xs + R) / static.res - 0.5
    cr, cc = (ys + R) / fire.cell - 0.5, (xs + R) / fire.cell - 0.5
    fb = _box(fr, fc, static.n_px)
    cb = _box(cr, cc, fire.n)
    sl = (slice(cb[0], cb[1]), slice(cb[2], cb[3]))
    burnt = np.where(fire.ignited[sl], 1.0 - np.clip(fire.fuel[sl], 0, 1), 0.0).astype(np.float32)
    return FrameJob(pose, camera, fb, cb, fire.heat[sl].copy(), burnt, fire.tscale[sl].copy(),
                    fire.wet[sl].copy(), noise_seed, netd_c, saturation_c)


def render(static: WorldStatic, job: FrameJob, fire_cell_m: float) -> np.ndarray:
    R = static.radius
    xs, ys = ground_coords(job.pose, job.camera)
    fr = (ys + R) / static.res - 0.5 - job.fine_box[0]
    fc = (xs + R) / static.res - 0.5 - job.fine_box[2]
    cr = (ys + R) / fire_cell_m - 0.5 - job.coarse_box[0]
    cc = (xs + R) / fire_cell_m - 0.5 - job.coarse_box[2]
    fsl = (slice(job.fine_box[0], job.fine_box[1]), slice(job.fine_box[2], job.fine_box[3]))
    fine = np.stack([fr, fc])
    coarse = np.stack([cr, cc])
    samp = lambda a, co, order: ndimage.map_coordinates(a, co, order=order, mode="nearest")  # noqa: E731
    bg = samp(static.bg[fsl], fine, 1)
    ch = samp(static.char[fsl], fine, 1)
    fx = samp(static.fire_tex[fsl], fine, 1)
    heat = np.clip(samp(job.heat, coarse, 1), 0.0, 1.0)
    burnt = np.clip(samp(job.burnt, coarse, 1), 0.0, 1.0)
    wet = samp(job.wet, coarse, 1)
    ts = samp(job.tscale, coarse, 0)
    ground = bg + (ch - bg) * burnt - np.minimum(np.maximum(wet, 0.0), 6.0) * 0.8
    if heat.max() > 0:
        flame = ground + np.maximum(fx - ground, 0.0) * ts
        rad = (1.0 - heat) * planck_radiance(ground) + heat * planck_radiance(flame)
        temp = np.where(heat > 1e-4, radiance_to_temp(rad), ground)
    else:
        temp = ground
    rng = np.random.default_rng(job.noise_seed)
    temp = temp + rng.normal(0.0, job.netd_c, temp.shape)
    return np.clip(temp, -40.0, job.saturation_c).astype(np.float32)
