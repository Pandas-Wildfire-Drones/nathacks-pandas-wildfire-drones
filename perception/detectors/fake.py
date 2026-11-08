"""FakeDetector (DetectorKind.FAKE): simulator ground truth -> Detections with tunable misses and false alarms.

Purpose: lets workstreams A, B and D run end to end from day one, before any real detector or thermal
synthesizer exists. It ignores frame.temp_c (you may pass an empty array) and uses the frame's pose and camera.

The sim supplies `truth(now) -> list[TruthSource]` with the CURRENT fire/decoy state (fires grow, get suppressed).
Probability model ([ASSUMPTION] defaults, tune for the scenario, don't tune to flatter the demo):
  * fire seen with p = pd_max * (1 - exp(-area_px / area_px_50)) * (1 if temp >= min_temp_c else 0)
    (small spots covering few pixels are missed more often: sub-pixel dilution)
  * decoy reported as a detection with p = p_decoy (it is warm; a good verifier should push this down)
  * clutter false alarms: Poisson(clutter_rate) per frame at uniform random pixels
  * centroid noise: N(0, pos_noise_px) per axis
Deterministic given the seed and the call order (hard rule 5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from contracts import Detection, DetectorKind, Position, ThermalFrame

from ..distortion import LensDistortion
from ..georef import world_to_pixel

__all__ = ["TruthSource", "FakeDetector"]


@dataclass(frozen=True)
class TruthSource:
    source_id: str
    position: Position
    area_m2: float
    temp_c: float
    is_fire: bool = True  # False = decoy (warm rock, vehicle, permitted campfire)


class FakeDetector:
    kind = DetectorKind.FAKE

    def __init__(self, truth: Callable[[float], list[TruthSource]], seed: int = 0, pd_max: float = 0.95,
                 area_px_50: float = 2.0, min_temp_c: float = 60.0, p_decoy: float = 0.3,
                 clutter_rate: float = 0.02, pos_noise_px: float = 0.7, distortion: LensDistortion | None = None):
        self.truth = truth
        self.distortion = distortion  # sim-only: emulate a real lens so georef calibration matters (decision 0002)
        self.rng = np.random.default_rng(seed)
        self.pd_max, self.area_px_50, self.min_temp_c = pd_max, area_px_50, min_temp_c
        self.p_decoy, self.clutter_rate, self.pos_noise_px = p_decoy, clutter_rate, pos_noise_px
        self.last_truth_ids: list[str | None] = []  # parallel to the last detect() output; None = clutter

    def _gsd(self, frame: ThermalFrame, ground_z: float) -> float:
        agl = max(frame.pose.position.z - ground_z, 0.1)
        return frame.camera.ground_sample_distance_m(agl)

    def detect(self, frame: ThermalFrame) -> list[Detection]:
        cam, out, ids = frame.camera, [], []
        for src in self.truth(frame.timestamp):
            uv = world_to_pixel(src.position, frame.pose, cam, self.distortion)
            if uv is None:
                continue
            area_px = src.area_m2 / self._gsd(frame, src.position.z) ** 2
            if src.is_fire:
                p = self.pd_max * (1 - math.exp(-area_px / self.area_px_50)) if src.temp_c >= self.min_temp_c else 0.0
                conf = float(self.rng.beta(8, 2))
            else:
                p, conf = self.p_decoy, float(self.rng.beta(3, 4))
            if self.rng.random() >= p:
                continue
            u = float(np.clip(uv[0] + self.rng.normal(0, self.pos_noise_px), 0, cam.width_px - 1e-3))
            v = float(np.clip(uv[1] + self.rng.normal(0, self.pos_noise_px), 0, cam.height_px - 1e-3))
            out.append(Detection(u=u, v=v, area_px=max(area_px, 1.0), peak_temp_c=src.temp_c,
                                 confidence=conf, source=DetectorKind.FAKE))
            ids.append(src.source_id)
        for _ in range(self.rng.poisson(self.clutter_rate)):
            out.append(Detection(u=float(self.rng.uniform(0, cam.width_px)), v=float(self.rng.uniform(0, cam.height_px)),
                                 area_px=1.0, peak_temp_c=float(self.rng.uniform(25, 60)),
                                 confidence=float(self.rng.beta(2, 5)), source=DetectorKind.FAKE))
            ids.append(None)
        self.last_truth_ids = ids
        return out
