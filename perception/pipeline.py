"""On-drone perception loop: one ThermalFrame (+ optional FrameContext) in -> changed Hotspots out.

Workstream B owns the loop that calls `process(frame, ctx)` each camera tick and publishes the returned hotspots as
HotspotReport; `map.pop_updates()` feeds MapSync. Swap the detector freely:
FakeDetector (day one) -> ThresholdDetector -> FusedDetector(OnnxCnnClassifier(...)).

Sensor wiring (docs/decisions/0002):
  * pose time-sync     PoseBuffer -> pose interpolated at the frame's exposure time; stale -> frame skipped
  * terrain + AGL      DemSampler for georef; GSD from the altimeter (if valid) else pose z - DEM
  * GNSS-RTK lost      wider tracker gate and map hit radius, lower position weight
  * thermal health/FFC frame skipped, and NO "searched, empty" evidence written to the map
  * lens distortion    LensDistortion passed to the georeferencer (perception-local calibration)
  * hot-blob bypass    saturation of the current gain mode; no bypass near friendly drones / known hot sources
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from contracts import Detector, GridSpec, Hotspot, Position, ThermalFrame

from .detectors.fused import ExclusionZone, FusedDetector
from .distortion import LensDistortion
from .frame_context import FrameContext, PoseBuffer
from .georef import FlatGroundGeoreferencer, TerrainGeoreferencer, footprint_corners, world_to_pixel
from .mapping import LogOddsMap
from .terrain import DemSampler
from .tracking import HotspotTracker, Observation

__all__ = ["SeekerPerception", "PerceptionStats"]


@dataclass
class PerceptionStats:
    frames: int = 0
    skipped_thermal: int = 0
    skipped_ffc: int = 0
    skipped_stale_pose: int = 0
    rtk_lost_frames: int = 0
    altimeter_used: int = 0


class SeekerPerception:
    def __init__(self, drone_id: str, detector: Detector, grid: GridSpec | None = None, georef=None,
                 tracker: HotspotTracker | None = None, ground_z: float = 0.0, dem: DemSampler | None = None,
                 pose_buffer: PoseBuffer | None = None, distortion: LensDistortion | None = None,
                 altimeter_max_m: float | None = 120.0, rtk_lost_extra_gate_m: float = 3.0,
                 rtk_lost_pos_weight: float = 0.25, exclusion_radius_m: float = 15.0):
        self.drone_id = drone_id
        self.detector = detector
        self.dem = dem
        self.ground_z = ground_z
        if georef is None:
            georef = (TerrainGeoreferencer(dem, distortion=distortion) if dem is not None
                      else FlatGroundGeoreferencer(ground_z, distortion=distortion))
        self.georef = georef
        self.distortion = distortion
        self.tracker = tracker or HotspotTracker(id_prefix=drone_id)
        self.map = LogOddsMap(grid) if grid is not None else None
        self.pose_buffer = pose_buffer
        self.altimeter_max_m = altimeter_max_m          # [ASSUMPTION] decision 0001 option A: >= 100 m altimeter
        self.rtk_lost_extra_gate_m = rtk_lost_extra_gate_m  # [ASSUMPTION] non-RTK GNSS ~1-3 m [VERIFY]
        self.rtk_lost_pos_weight = rtk_lost_pos_weight
        self.exclusion_radius_m = exclusion_radius_m    # [ASSUMPTION] around friendly drones / flares
        self.stats = PerceptionStats()

    def _ground_at(self, x: float, y: float) -> float:
        return self.dem(x, y) if self.dem is not None else self.ground_z

    def agl_m(self, frame: ThermalFrame, ctx: FrameContext) -> float:
        """Altimeter if it is within its rated range, else pose height minus terrain under the drone."""
        if ctx.agl_m is not None and ctx.agl_m > 0 and (self.altimeter_max_m is None or ctx.agl_m <= self.altimeter_max_m):
            self.stats.altimeter_used += 1
            return ctx.agl_m
        p = frame.pose.position
        return max(p.z - self._ground_at(p.x, p.y), 0.1)

    def _exclusions(self, frame: ThermalFrame, ctx: FrameContext, gsd: float) -> list[ExclusionZone]:
        zones = []
        for reason, pts in (("friendly_drone", ctx.friendly_positions), ("industrial", ctx.known_hot_sources)):
            for pt in pts:  # a drone's hot engine is where the drone is (in the air), so project its 3D position
                uv = world_to_pixel(pt, frame.pose, frame.camera, self.distortion)
                if uv is not None:
                    zones.append(ExclusionZone(uv[0], uv[1], self.exclusion_radius_m / gsd, reason))
        return zones

    def process(self, frame: ThermalFrame, ctx: FrameContext | None = None) -> list[Hotspot]:
        ctx = ctx or FrameContext()
        self.stats.frames += 1
        if not ctx.thermal_ok:
            self.stats.skipped_thermal += 1
            return []
        if ctx.ffc_active:
            self.stats.skipped_ffc += 1
            return []
        if self.pose_buffer is not None:
            pose = self.pose_buffer.at(frame.timestamp)
            if pose is None:
                self.stats.skipped_stale_pose += 1
                return []
            frame = replace(frame, pose=pose)

        agl = self.agl_m(frame, ctx)
        gsd = frame.camera.ground_sample_distance_m(agl)
        if isinstance(self.detector, FusedDetector):
            detections = self.detector.detect(frame, gsd_m=gsd, exclusions=self._exclusions(frame, ctx, gsd),
                                              saturation_c=ctx.saturation_c)
        else:
            detections = self.detector.detect(frame)

        rtk_ok = ctx.gnss_rtk_ok
        if not rtk_ok:
            self.stats.rtk_lost_frames += 1
        pos_w = 1.0 if rtk_ok else self.rtk_lost_pos_weight
        ifov = 2 * math.tan(math.radians(frame.camera.hfov_deg) / 2) / frame.camera.width_px  # rad-ish per pixel
        cam = frame.pose.position
        obs: list[Observation] = []
        hits: list[tuple[Position, float]] = []
        for d in detections:
            pos = self.georef.pixel_to_world(d.u, d.v, frame.pose, frame.camera)
            if pos is None:
                continue
            slant = math.dist((cam.x, cam.y, cam.z), (pos.x, pos.y, pos.z))
            px_m = slant * ifov  # ground size of one pixel at this range (nadir-ish approximation)
            obs.append(Observation(position=pos, confidence=d.confidence, peak_temp_c=d.peak_temp_c,
                                   area_m2=d.area_px * px_m ** 2, timestamp=frame.timestamp, pos_weight=pos_w))
            hits.append((pos, d.confidence))
        if self.map is not None:
            extra = 0.0 if rtk_ok else self.rtk_lost_extra_gate_m
            self.map.update(footprint_corners(frame.pose, frame.camera, self.georef), hits, frame.timestamp,
                            hit_radius_m=self.map.hit_radius_m + extra)
        gate = self.tracker.gate_m + (0.0 if rtk_ok else self.rtk_lost_extra_gate_m)
        return self.tracker.update(obs, gate_m=gate)
