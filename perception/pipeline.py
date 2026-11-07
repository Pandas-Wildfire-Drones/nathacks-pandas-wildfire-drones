"""On-drone perception loop: one ThermalFrame in -> changed Hotspots out (+ map deltas kept for MapSync).

Workstream B (drone autonomy) owns the loop that calls `process(frame)` each camera tick and publishes the
returned hotspots as HotspotReport; `map.pop_updates()` feeds MapSync. Swap the detector freely:
FakeDetector (day one) -> ThresholdDetector -> FusedDetector(OnnxCnnClassifier(...)).
"""
from __future__ import annotations

from contracts import Detector, GridSpec, Hotspot, Position, ThermalFrame

from .georef import FlatGroundGeoreferencer, footprint_corners
from .mapping import LogOddsMap
from .tracking import HotspotTracker, Observation

__all__ = ["SeekerPerception"]


class SeekerPerception:
    def __init__(self, drone_id: str, detector: Detector, grid: GridSpec | None = None, georef=None,
                 tracker: HotspotTracker | None = None, ground_z: float = 0.0):
        self.drone_id = drone_id
        self.detector = detector
        self.georef = georef or FlatGroundGeoreferencer(ground_z)
        self.tracker = tracker or HotspotTracker(id_prefix=drone_id)
        self.map = LogOddsMap(grid) if grid is not None else None
        self.ground_z = ground_z

    def process(self, frame: ThermalFrame) -> list[Hotspot]:
        detections = self.detector.detect(frame)
        agl = max(frame.pose.position.z - self.ground_z, 0.1)
        px_area_m2 = frame.camera.ground_sample_distance_m(agl) ** 2
        obs: list[Observation] = []
        hits: list[tuple[Position, float]] = []
        for d in detections:
            pos = self.georef.pixel_to_world(d.u, d.v, frame.pose, frame.camera)
            if pos is None:
                continue
            obs.append(Observation(position=pos, confidence=d.confidence, peak_temp_c=d.peak_temp_c,
                                   area_m2=d.area_px * px_area_m2, timestamp=frame.timestamp))
            hits.append((pos, d.confidence))
        if self.map is not None:
            self.map.update(footprint_corners(frame.pose, frame.camera, self.georef), hits, frame.timestamp)
        return self.tracker.update(obs)
