"""Per-frame sensor context + pose time-sync. Perception-local: no contract changes (docs/decisions/0002).

FrameContext carries what the thermal frame alone doesn't: altimeter reading, sensor health flags (from the same
sources as DroneState.sensors_ok), camera housekeeping (gain mode saturation, flat-field-correction in progress)
and positions that should not get the hot-blob bypass (friendly drones from heartbeats, known industrial heat).

PoseBuffer fixes pose/frame time mismatch. The autopilot publishes pose at 50-200 Hz; the camera at 9 Hz with
some latency. Using "the latest pose" instead of "the pose at exposure time" costs speed x latency: 15 m/s x 50 ms
= 0.75 m, plus attitude error while turning. ThermalFrame.timestamp is defined as the EXPOSURE time.
"""
from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass

from contracts import Pose, Position

__all__ = ["FrameContext", "PoseBuffer"]


@dataclass(frozen=True)
class FrameContext:
    agl_m: float | None = None              # measured height above ground (altimeter); None = use DEM
    gnss_rtk_ok: bool = True                # False -> wider association gate, lower position weight
    thermal_ok: bool = True                 # False -> skip frame (and record no "searched, empty" evidence)
    ffc_active: bool = False                # flat-field correction in progress: frame is frozen -> skip
    saturation_c: float | None = None       # upper limit of the CURRENT gain mode (hot-blob bypass trigger)
    friendly_positions: tuple[Position, ...] = ()   # other drones (from heartbeats): no bypass near them
    known_hot_sources: tuple[Position, ...] = ()    # e.g. gas flares from a facility map: no bypass near them


def _lerp_angle(a: float, b: float, t: float) -> float:
    d = (b - a + 180.0) % 360.0 - 180.0
    return (a + t * d) % 360.0


class PoseBuffer:
    """Time-ordered pose samples with interpolation. Deterministic; time comes from the caller (hard rule 5)."""

    def __init__(self, maxlen: int = 4000, max_gap_s: float = 0.25, max_extrapolation_s: float = 0.05):
        self._t: deque[float] = deque(maxlen=maxlen)
        self._p: deque[Pose] = deque(maxlen=maxlen)
        self.max_gap_s, self.max_extrapolation_s = max_gap_s, max_extrapolation_s

    def add(self, t: float, pose: Pose) -> None:
        if self._t and t <= self._t[-1]:
            return  # ignore out-of-order / duplicate samples
        self._t.append(t)
        self._p.append(pose)

    def at(self, t: float) -> Pose | None:
        """Pose at time t, or None if t is outside the buffer (beyond max_extrapolation_s) or in a gap > max_gap_s."""
        if not self._t:
            return None
        ts = self._t
        if t <= ts[0]:
            return self._p[0] if ts[0] - t <= self.max_extrapolation_s else None
        if t >= ts[-1]:
            return self._p[-1] if t - ts[-1] <= self.max_extrapolation_s else None
        i = bisect.bisect_right(ts, t)
        t0, t1 = ts[i - 1], ts[i]
        if t1 - t0 > self.max_gap_s:
            return None
        a, b = self._p[i - 1], self._p[i]
        w = (t - t0) / (t1 - t0)
        pa, pb = a.position, b.position
        return Pose(position=Position(x=pa.x + w * (pb.x - pa.x), y=pa.y + w * (pb.y - pa.y), z=pa.z + w * (pb.z - pa.z)),
                    yaw_deg=_lerp_angle(a.yaw_deg, b.yaw_deg, w),
                    pitch_deg=a.pitch_deg + w * (b.pitch_deg - a.pitch_deg),
                    roll_deg=a.roll_deg + w * (b.roll_deg - a.roll_deg))
