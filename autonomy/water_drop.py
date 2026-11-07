"""Safety checks for releasing water on a confirmed wildfire hotspot."""

from dataclasses import dataclass
from math import isfinite

from contracts.entities import DroneState, Hotspot, Task
from contracts.enums import (
    DroneKind,
    HotspotStatus,
    TaskStatus,
    TaskType,
)


@dataclass(frozen=True)
class WaterDropLimits:
    """Configurable release limits. Altitude is explicitly measured AGL."""

    min_confidence: float
    max_distance_m: float
    min_altitude_agl_m: float
    max_altitude_agl_m: float
    max_speed_mps: float

    def __post_init__(self):
        values = (
            self.min_confidence,
            self.max_distance_m,
            self.min_altitude_agl_m,
            self.max_altitude_agl_m,
            self.max_speed_mps,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Drop limits must be finite")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("Minimum confidence must be between 0 and 1")
        if any(value < 0 for value in values[1:]):
            raise ValueError("Drop limits cannot be negative")
        if self.min_altitude_agl_m > self.max_altitude_agl_m:
            raise ValueError("Minimum altitude exceeds maximum altitude")


@dataclass(frozen=True)
class DropDecision:
    should_drop: bool
    water_to_release_liters: float
    reason: str


def _reject(reason: str) -> DropDecision:
    return DropDecision(False, 0.0, reason)


def evaluate_water_drop(
    drone: DroneState,
    hotspot: Hotspot,
    task: Task,
    altitude_agl_m: float,
    limits: WaterDropLimits,
) -> DropDecision:
    """Evaluate whether this drone may release water on this task's hotspot."""

    if not isfinite(altitude_agl_m) or altitude_agl_m < 0:
        raise ValueError("AGL altitude must be finite and non-negative")

    if drone.kind is not DroneKind.EXTINGUISHER:
        return _reject("Only extinguisher drones may release water")

    if hotspot.status not in (HotspotStatus.CONFIRMED, HotspotStatus.ASSIGNED):
        return _reject("Hotspot has not been confirmed")

    if task.type is not TaskType.EXTINGUISH:
        return _reject("Task is not an extinguishing task")

    if task.target_hotspot_id != hotspot.hotspot_id:
        return _reject("Task target does not match the supplied hotspot")

    if task.status not in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
        return _reject("Task is not active")

    if task.assigned_to != drone.drone_id:
        return _reject("Task is not assigned to this drone")

    if hotspot.confidence < limits.min_confidence:
        return _reject("Target confidence too low")

    distance_m = drone.position.distance_to(
        hotspot.position,
        horizontal=True,
    )
    if distance_m > limits.max_distance_m:
        return _reject("Drone is too far from target")

    if not limits.min_altitude_agl_m <= altitude_agl_m <= limits.max_altitude_agl_m:
        return _reject("AGL altitude outside release limits")

    if drone.speed_mps > limits.max_speed_mps:
        return _reject("Drone is moving too fast")

    water_available = drone.water_remaining_l or 0.0
    if water_available <= 0:
        return _reject("No water available")

    requested = task.volume_l if task.volume_l is not None else water_available
    amount = min(water_available, requested)

    if amount <= 0:
        return _reject("Requested drop must be positive")

    return DropDecision(
        should_drop=True,
        water_to_release_liters=amount,
        reason="Release conditions satisfied",
    )
