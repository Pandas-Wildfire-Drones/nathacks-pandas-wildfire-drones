"""Drone health and fail-safe decisions."""

from __future__ import annotations

from dataclasses import dataclass

from contracts.entities import DroneState
from contracts.enums import DroneStatus


@dataclass(frozen=True)
class HealthDecision:
    action: str
    safe: bool
    reason: str


def evaluate_drone_health(
    drone: DroneState,
    *,
    minimum_link_quality: float = 0.20,
) -> HealthDecision:
    """
    Decide the immediate autonomy response to drone health state.

    This does not mutate DroneState. The simulator/backend remains
    authoritative for the actual drone state.
    """

    if not 0 <= minimum_link_quality <= 1:
        raise ValueError("minimum_link_quality must be between 0 and 1")

    # Lost is the strongest health failure: autonomy must not continue
    # mission execution.
    if drone.status is DroneStatus.LOST:
        return HealthDecision(
            action="LOST",
            safe=False,
            reason="Drone is marked LOST",
        )

    # A fault means the drone has an internal failure. The controller
    # should not continue mission actions.
    if drone.status is DroneStatus.FAULT:
        return HealthDecision(
            action="RETURN_OR_HOLD",
            safe=False,
            reason="Drone is in FAULT state",
        )

    # sensors_ok is a dictionary keyed by subsystem/sensor name.
    # An empty dictionary means no sensor failure has been reported.
    # Any explicitly reported failed sensor blocks mission execution.
    failed_sensors = [
        name
        for name, healthy in drone.sensors_ok.items()
        if not healthy
    ]

    if failed_sensors:
        return HealthDecision(
            action="HOLD",
            safe=False,
            reason=f"Unhealthy sensors: {', '.join(sorted(failed_sensors))}",
        )

    # If link quality is unavailable, we cannot establish that
    # communication is healthy enough for continued autonomous operation.
    if drone.link_quality is None:
        return HealthDecision(
            action="RETURN",
            safe=False,
            reason="Communication link quality is unavailable",
        )

    if drone.link_quality < minimum_link_quality:
        return HealthDecision(
            action="RETURN",
            safe=False,
            reason="Communication link quality is too low",
        )

    return HealthDecision(
        action="CONTINUE",
        safe=True,
        reason="Drone health is acceptable",
    )
