"""Safe refill planning and execution for one drone."""

from __future__ import annotations

from dataclasses import dataclass

from contracts.entities import (
    DroneState,
    RefillStation,
    WaterSource,
)
from contracts.enums import (
    DroneKind,
    ExtinguishingAgent,
    WaterSourceStatus,
)
from contracts.geometry import Position
from contracts.interfaces import Autopilot, PayloadSystem


@dataclass(frozen=True)
class RefillDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class RefillExecutionResult:
    status: str
    success: bool
    water_added_l: float
    reason: str


def validate_water_source(source: WaterSource) -> RefillDecision:
    """Check whether a natural water source is currently usable."""

    if source.status is not WaterSourceStatus.VALIDATED:
        return RefillDecision(
            allowed=False,
            reason="Water source is not validated",
        )

    if source.confidence <= 0:
        return RefillDecision(
            allowed=False,
            reason="Water source has no confidence",
        )

    if source.est_volume_l is not None and source.est_volume_l <= 0:
        return RefillDecision(
            allowed=False,
            reason="Water source has no estimated usable volume",
        )

    return RefillDecision(
        allowed=True,
        reason="Water source is validated",
    )


def validate_refill_station(
    station: RefillStation,
    agent: ExtinguishingAgent,
) -> RefillDecision:
    """Check whether a refill station can currently service the drone."""

    if station.free_bays <= 0:
        return RefillDecision(
            allowed=False,
            reason="No refill bay is available",
        )

    if station.water_available_l <= 0:
        return RefillDecision(
            allowed=False,
            reason="Refill station has no water available",
        )

    if agent not in station.agents_available:
        return RefillDecision(
            allowed=False,
            reason="Requested extinguishing agent is unavailable",
        )

    return RefillDecision(
        allowed=True,
        reason="Refill station is available",
    )


def execute_water_refill(
    drone: DroneState,
    autopilot: Autopilot,
    payload: PayloadSystem,
    target_position: Position,
    requested_volume_l: float,
    *,
    source: WaterSource | None = None,
    station: RefillStation | None = None,
) -> RefillExecutionResult:
    """
    Execute the autonomy side of a water refill.

    The simulator/backend remains responsible for authoritative station/source
    inventory. This function only navigates, validates and calls PayloadSystem.
    """

    if drone.kind is not DroneKind.EXTINGUISHER:
        return RefillExecutionResult(
            status="REJECTED",
            success=False,
            water_added_l=0.0,
            reason="Only extinguisher drones may refill water",
        )

    if requested_volume_l <= 0:
        return RefillExecutionResult(
            status="REJECTED",
            success=False,
            water_added_l=0.0,
            reason="Requested refill volume must be positive",
        )

    if source is None and station is None:
        return RefillExecutionResult(
            status="REJECTED",
            success=False,
            water_added_l=0.0,
            reason="A water source or refill station is required",
        )

    if source is not None:
        decision = validate_water_source(source)
        if not decision.allowed:
            return RefillExecutionResult(
                status="REJECTED",
                success=False,
                water_added_l=0.0,
                reason=decision.reason,
            )

    if station is not None:
        decision = validate_refill_station(
            station,
            drone.agent,
        )
        if not decision.allowed:
            return RefillExecutionResult(
                status="REJECTED",
                success=False,
                water_added_l=0.0,
                reason=decision.reason,
            )

    autopilot.goto(target_position)

    if not autopilot.at_target():
        return RefillExecutionResult(
            status="TRANSIT",
            success=False,
            water_added_l=0.0,
            reason="Drone is still travelling to refill point",
        )

    actual_l = payload.fill(
        requested_volume_l,
        agent=drone.agent,
    )

    if actual_l < 0:
        raise ValueError("PayloadSystem.fill returned a negative volume")

    if actual_l > requested_volume_l + 1e-6:
        raise ValueError(
            "PayloadSystem.fill returned more water than requested"
        )

    if actual_l == 0:
        return RefillExecutionResult(
            status="FAILED",
            success=False,
            water_added_l=0.0,
            reason="Payload system added no water",
        )

    autopilot.hold()

    return RefillExecutionResult(
        status="DONE",
        success=True,
        water_added_l=actual_l,
        reason="Water refill completed",
    )
