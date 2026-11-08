"""Create and publish time-bounded 3D path claims."""

from __future__ import annotations

from math import isfinite

from contracts.entities import DroneState
from contracts.geometry import Position
from contracts.interfaces import Bus
from contracts.messages import PathClaim, SwarmMessage
from contracts.topics import topic_for


class PathClaimManager:
    """Creates path reservations without mutating authoritative drone state."""

    def __init__(self, bus: Bus):
        self.bus = bus
        self._seq = 0

    def create_claim(
        self,
        drone: DroneState,
        waypoints: list[Position],
        *,
        valid_from: float,
        valid_until: float,
        altitude_band_m: tuple[float, float] | None = None,
    ) -> SwarmMessage:
        if not waypoints:
            raise ValueError("Path claim requires at least one waypoint")

        if valid_until <= valid_from:
            raise ValueError(
                "Path claim valid_until must be greater than valid_from"
            )

        if altitude_band_m is not None:
            lower, upper = altitude_band_m

            if lower >= upper:
                raise ValueError(
                    "Path claim altitude band must have lower < upper"
                )

        self._seq += 1

        claim = PathClaim(
            drone_id=drone.drone_id,
            waypoints=waypoints,
            valid_from=valid_from,
            valid_until=valid_until,
            altitude_band_m=altitude_band_m,
        )

        return SwarmMessage.create(
            sender_id=drone.drone_id,
            body=claim,
            now=valid_from,
            seq=self._seq,
        )

    def publish_claim(
        self,
        drone: DroneState,
        waypoints: list[Position],
        *,
        valid_from: float,
        valid_until: float,
        altitude_band_m: tuple[float, float] | None = None,
    ) -> SwarmMessage:
        message = self.create_claim(
            drone,
            waypoints,
            valid_from=valid_from,
            valid_until=valid_until,
            altitude_band_m=altitude_band_m,
        )

        self.bus.publish(
            topic_for(message),
            message,
        )

        return message

    def create_from_planned_path(
        self,
        drone: DroneState,
        waypoints: list[Position],
        *,
        valid_from: float,
        time_step_s: float,
        altitude_band_m: tuple[float, float] | None = None,
    ) -> SwarmMessage:
        """Create a PathClaim from a CBS-planned route.

        The claim window runs from valid_from for the planned duration,
        (len(waypoints) - 1) * time_step_s. A one-waypoint route (the drone
        is already at its goal) still gets one planning step, because
        create_claim requires valid_until > valid_from.
        """

        if not waypoints:
            raise ValueError("waypoints must not be empty")

        if not isfinite(time_step_s) or time_step_s <= 0:
            raise ValueError(
                "time_step_s must be finite and greater than zero"
            )

        duration_s = max(1, len(waypoints) - 1) * time_step_s

        return self.create_claim(
            drone,
            waypoints,
            valid_from=valid_from,
            valid_until=valid_from + duration_s,
            altitude_band_m=altitude_band_m,
        )

    def publish_from_planned_path(
        self,
        drone: DroneState,
        waypoints: list[Position],
        *,
        valid_from: float,
        time_step_s: float,
        altitude_band_m: tuple[float, float] | None = None,
    ) -> SwarmMessage:
        """Create a claim from a planned path and publish it on the bus."""

        message = self.create_from_planned_path(
            drone,
            waypoints,
            valid_from=valid_from,
            time_step_s=time_step_s,
            altitude_band_m=altitude_band_m,
        )

        self.bus.publish(
            topic_for(message),
            message,
        )

        return message