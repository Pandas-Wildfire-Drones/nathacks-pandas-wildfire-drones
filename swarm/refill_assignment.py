"""Nearest-safe-resource assignment for water and fuel refill requests."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from contracts import (
    Bus,
    Clock,
    Position,
    RefillGrant,
    RefillRequest,
    RefillStation,
    RefillTargetKind,
    Subscription,
    SwarmMessage,
    WaterSource,
    WaterSourceStatus,
)

from .communication import SwarmCommunicator

__all__ = ["RefillAssignmentCoordinator", "RefillDecision"]


@dataclass(frozen=True)
class RefillDecision:
    """A grant sent to a drone, or a reason no safe resource was available."""

    drone_id: str
    grant: RefillGrant | None
    reason: str | None = None


class RefillAssignmentCoordinator:
    """Assign requests to validated water sources or compatible refill stations.

    A request needing fuel is restricted to stations. Station bays are reserved
    until the caller invokes ``release_reservation`` after refill completion or
    cancellation; the swarm message contract has no refill-complete message.
    """

    def __init__(
        self,
        bus: Bus,
        clock: Clock,
        sender_id: str,
        stations: Iterable[RefillStation],
        water_sources: Iterable[WaterSource],
        *,
        epoch: int,
    ) -> None:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        self._clock = clock
        self._communicator = SwarmCommunicator(bus, clock, sender_id)
        self._epoch = epoch
        self._stations = {station.station_id: station.model_copy(deep=True) for station in stations}
        self._sources = {source.source_id: source.model_copy(deep=True) for source in water_sources}
        self._reservations: dict[str, tuple[RefillTargetKind, str, int | None]] = {}
        self._decisions: list[RefillDecision] = []
        self._subscription: Subscription | None = None

    @property
    def decisions(self) -> tuple[RefillDecision, ...]:
        return tuple(self._decisions)

    def start(self) -> None:
        """Subscribe to refill requests once."""
        if self._subscription is not None:
            raise RuntimeError("refill coordinator is already started")
        self._subscription = self._communicator.subscribe(
            "swarm/refill/requests",
            self._on_message,
        )

    def update_resources(
        self,
        stations: Iterable[RefillStation],
        water_sources: Iterable[WaterSource],
    ) -> None:
        """Replace the latest resource picture without changing current reservations."""
        self._stations = {station.station_id: station.model_copy(deep=True) for station in stations}
        self._sources = {source.source_id: source.model_copy(deep=True) for source in water_sources}

    def update_epoch(self, epoch: int) -> None:
        """Adopt a newer leader epoch; reject stale refill requests afterward."""
        if epoch < self._epoch:
            raise ValueError("epoch cannot move backward")
        self._epoch = epoch

    def _now(self) -> float:
        now = self._clock.now()
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite time")
        return now

    def _on_message(self, message: SwarmMessage) -> None:
        now = self._now()
        if message.is_expired(now) or not isinstance(message.body, RefillRequest):
            return
        request = message.body
        if request.drone_id != message.sender_id:
            return
        if message.epoch is not None and message.epoch < self._epoch:
            return
        self.assign(request)

    def assign(self, request: RefillRequest) -> RefillDecision:
        """Choose the closest eligible resource, publish a directed grant, and reserve it."""
        existing = self._reservations.get(request.drone_id)
        if existing is not None:
            kind, target_id, bay = existing
            position = self._resource_position(kind, target_id)
            if position is not None:
                grant = RefillGrant(
                    drone_id=request.drone_id,
                    target_kind=kind,
                    target_id=target_id,
                    position=position,
                    bay=bay,
                    epoch=self._epoch,
                )
                self._publish_grant(grant)
                decision = RefillDecision(request.drone_id, grant)
                self._decisions.append(decision)
                return decision
        options: list[tuple[float, str, RefillTargetKind, Position, int | None]] = []
        needs_water = request.water_remaining_l <= 0.0
        for station in self._stations.values():
            if station.free_bays <= 0:
                continue
            reserved_bays = {
                bay
                for kind, resource_id, bay in self._reservations.values()
                if (
                    kind is RefillTargetKind.STATION
                    and resource_id == station.station_id
                    and bay is not None
                )
            }
            if len(reserved_bays) >= station.free_bays:
                continue
            if request.agent not in station.agents_available:
                continue
            if needs_water and station.water_available_l <= 0:
                continue
            if request.needs_fuel and station.fuel_available_kg <= 0:
                continue
            available_bays = [
                bay for bay in range(station.total_bays)
                if bay not in reserved_bays
            ]
            if not available_bays:
                continue
            bay = min(available_bays)
            options.append((
                request.position.distance_to(station.position),
                station.station_id,
                RefillTargetKind.STATION,
                station.position,
                bay,
            ))

        if needs_water and not request.needs_fuel:
            for source in self._sources.values():
                if source.status is not WaterSourceStatus.VALIDATED:
                    continue
                if source.est_volume_l is not None and source.est_volume_l <= 0:
                    continue
                options.append((
                    request.position.distance_to(source.position),
                    source.source_id,
                    RefillTargetKind.WATER_SOURCE,
                    source.position,
                    None,
                ))

        if not options:
            reason = (
                "no compatible station with fuel and available capacity"
                if request.needs_fuel
                else "no compatible water source or refill station is available"
            )
            decision = RefillDecision(request.drone_id, None, reason)
            self._decisions.append(decision)
            return decision

        _, target_id, kind, position, bay = min(
            options,
            key=lambda option: (option[0], option[1], option[4] if option[4] is not None else -1),
        )
        grant = RefillGrant(
            drone_id=request.drone_id,
            target_kind=kind,
            target_id=target_id,
            position=position,
            bay=bay,
            epoch=self._epoch,
        )
        self._reservations[request.drone_id] = (kind, target_id, bay)
        self._publish_grant(grant)
        decision = RefillDecision(request.drone_id, grant)
        self._decisions.append(decision)
        return decision

    def assign_battery_swap(
        self,
        drone_id: str,
        position: Position,
    ) -> RefillDecision:
        """Assign a low-battery seeker to a station that offers battery swaps.

        ``RefillRequest`` has no battery-swap flag, so the leader invokes this
        method from its battery-health policy rather than inferring a swap from
        the water/fuel fields.
        """
        existing = self._reservations.get(drone_id)
        if existing is not None:
            kind, target_id, bay = existing
            station = self._stations.get(target_id) if kind is RefillTargetKind.STATION else None
            if station is not None and station.battery_swap_s is not None:
                grant = RefillGrant(
                    drone_id=drone_id,
                    target_kind=RefillTargetKind.STATION,
                    target_id=target_id,
                    position=station.position,
                    bay=bay,
                    epoch=self._epoch,
                )
                self._publish_grant(grant)
                decision = RefillDecision(drone_id, grant)
                self._decisions.append(decision)
                return decision
            decision = RefillDecision(drone_id, None, "drone already has a refill reservation")
            self._decisions.append(decision)
            return decision

        options: list[tuple[float, str, RefillStation, int]] = []
        for station in self._stations.values():
            if station.battery_swap_s is None:
                continue
            reserved_bays = {
                bay
                for kind, resource_id, bay in self._reservations.values()
                if (
                    kind is RefillTargetKind.STATION
                    and resource_id == station.station_id
                    and bay is not None
                )
            }
            if len(reserved_bays) >= station.free_bays:
                continue
            available_bays = [
                bay for bay in range(station.total_bays)
                if bay not in reserved_bays
            ]
            if available_bays:
                options.append((
                    position.distance_to(station.position),
                    station.station_id,
                    station,
                    min(available_bays),
                ))

        if not options:
            decision = RefillDecision(drone_id, None, "no battery-swap station with an available bay")
            self._decisions.append(decision)
            return decision

        _, _, station, bay = min(options, key=lambda option: (option[0], option[1], option[3]))
        grant = RefillGrant(
            drone_id=drone_id,
            target_kind=RefillTargetKind.STATION,
            target_id=station.station_id,
            position=station.position,
            bay=bay,
            epoch=self._epoch,
        )
        self._reservations[drone_id] = (
            RefillTargetKind.STATION,
            station.station_id,
            bay,
        )
        self._publish_grant(grant)
        decision = RefillDecision(drone_id, grant)
        self._decisions.append(decision)
        return decision

    def _publish_grant(self, grant: RefillGrant) -> None:
        self._communicator.publish(
            grant,
            receiver_id=grant.drone_id,
            epoch=self._epoch,
        )

    def _resource_position(
        self,
        kind: RefillTargetKind,
        target_id: str,
    ) -> Position | None:
        if kind is RefillTargetKind.STATION:
            resource = self._stations.get(target_id)
        else:
            resource = self._sources.get(target_id)
        return None if resource is None else resource.position

    def release_reservation(self, drone_id: str) -> bool:
        """Release a station bay after refill completion; return whether one existed."""
        return self._reservations.pop(drone_id, None) is not None

    def close(self) -> None:
        if self._subscription is not None:
            self._subscription.cancel()
            self._subscription = None
