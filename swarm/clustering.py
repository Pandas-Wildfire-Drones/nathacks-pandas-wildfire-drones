"""Geographic cluster formation and cluster-local task coordination."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from contracts import (
    Bus,
    Clock,
    ClusterSnapshot,
    ClusterUpdate,
    DroneState,
    DroneStatus,
    Heartbeat,
    LeaderClaim,
    Position,
    SwarmMessage,
    Subscription,
    Task,
    Zone,
)

from .communication import SwarmCommunicator
from .leader_election import leader_score
from .task_lifecycle import TaskLifecycleCoordinator
from .zone_generation import generate_grid_zones

__all__ = [
    "ClusterFormation",
    "ClusterCoordinator",
    "form_clusters",
    "form_clusters_for_area",
    "elect_cluster_leader",
]


@dataclass(frozen=True)
class ClusterFormation:
    """Cluster snapshots plus copies of drones and zones annotated with ownership."""

    clusters: tuple[ClusterSnapshot, ...]
    drones: tuple[DroneState, ...]
    zones: tuple[Zone, ...]


def _polygon_center(zone: Zone) -> tuple[float, float]:
    """Return the arithmetic mean of polygon vertices as a stable assignment point."""
    return (
        sum(point.x for point in zone.boundary) / len(zone.boundary),
        sum(point.y for point in zone.boundary) / len(zone.boundary),
    )


def _distance_squared(a: Position, b: Position) -> float:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def form_clusters(
    drones: Iterable[DroneState],
    zones: Iterable[Zone],
    *,
    max_members: int,
    now: float,
    lease_s: float = 10.0,
) -> ClusterFormation:
    """Partition healthy drones into balanced geographic clusters.

    Farthest-point seeding spreads cluster centers spatially; capacity-limited
    nearest-center assignment then bounds membership. Zones are assigned to their
    nearest center. Fire urgency remains a task-ordering concern, not a boundary
    mutation signal.
    """
    drone_list = sorted(drones, key=lambda drone: drone.drone_id)
    zone_list = sorted(zones, key=lambda zone: zone.zone_id)
    if max_members < 1:
        raise ValueError("max_members must be >= 1")
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    if not math.isfinite(lease_s) or lease_s <= 0 or not math.isfinite(now + lease_s):
        raise ValueError("lease_s must be finite, positive, and produce a finite lease")
    if len({drone.drone_id for drone in drone_list}) != len(drone_list):
        raise ValueError("drone IDs must be unique")
    if len({zone.zone_id for zone in zone_list}) != len(zone_list):
        raise ValueError("zone IDs must be unique")

    viable = [
        drone for drone in drone_list
        if drone.status not in (DroneStatus.FAULT, DroneStatus.LOST)
    ]
    if not viable:
        raise ValueError("cannot form clusters without a viable drone")

    cluster_count = math.ceil(len(viable) / max_members)
    centers = [viable[0].position]
    seed_ids = {viable[0].drone_id}
    while len(centers) < cluster_count:
        next_drone = max(
            (drone for drone in viable if drone.drone_id not in seed_ids),
            key=lambda drone: (
                min(_distance_squared(drone.position, center) for center in centers),
                drone.drone_id,
            ),
        )
        centers.append(next_drone.position)
        seed_ids.add(next_drone.drone_id)

    capacities = [max_members] * cluster_count
    groups: list[list[DroneState]] = [[] for _ in centers]
    candidate_pairs = sorted(
        (
            _distance_squared(drone.position, center),
            drone.drone_id,
            cluster_index,
        )
        for drone in viable
        for cluster_index, center in enumerate(centers)
    )
    drone_by_id = {drone.drone_id: drone for drone in viable}
    assigned: set[str] = set()
    for _, drone_id, cluster_index in candidate_pairs:
        if drone_id in assigned or capacities[cluster_index] == 0:
            continue
        drone = drone_by_id[drone_id]
        groups[cluster_index].append(drone)
        capacities[cluster_index] -= 1
        assigned.add(drone_id)

    cluster_ids = [f"cluster-{index + 1:04d}" for index in range(cluster_count)]
    zone_groups: list[list[Zone]] = [[] for _ in centers]
    for zone in zone_list:
        cx, cy = _polygon_center(zone)
        cluster_index = min(
            range(cluster_count),
            key=lambda index: (
                (centers[index].x - cx) ** 2 + (centers[index].y - cy) ** 2,
                cluster_ids[index],
            ),
        )
        zone_groups[cluster_index].append(zone)

    snapshots: list[ClusterSnapshot] = []
    updated_drones: list[DroneState] = []
    updated_zones: list[Zone] = []
    for index, group in enumerate(groups):
        if not group:
            continue
        cluster_id = cluster_ids[index]
        cluster_epoch = max(drone.cluster_epoch for drone in group) + 1
        leader = max(group, key=lambda drone: (leader_score(drone), drone.drone_id))
        member_ids = sorted(drone.drone_id for drone in group)
        snapshots.append(
            ClusterSnapshot(
                cluster_id=cluster_id,
                leader_id=leader.drone_id,
                cluster_epoch=cluster_epoch,
                member_ids=member_ids,
                zone_ids=sorted(zone.zone_id for zone in zone_groups[index]),
                updated_at=now,
                lease_until=now + lease_s,
            )
        )
        updated_drones.extend(
            drone.model_copy(
                update={"cluster_id": cluster_id, "cluster_epoch": cluster_epoch},
                deep=True,
            )
            for drone in group
        )
        updated_zones.extend(
            zone.model_copy(update={"cluster_id": cluster_id}, deep=True)
            for zone in zone_groups[index]
        )

    updated_drones.extend(
        drone.model_copy(update={"cluster_id": None}, deep=True)
        for drone in drone_list
        if drone.status in (DroneStatus.FAULT, DroneStatus.LOST)
    )
    updated_drones.sort(key=lambda drone: drone.drone_id)
    updated_zones.sort(key=lambda zone: zone.zone_id)
    return ClusterFormation(tuple(snapshots), tuple(updated_drones), tuple(updated_zones))


def form_clusters_for_area(
    drones: Iterable[DroneState],
    southwest: Position,
    northeast: Position,
    *,
    max_members: int,
    zone_width_m: float,
    zone_height_m: float | None = None,
    now: float,
    lease_s: float = 10.0,
    zone_id_prefix: str = "zone",
) -> ClusterFormation:
    """Generate rectangular search zones, then assign zones and drones to clusters."""
    zones = generate_grid_zones(
        southwest,
        northeast,
        zone_width_m=zone_width_m,
        zone_height_m=zone_height_m,
        zone_id_prefix=zone_id_prefix,
    )
    return form_clusters(
        drones,
        zones,
        max_members=max_members,
        now=now,
        lease_s=lease_s,
    )


def elect_cluster_leader(
    snapshot: ClusterSnapshot,
    candidates: Iterable[DroneState],
    *,
    now: float,
    lease_s: float = 10.0,
    current_epoch: int = 0,
    exclude_drone_id: str | None = None,
) -> tuple[LeaderClaim, ClusterSnapshot]:
    """Choose a viable cluster member and issue a newer cluster term."""
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    if not math.isfinite(lease_s) or lease_s <= 0 or not math.isfinite(now + lease_s):
        raise ValueError("lease_s must be finite, positive, and produce a finite lease")
    if current_epoch < 0:
        raise ValueError("current_epoch must be >= 0")
    eligible = [
        drone for drone in candidates
        if drone.drone_id in snapshot.member_ids
        and drone.drone_id != exclude_drone_id
        and drone.status not in (DroneStatus.FAULT, DroneStatus.LOST)
    ]
    if not eligible:
        raise ValueError("cannot elect a cluster leader without a viable member")
    winner = max(eligible, key=lambda drone: (leader_score(drone), drone.drone_id))
    next_cluster_epoch = snapshot.cluster_epoch + 1
    next_global_epoch = max(current_epoch, *(drone.epoch for drone in eligible)) + 1
    claim = LeaderClaim(
        candidate_id=winner.drone_id,
        epoch=next_global_epoch,
        score=leader_score(winner),
        cluster_id=snapshot.cluster_id,
        cluster_epoch=next_cluster_epoch,
    )
    updated = snapshot.model_copy(
        update={
            "leader_id": winner.drone_id,
            "cluster_epoch": next_cluster_epoch,
            "updated_at": now,
            "lease_until": now + lease_s,
        },
        deep=True,
    )
    return claim, updated


class ClusterCoordinator:
    """Run one cluster's task lifecycle and accept versioned membership updates.

    Instantiate one coordinator per cluster leader/service. Heartbeats are
    subscribed per member ID, avoiding an all-clusters subscription to the full
    drone heartbeat stream.
    """

    def __init__(
        self,
        bus: Bus,
        clock: Clock,
        sender_id: str,
        snapshot: ClusterSnapshot,
        zones: Iterable[Zone],
        target_positions: Mapping[str, Position],
        *,
        epoch: int,
        local_state_provider: Callable[[], DroneState] | None = None,
        leader_timeout_s: float = 3.0,
        bid_window_s: float = 2.0,
    ) -> None:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        if not math.isfinite(leader_timeout_s) or leader_timeout_s <= 0:
            raise ValueError("leader_timeout_s must be finite and > 0")
        self._bus = bus
        self._clock = clock
        self._sender_id = sender_id
        self._snapshot = snapshot.model_copy(deep=True)
        self._all_zones = tuple(zones)
        self._target_positions = dict(target_positions)
        self._zones = self._zones_for_snapshot(snapshot)
        self._epoch = epoch
        self._local_state_provider = local_state_provider
        self._leader_timeout_s = leader_timeout_s
        self._communicator = SwarmCommunicator(bus, clock, sender_id)
        self._states: dict[str, DroneState] = {}
        self._received_at: dict[str, float] = {}
        self._subscriptions: list[Subscription] = []
        self._started = False
        self._lifecycle = self._make_lifecycle(bid_window_s)

    def _zones_for_snapshot(self, snapshot: ClusterSnapshot) -> tuple[Zone, ...]:
        zone_ids = set(snapshot.zone_ids)
        if zone_ids:
            return tuple(zone for zone in self._all_zones if zone.zone_id in zone_ids)
        return tuple(
            zone for zone in self._all_zones
            if zone.cluster_id == snapshot.cluster_id
        )

    @property
    def snapshot(self) -> ClusterSnapshot:
        return self._snapshot.model_copy(deep=True)

    @property
    def member_states(self) -> tuple[DroneState, ...]:
        return tuple(self._states[key].model_copy(deep=True) for key in sorted(self._states))

    @property
    def tasks(self) -> tuple[Task, ...]:
        return self._lifecycle.tasks

    @property
    def zones(self) -> tuple[Zone, ...]:
        return self._lifecycle.zones

    def _make_lifecycle(self, bid_window_s: float) -> TaskLifecycleCoordinator:
        return TaskLifecycleCoordinator(
            self._bus,
            self._clock,
            self._sender_id,
            self._target_positions,
            epoch=self._epoch,
            bid_window_s=bid_window_s,
            is_leader=self._snapshot.leader_id == self._sender_id,
            cluster_id=self._snapshot.cluster_id,
            cluster_epoch=self._snapshot.cluster_epoch,
            member_ids=self._snapshot.member_ids,
            zones=self._zones,
        )

    def start(self) -> None:
        if self._started:
            raise RuntimeError("cluster coordinator is already started")
        self._started = True
        self._refresh_local_state()
        self._lifecycle.start()
        self._subscriptions = [
            self._communicator.subscribe(
                f"swarm/clusters/{self._snapshot.cluster_id}/state",
                self._on_cluster_update,
            ),
            self._communicator.subscribe(
                f"swarm/clusters/{self._snapshot.cluster_id}/leader",
                self._on_leader_claim,
            ),
        ]
        self._subscribe_members()
        if self._snapshot.leader_id == self._sender_id:
            self.publish_snapshot()

    def _subscribe_members(self) -> None:
        for subscription in self._subscriptions[2:]:
            subscription.cancel()
        self._subscriptions = self._subscriptions[:2]
        for member_id in self._snapshot.member_ids:
            self._subscriptions.append(
                self._communicator.subscribe(
                    f"drone/{member_id}/heartbeat",
                    self._on_heartbeat,
                )
            )

    def _on_heartbeat(self, message: SwarmMessage) -> None:
        if message.is_expired(self._clock.now()) or not isinstance(message.body, Heartbeat):
            return
        state = message.body.state
        if (
            state.drone_id != message.sender_id
            or state.drone_id not in self._snapshot.member_ids
            or state.cluster_id != self._snapshot.cluster_id
            or state.cluster_epoch < self._snapshot.cluster_epoch
        ):
            return
        current = self._states.get(state.drone_id)
        if current is None or state.timestamp >= current.timestamp:
            self._states[state.drone_id] = state.model_copy(deep=True)
            self._received_at[state.drone_id] = self._clock.now()

    def _refresh_local_state(self) -> None:
        if self._local_state_provider is None:
            return
        state = self._local_state_provider()
        if state.drone_id != self._sender_id:
            raise ValueError("local state drone_id must match cluster coordinator sender_id")
        if state.cluster_id != self._snapshot.cluster_id:
            raise ValueError("local state cluster_id does not match the cluster snapshot")
        if state.cluster_epoch != self._snapshot.cluster_epoch:
            return
        self._states[state.drone_id] = state.model_copy(deep=True)
        self._received_at[state.drone_id] = self._clock.now()

    def _on_cluster_update(self, message: SwarmMessage) -> None:
        if message.is_expired(self._clock.now()) or not isinstance(message.body, ClusterUpdate):
            return
        incoming = message.body.cluster
        if (
            message.sender_id != incoming.leader_id
            or message.epoch is None
            or message.epoch < self._epoch
            or incoming.cluster_id != self._snapshot.cluster_id
            or incoming.cluster_epoch < self._snapshot.cluster_epoch
            or (
                incoming.cluster_epoch == self._snapshot.cluster_epoch
                and (
                    incoming.leader_id != self._snapshot.leader_id
                    or incoming.member_ids != self._snapshot.member_ids
                    or incoming.zone_ids != self._snapshot.zone_ids
                    or incoming.updated_at < self._snapshot.updated_at
                    or (
                        incoming.updated_at == self._snapshot.updated_at
                        and incoming.lease_until <= self._snapshot.lease_until
                    )
                )
            )
            or incoming.lease_until <= self._clock.now()
        ):
            return
        self._epoch = message.epoch
        self._accept_snapshot(incoming)

    def _on_leader_claim(self, message: SwarmMessage) -> None:
        if message.is_expired(self._clock.now()) or not isinstance(message.body, LeaderClaim):
            return
        claim = message.body
        if (
            claim.cluster_id != self._snapshot.cluster_id
            or claim.candidate_id != message.sender_id
            or message.epoch != claim.epoch
            or claim.epoch < self._epoch
            or claim.cluster_epoch <= self._snapshot.cluster_epoch
            or claim.candidate_id not in self._snapshot.member_ids
        ):
            return
        now = self._clock.now()
        updated = self._snapshot.model_copy(
            update={
                "leader_id": claim.candidate_id,
                "cluster_epoch": claim.cluster_epoch,
                "updated_at": now,
                "lease_until": now + 10.0,
            },
            deep=True,
        )
        self._epoch = claim.epoch
        self._accept_snapshot(updated)
        if claim.candidate_id == self._sender_id:
            self.publish_snapshot()

    def _accept_snapshot(self, snapshot: ClusterSnapshot) -> None:
        previous_zones = {
            zone.zone_id: zone
            for zone in self._lifecycle.zones
        }
        self._snapshot = snapshot.model_copy(deep=True)
        updated_zones: list[Zone] = []
        for zone in self._zones_for_snapshot(snapshot):
            previous = previous_zones.get(zone.zone_id)
            if previous is not None and previous.boundary == zone.boundary:
                zone = zone.model_copy(
                    update={
                        "assigned_to": previous.assigned_to,
                        "last_scanned": previous.last_scanned,
                    },
                    deep=True,
                )
            updated_zones.append(zone)
        self._zones = tuple(updated_zones)
        self._states = {
            member_id: state
            for member_id, state in self._states.items()
            if member_id in snapshot.member_ids
        }
        self._received_at = {
            member_id: received_at
            for member_id, received_at in self._received_at.items()
            if member_id in snapshot.member_ids
        }
        self._subscribe_members()
        self._lifecycle.update_cluster(
            leader_id=snapshot.leader_id,
            epoch=self._epoch,
            cluster_epoch=snapshot.cluster_epoch,
            member_ids=snapshot.member_ids,
            zones=self._zones,
        )

    def publish_snapshot(self) -> SwarmMessage:
        if self._snapshot.leader_id != self._sender_id:
            raise RuntimeError("only the current cluster leader can publish its snapshot")
        now = self._clock.now()
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite time")
        renewed = self._snapshot.model_copy(
            update={"updated_at": now, "lease_until": max(self._snapshot.lease_until, now + 10.0)},
            deep=True,
        )
        self._snapshot = renewed
        return self._communicator.publish(
            ClusterUpdate(cluster=renewed),
            epoch=self._epoch,
        )

    def elect_successor(self, *, lease_s: float = 10.0) -> ClusterSnapshot:
        """Elect from recently observed members and publish a new cluster term."""
        now = self._clock.now()
        self._refresh_local_state()
        claim, updated = self._successor_claim(now, lease_s)
        if claim is None or updated is None:
            raise ValueError("cannot elect a successor without a fresh viable member")
        if claim.candidate_id != self._sender_id:
            raise RuntimeError("only the elected successor may publish its leader claim")
        self._epoch = claim.epoch
        self._accept_snapshot(updated)
        self._communicator.publish(claim, epoch=claim.epoch)
        if updated.leader_id == self._sender_id:
            self.publish_snapshot()
        return self.snapshot

    def _successor_claim(
        self,
        now: float,
        lease_s: float,
    ) -> tuple[LeaderClaim | None, ClusterSnapshot | None]:
        fresh_states = [
            state for state in self._states.values()
            if now - self._received_at.get(state.drone_id, float("-inf"))
            < self._leader_timeout_s
        ]
        if not any(state.drone_id != self._snapshot.leader_id for state in fresh_states):
            return None, None
        return elect_cluster_leader(
            self._snapshot,
            fresh_states,
            now=now,
            lease_s=lease_s,
            current_epoch=self._epoch,
            exclude_drone_id=self._snapshot.leader_id,
        )

    def tick(self) -> tuple[Task, ...]:
        self._refresh_local_state()
        now = self._clock.now()
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite time")
        leader_last_seen = self._received_at.get(
            self._snapshot.leader_id,
            self._snapshot.updated_at,
        )
        leader_stale = now - leader_last_seen >= self._leader_timeout_s
        lease_expired = self._snapshot.lease_until <= now
        if leader_stale or lease_expired:
            claim, updated = self._successor_claim(now, max(10.0, self._snapshot.lease_until - now))
            if claim is not None and updated is not None and claim.candidate_id == self._sender_id:
                self._epoch = claim.epoch
                self._accept_snapshot(updated)
                self._communicator.publish(claim, epoch=claim.epoch)
                self.publish_snapshot()
        if self._snapshot.lease_until <= now:
            return ()
        return self._lifecycle.tick()

    def close(self) -> None:
        for subscription in self._subscriptions:
            subscription.cancel()
        self._subscriptions.clear()
        self._lifecycle.close()
        self._started = False
