"""Leader-side hotspot intake, task auctioning, and task status lifecycle."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from contracts import (
    Bus,
    Clock,
    Hotspot,
    HotspotReport,
    Position,
    SwarmMessage,
    Subscription,
    Task,
    TaskAck,
    TaskAnnounce,
    TaskAssign,
    TaskComplete,
    TaskStatus,
    TaskType,
    Zone,
)

from .communication import SwarmCommunicator
from .failure_handling import FailureReport
from .task_auction import TaskAuctionCoordinator
from .task_creation import create_tasks_from_hotspots
from .zone_generation import zone_center

__all__ = ["TaskLifecycleCoordinator"]


class TaskLifecycleCoordinator:
    """Create tasks from hotspot reports and run their full assign/ack/complete loop.

    The class owns task state for one leader. On leadership change, call
    ``update_leadership``; pending and active tasks remain available for the new
    leader, while any interrupted auction is safely returned to PENDING.
    """

    def __init__(
        self,
        bus: Bus,
        clock: Clock,
        sender_id: str,
        target_positions: Mapping[str, Position],
        *,
        epoch: int,
        bid_window_s: float = 2.0,
        ack_timeout_s: float = 5.0,
        retry_delay_s: float = 1.0,
        is_leader: bool = True,
        cluster_id: str | None = None,
        cluster_epoch: int = 0,
        member_ids: Iterable[str] | None = None,
        zones: Iterable[Zone] = (),
    ) -> None:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        if cluster_epoch < 0:
            raise ValueError("cluster_epoch must be >= 0")
        if not math.isfinite(ack_timeout_s) or ack_timeout_s <= 0:
            raise ValueError("ack_timeout_s must be finite and > 0")
        if not math.isfinite(retry_delay_s) or retry_delay_s < 0:
            raise ValueError("retry_delay_s must be finite and >= 0")
        self._bus = bus
        self._clock = clock
        self._sender_id = sender_id
        self._epoch = epoch
        self._cluster_id = cluster_id
        self._cluster_epoch = cluster_epoch
        self._member_ids = None if member_ids is None else tuple(sorted(set(member_ids)))
        self._zones = tuple(zone.model_copy(deep=True) for zone in zones)
        self._is_leader = is_leader
        self._ack_timeout_s = ack_timeout_s
        self._retry_delay_s = retry_delay_s
        self._target_positions = dict(target_positions)
        self._hotspots: dict[str, Hotspot] = {}
        self._hotspot_aliases: dict[str, str] = {}
        self._tasks: dict[str, Task] = {}
        self._retry_at: dict[str, float] = {}
        self._ack_deadline: dict[str, float] = {}
        self._active_task_id: str | None = None
        self._subscriptions: list[Subscription] = []
        self._started = False
        self._communicator = SwarmCommunicator(bus, clock, sender_id)
        self._bid_window_s = bid_window_s
        self._ensure_zone_tasks()
        self._auction = self._make_auction()

    def _ensure_zone_tasks(self) -> None:
        zones_by_id = {zone.zone_id: zone for zone in self._zones}
        for zone in self._zones:
            self._target_positions[zone.zone_id] = zone_center(zone)
            task_id = f"scan-zone-{zone.zone_id}"
            existing = self._tasks.get(task_id)
            if zone.last_scanned is not None:
                if existing is None:
                    self._tasks[task_id] = Task(
                        task_id=task_id,
                        type=TaskType.SCAN_ZONE,
                        status=TaskStatus.DONE,
                        target_zone_id=zone.zone_id,
                        created_at=zone.last_scanned,
                        cluster_id=self._cluster_id,
                        cluster_epoch=self._cluster_epoch,
                    )
                else:
                    existing.status = TaskStatus.DONE
                    existing.assigned_to = None
            elif existing is None:
                self._tasks[task_id] = Task(
                    task_id=task_id,
                    type=TaskType.SCAN_ZONE,
                    priority=0.5,
                    target_zone_id=zone.zone_id,
                    created_at=self._now(),
                    cluster_id=self._cluster_id,
                    cluster_epoch=self._cluster_epoch,
                )
        for task in self._tasks.values():
            if (
                task.type is TaskType.SCAN_ZONE
                and task.target_zone_id not in zones_by_id
                and task.status is TaskStatus.PENDING
            ):
                task.status = TaskStatus.CANCELLED

    def _update_zone_assignment(
        self,
        task: Task,
        *,
        drone_id: str | None,
        scanned_at: float | None = None,
    ) -> None:
        if task.type is not TaskType.SCAN_ZONE or task.target_zone_id is None:
            return
        for zone in self._zones:
            if zone.zone_id == task.target_zone_id:
                zone.assigned_to = drone_id
                if scanned_at is not None:
                    zone.last_scanned = scanned_at
                return

    def _make_auction(self) -> TaskAuctionCoordinator:
        return TaskAuctionCoordinator(
            self._bus,
            self._clock,
            self._sender_id,
            self._target_positions,
            epoch=self._epoch,
            bid_window_s=self._bid_window_s,
            cluster_id=self._cluster_id,
            cluster_epoch=self._cluster_epoch,
        )

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(self._tasks[key].model_copy(deep=True) for key in sorted(self._tasks))

    @property
    def zones(self) -> tuple[Zone, ...]:
        return tuple(zone.model_copy(deep=True) for zone in self._zones)

    @property
    def hotspots(self) -> tuple[Hotspot, ...]:
        return tuple(self._hotspots[key].model_copy(deep=True) for key in sorted(self._hotspots))

    @property
    def active_task(self) -> Task | None:
        task = self._auction.active_task
        return None if task is None else task.model_copy(deep=True)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("task lifecycle coordinator is already started")
        self._started = True
        hotspot_topics = (
            [f"drone/{member_id}/hotspots" for member_id in self._member_ids]
            if self._member_ids is not None
            else ["drone/*/hotspots"]
        )
        assignment_topics = (
            [f"drone/{member_id}/cmd" for member_id in self._member_ids]
            if self._member_ids is not None
            else ["drone/*/cmd"]
        )
        self._subscriptions = [
            *(self._communicator.subscribe(topic, self._on_hotspot) for topic in hotspot_topics),
            self._communicator.subscribe(
                (
                    f"swarm/clusters/{self._cluster_id}/tasks/announce"
                    if self._cluster_id is not None
                    else "swarm/tasks/announce"
                ),
                self._on_announce,
            ),
            *(self._communicator.subscribe(topic, self._on_assignment) for topic in assignment_topics),
            self._communicator.subscribe("swarm/tasks/acks", self._on_ack),
            self._communicator.subscribe("swarm/tasks/complete", self._on_complete),
        ]

    def _now(self) -> float:
        now = self._clock.now()
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite time")
        return now

    def _on_hotspot(self, message: SwarmMessage) -> None:
        if message.is_expired(self._now()) or not isinstance(message.body, HotspotReport):
            return
        hotspot = message.body.hotspot.model_copy(deep=True)
        if hotspot.reported_by is not None and hotspot.reported_by != message.sender_id:
            return
        if self._member_ids is not None and message.sender_id not in self._member_ids:
            return
        if self._zones and not any(
            zone.contains(hotspot.position)
            for zone in self._zones
            if zone.cluster_id in (None, self._cluster_id)
        ):
            return
        source_id = hotspot.hotspot_id
        canonical_id = self._hotspot_aliases.get(source_id)
        if canonical_id is None and self._cluster_id is not None:
            nearby = [
                known
                for known in self._hotspots.values()
                if known.position.distance_to(hotspot.position, horizontal=True) <= 5.0
            ]
            if nearby:
                canonical_id = min(
                    nearby,
                    key=lambda known: (
                        known.position.distance_to(hotspot.position, horizontal=True),
                        known.hotspot_id,
                    ),
                ).hotspot_id
                self._hotspot_aliases[source_id] = canonical_id
        if canonical_id is not None:
            hotspot.hotspot_id = canonical_id
        else:
            canonical_id = source_id
            self._hotspot_aliases[source_id] = canonical_id
        previous = self._hotspots.get(canonical_id)
        if previous is not None:
            if hotspot.last_seen < previous.last_seen:
                return
            if previous.status is not hotspot.status:
                hotspot.status = previous.status
            if previous.assigned_to is not None:
                hotspot.assigned_to = previous.assigned_to
        self._hotspots[hotspot.hotspot_id] = hotspot
        self._target_positions[hotspot.hotspot_id] = hotspot.position
        created = create_tasks_from_hotspots(
            [hotspot],
            self._now(),
            existing_tasks=self._tasks.values(),
        )
        for task in created:
            if task.task_id not in self._tasks:
                task.cluster_id = self._cluster_id
                task.cluster_epoch = self._cluster_epoch
                self._tasks[task.task_id] = task

    def _on_announce(self, message: SwarmMessage) -> None:
        if message.is_expired(self._now()) or not isinstance(message.body, TaskAnnounce):
            return
        if message.epoch is None or message.epoch < self._epoch:
            return
        announced = message.body.task.model_copy(deep=True)
        if (
            announced.cluster_id != self._cluster_id
            or announced.cluster_epoch != self._cluster_epoch
            or (
                self._cluster_id is not None
                and (message.epoch != announced.epoch or message.epoch != self._epoch)
            )
        ):
            return
        current = self._tasks.get(announced.task_id)
        if current is not None and current.status in (TaskStatus.DONE, TaskStatus.CANCELLED):
            return
        if current is None or announced.epoch > current.epoch or (
            announced.epoch == current.epoch and current.status is TaskStatus.PENDING
        ):
            self._tasks[announced.task_id] = announced

    def _on_assignment(self, message: SwarmMessage) -> None:
        if message.is_expired(self._now()) or not isinstance(message.body, TaskAssign):
            return
        assigned = message.body.task.model_copy(deep=True)
        if (
            message.receiver_id != assigned.assigned_to
            or message.epoch != message.body.epoch
            or assigned.epoch != message.body.epoch
            or message.body.epoch < self._epoch
            or assigned.cluster_id != self._cluster_id
            or assigned.cluster_epoch != self._cluster_epoch
            or (
                self._cluster_id is not None
                and (message.body.epoch != self._epoch or assigned.epoch != self._epoch)
            )
        ):
            return
        current = self._tasks.get(assigned.task_id)
        if current is not None and current.status in (TaskStatus.DONE, TaskStatus.CANCELLED):
            return
        if current is None or assigned.epoch > current.epoch or (
            assigned.epoch == current.epoch and current.status is TaskStatus.PENDING
        ):
            self._tasks[assigned.task_id] = assigned

    def _on_ack(self, message: SwarmMessage) -> None:
        if message.is_expired(self._now()) or not isinstance(message.body, TaskAck):
            return
        ack = message.body
        task = self._tasks.get(ack.task_id)
        if (
            ack.drone_id != message.sender_id
            or task is None
            or task.status is not TaskStatus.ASSIGNED
            or task.assigned_to != ack.drone_id
            or ack.epoch != task.epoch
            or message.epoch != ack.epoch
        ):
            return
        self._ack_deadline.pop(task.task_id, None)
        if ack.accepted:
            task.status = TaskStatus.IN_PROGRESS
        else:
            self._return_to_pending(task)

    def _on_complete(self, message: SwarmMessage) -> None:
        if message.is_expired(self._now()) or not isinstance(message.body, TaskComplete):
            return
        complete = message.body
        task = self._tasks.get(complete.task_id)
        if (
            complete.drone_id != message.sender_id
            or task is None
            or task.status not in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
            or task.assigned_to != complete.drone_id
            or message.epoch != task.epoch
        ):
            return
        task.status = TaskStatus.DONE if complete.success else TaskStatus.FAILED
        self._ack_deadline.pop(task.task_id, None)
        if complete.success:
            self._update_zone_assignment(
                task,
                drone_id=None,
                scanned_at=self._now(),
            )
        else:
            self._return_to_pending(task)
        if complete.result_status is not None and task.target_hotspot_id is not None:
            hotspot = self._hotspots.get(task.target_hotspot_id)
            if hotspot is not None:
                hotspot.status = complete.result_status

    def _return_to_pending(self, task: Task) -> None:
        task.status = TaskStatus.PENDING
        task.assigned_to = None
        self._update_zone_assignment(task, drone_id=None)
        self._ack_deadline.pop(task.task_id, None)
        self._retry_at[task.task_id] = self._now() + self._retry_delay_s

    def apply_failure_report(self, report: FailureReport) -> None:
        """Requeue work returned by the heartbeat monitor for failed drones."""
        for failed in report.requeued_tasks:
            task = self._tasks.get(failed.task_id)
            if task is not None:
                task.status = TaskStatus.PENDING
                task.assigned_to = None
                self._update_zone_assignment(task, drone_id=None)
                self._ack_deadline.pop(task.task_id, None)
                self._retry_at[task.task_id] = self._now()

    def update_leadership(self, leader_id: str, epoch: int) -> None:
        """Switch auction authority to a newer elected leader."""
        if epoch < self._epoch:
            raise ValueError("leader epoch cannot move backward")
        if epoch == self._epoch and (leader_id == self._sender_id) == self._is_leader:
            return
        active = self._auction.active_task
        if active is not None:
            task = self._tasks.get(active.task_id)
            if task is not None and task.status is TaskStatus.PENDING:
                self._retry_at[task.task_id] = self._now()
        self._active_task_id = None
        if active is not None:
            self._retry_at[active.task_id] = self._now()
        self._auction.close()
        self._epoch = epoch
        self._is_leader = leader_id == self._sender_id
        self._auction = TaskAuctionCoordinator(
            self._bus,
            self._clock,
            self._sender_id,
            self._target_positions,
            epoch=epoch,
            bid_window_s=self._bid_window_s,
            cluster_id=self._cluster_id,
            cluster_epoch=self._cluster_epoch,
        )

    def update_cluster(
        self,
        *,
        leader_id: str,
        epoch: int,
        cluster_epoch: int,
        member_ids: Iterable[str],
        zones: Iterable[Zone] | None = None,
    ) -> None:
        """Adopt a newer cluster term/membership and reset any interrupted auction."""
        if epoch < self._epoch:
            raise ValueError("leader epoch cannot move backward")
        if cluster_epoch < self._cluster_epoch:
            raise ValueError("cluster epoch cannot move backward")
        members = tuple(sorted(set(member_ids)))
        if not members:
            raise ValueError("cluster must have at least one member")
        updated_zones = (
            self._zones
            if zones is None
            else tuple(zone.model_copy(deep=True) for zone in zones)
        )
        if updated_zones != self._zones:
            self._zones = updated_zones
            self._ensure_zone_tasks()
        changed = (
            cluster_epoch != self._cluster_epoch
            or members != self._member_ids
            or (leader_id == self._sender_id) != self._is_leader
            or epoch != self._epoch
        )
        if not changed:
            return

        active = self._auction.active_task
        if active is not None:
            self._retry_at[active.task_id] = self._now()
            task = self._tasks.get(active.task_id)
            if task is not None and task.status is TaskStatus.PENDING:
                task.cluster_epoch = cluster_epoch
        self._active_task_id = None
        self._auction.close()
        self._epoch = epoch
        self._cluster_epoch = cluster_epoch
        self._member_ids = members
        self._is_leader = leader_id == self._sender_id
        for task in self._tasks.values():
            if task.status is TaskStatus.PENDING:
                task.cluster_epoch = cluster_epoch
        self._auction = self._make_auction()
        if self._started:
            self._refresh_hotspot_subscriptions()

    def _refresh_hotspot_subscriptions(self) -> None:
        for subscription in self._subscriptions:
            subscription.cancel()
        self._subscriptions = []
        hotspot_topics = (
            [f"drone/{member_id}/hotspots" for member_id in self._member_ids]
            if self._member_ids is not None
            else ["drone/*/hotspots"]
        )
        self._subscriptions.extend(
            self._communicator.subscribe(topic, self._on_hotspot)
            for topic in hotspot_topics
        )
        self._subscriptions.extend([
            self._communicator.subscribe(
                f"swarm/clusters/{self._cluster_id}/tasks/announce"
                if self._cluster_id is not None
                else "swarm/tasks/announce",
                self._on_announce,
            ),
            self._communicator.subscribe("swarm/tasks/acks", self._on_ack),
            self._communicator.subscribe("swarm/tasks/complete", self._on_complete),
        ])
        assignment_topics = (
            [f"drone/{member_id}/cmd" for member_id in self._member_ids]
            if self._member_ids is not None
            else ["drone/*/cmd"]
        )
        self._subscriptions.extend(
            self._communicator.subscribe(topic, self._on_assignment)
            for topic in assignment_topics
        )

    def tick(self) -> tuple[Task, ...]:
        """Advance the auction/ACK timeouts and start the next eligible task."""
        now = self._now()
        for task_id, deadline in tuple(self._ack_deadline.items()):
            if now >= deadline:
                task = self._tasks.get(task_id)
                if task is not None and task.status is TaskStatus.ASSIGNED:
                    self._return_to_pending(task)

        if self._active_task_id is not None:
            outcome = self._auction.tick()
            if outcome is None:
                return ()
            task_id = self._active_task_id
            self._active_task_id = None
            task = self._tasks[task_id]
            if outcome.assignment is None:
                self._retry_at[task_id] = now + self._retry_delay_s
                return ()
            task.status = TaskStatus.ASSIGNED
            task.assigned_to = outcome.winner_id
            self._update_zone_assignment(task, drone_id=outcome.winner_id)
            task.epoch = self._epoch
            self._ack_deadline[task_id] = now + self._ack_timeout_s
            return (task.model_copy(deep=True),)

        if not self._is_leader:
            return ()
        eligible = [
            task for task in self._tasks.values()
            if task.status is TaskStatus.PENDING and self._retry_at.get(task.task_id, 0.0) <= now
        ]
        if not eligible:
            return ()
        eligible.sort(key=lambda task: (-task.priority, task.created_at, task.task_id))
        selected = eligible[0]
        if self._cluster_id is not None:
            selected.cluster_id = self._cluster_id
            selected.cluster_epoch = self._cluster_epoch
        self._auction.start_auction(selected)
        self._active_task_id = selected.task_id
        return ()

    def close(self) -> None:
        for subscription in self._subscriptions:
            subscription.cancel()
        self._subscriptions.clear()
        self._auction.close()
        self._started = False
