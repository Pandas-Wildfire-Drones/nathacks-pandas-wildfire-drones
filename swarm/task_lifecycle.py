"""Leader-side hotspot intake, task auctioning, and task status lifecycle."""

from __future__ import annotations

import math
from collections.abc import Mapping

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
)

from .communication import SwarmCommunicator
from .failure_handling import FailureReport
from .task_auction import TaskAuctionCoordinator
from .task_creation import create_tasks_from_hotspots

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
    ) -> None:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        if not math.isfinite(ack_timeout_s) or ack_timeout_s <= 0:
            raise ValueError("ack_timeout_s must be finite and > 0")
        if not math.isfinite(retry_delay_s) or retry_delay_s < 0:
            raise ValueError("retry_delay_s must be finite and >= 0")
        self._bus = bus
        self._clock = clock
        self._sender_id = sender_id
        self._epoch = epoch
        self._is_leader = is_leader
        self._ack_timeout_s = ack_timeout_s
        self._retry_delay_s = retry_delay_s
        self._target_positions = dict(target_positions)
        self._hotspots: dict[str, Hotspot] = {}
        self._tasks: dict[str, Task] = {}
        self._retry_at: dict[str, float] = {}
        self._ack_deadline: dict[str, float] = {}
        self._active_task_id: str | None = None
        self._subscriptions: list[Subscription] = []
        self._started = False
        self._communicator = SwarmCommunicator(bus, clock, sender_id)
        self._auction = TaskAuctionCoordinator(
            bus,
            clock,
            sender_id,
            self._target_positions,
            epoch=epoch,
            bid_window_s=bid_window_s,
        )
        self._bid_window_s = bid_window_s

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(self._tasks[key].model_copy(deep=True) for key in sorted(self._tasks))

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
        self._subscriptions = [
            self._communicator.subscribe(
                "drone/*/hotspots",
                self._on_hotspot,
            ),
            self._communicator.subscribe("swarm/tasks/announce", self._on_announce),
            self._communicator.subscribe("drone/*/cmd", self._on_assignment),
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
        self._hotspots[hotspot.hotspot_id] = hotspot
        self._target_positions[hotspot.hotspot_id] = hotspot.position
        created = create_tasks_from_hotspots(
            [hotspot],
            self._now(),
            existing_tasks=self._tasks.values(),
        )
        for task in created:
            if task.task_id not in self._tasks:
                self._tasks[task.task_id] = task

    def _on_announce(self, message: SwarmMessage) -> None:
        if message.is_expired(self._now()) or not isinstance(message.body, TaskAnnounce):
            return
        if message.epoch is None or message.epoch < self._epoch:
            return
        announced = message.body.task.model_copy(deep=True)
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
        if not complete.success:
            self._return_to_pending(task)
        if complete.result_status is not None and task.target_hotspot_id is not None:
            hotspot = self._hotspots.get(task.target_hotspot_id)
            if hotspot is not None:
                hotspot.status = complete.result_status

    def _return_to_pending(self, task: Task) -> None:
        task.status = TaskStatus.PENDING
        task.assigned_to = None
        self._ack_deadline.pop(task.task_id, None)
        self._retry_at[task.task_id] = self._now() + self._retry_delay_s

    def apply_failure_report(self, report: FailureReport) -> None:
        """Requeue work returned by the heartbeat monitor for failed drones."""
        for failed in report.requeued_tasks:
            task = self._tasks.get(failed.task_id)
            if task is not None:
                task.status = TaskStatus.PENDING
                task.assigned_to = None
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
        self._auction.start_auction(selected)
        self._active_task_id = selected.task_id
        return ()

    def close(self) -> None:
        for subscription in self._subscriptions:
            subscription.cancel()
        self._subscriptions.clear()
        self._auction.close()
        self._started = False
