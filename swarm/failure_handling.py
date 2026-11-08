"""Heartbeat timeouts, lost-drone alerts, and safe task re-queueing."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from contracts import (
    Alert,
    AlertKind,
    DroneState,
    DroneStatus,
    LeaderClaim,
    Severity,
    SwarmRole,
    Task,
    TaskStatus,
)

from .leader_election import elect_leader

__all__ = ["FailureMonitor", "FailureReport"]


@dataclass(frozen=True)
class FailureReport:
    newly_lost: tuple[str, ...]
    lost_states: tuple[DroneState, ...]
    requeued_tasks: tuple[Task, ...]
    alerts: tuple[Alert, ...]
    hold_after_current_task: bool
    leader_claim: LeaderClaim | None


class FailureMonitor:
    """Treat a drone as unresponsive after several missed heartbeat intervals.

    A timeout indicates loss of communication, not proof that the aircraft itself
    has failed. A returning heartbeat clears the unresponsive state.
    """

    def __init__(
        self,
        *,
        heartbeat_period_s: float = 0.5,
        missed_heartbeat_threshold: int = 6,
        heartbeat_timeout_s: float | None = None,
    ) -> None:
        if not math.isfinite(heartbeat_period_s) or heartbeat_period_s <= 0:
            raise ValueError("heartbeat_period_s must be finite and > 0")
        if missed_heartbeat_threshold < 1:
            raise ValueError("missed_heartbeat_threshold must be >= 1")
        if heartbeat_timeout_s is not None and (
            not math.isfinite(heartbeat_timeout_s) or heartbeat_timeout_s <= 0
        ):
            raise ValueError("heartbeat_timeout_s must be finite and > 0 when provided")
        missed_timeout_s = heartbeat_period_s * missed_heartbeat_threshold
        self._timeout_s = max(missed_timeout_s, heartbeat_timeout_s or 0.0)
        self._last_seen: dict[str, float] = {}
        self._states: dict[str, DroneState] = {}
        self._lost: set[str] = set()

    @property
    def lost_drone_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._lost))

    def observe_heartbeat(self, state: DroneState, received_at: float) -> bool:
        """Record a fresh heartbeat; return False for an out-of-order arrival."""
        if not math.isfinite(received_at):
            raise ValueError("received_at must be finite")
        previous = self._last_seen.get(state.drone_id)
        if previous is not None and received_at < previous:
            return False
        self._last_seen[state.drone_id] = received_at
        self._states[state.drone_id] = state.model_copy(deep=True)
        self._lost.discard(state.drone_id)
        return True

    def check_timeouts(self, now: float, tasks: Iterable[Task] = ()) -> FailureReport:
        """Mark newly stale drones and return alerts plus active tasks needing reassignment."""
        if not math.isfinite(now):
            raise ValueError("now must be finite")

        newly_lost: list[str] = []
        lost_states: list[DroneState] = []
        requeued_by_id: dict[str, Task] = {}
        alerts: list[Alert] = []
        hold_after_current_task = False
        leader_lost = False
        task_list = list(tasks)

        for drone_id in sorted(self._last_seen):
            age_s = now - self._last_seen[drone_id]
            if age_s < self._timeout_s or drone_id in self._lost:
                continue

            state = self._states[drone_id]
            self._lost.add(drone_id)
            newly_lost.append(drone_id)
            lost_state = state.model_copy(update={"status": DroneStatus.LOST}, deep=True)
            self._states[drone_id] = lost_state
            lost_states.append(lost_state)
            is_leader = state.role is SwarmRole.LEADER
            leader_lost = leader_lost or is_leader
            alert_kind = AlertKind.LEADER_LOST if is_leader else AlertKind.DRONE_LOST
            alerts.append(
                Alert(
                    alert_id=f"{alert_kind.value.lower()}-{drone_id}-{now:.6f}",
                    severity=Severity.CRITICAL if is_leader else Severity.WARNING,
                    kind=alert_kind,
                    subject_id=drone_id,
                    message=(
                        f"{drone_id} has been unresponsive for {age_s:.1f} seconds "
                        f"({self._timeout_s:.1f}-second timeout)"
                    ),
                    position=state.position,
                    timestamp=now,
                )
            )
            hold_after_current_task = hold_after_current_task or is_leader

            for task in task_list:
                owns_task = task.assigned_to == drone_id or task.task_id == state.current_task_id
                if owns_task and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                    requeued_by_id[task.task_id] = task.model_copy(
                        update={"assigned_to": None, "status": TaskStatus.PENDING},
                        deep=True,
                    )

        viable_candidates = [
            state for drone_id, state in self._states.items()
            if drone_id not in self._lost
            and state.status not in (DroneStatus.FAULT, DroneStatus.LOST)
        ]
        leader_claim = (
            elect_leader(self._states.values())
            if leader_lost and viable_candidates
            else None
        )

        return FailureReport(
            newly_lost=tuple(newly_lost),
            lost_states=tuple(lost_states),
            requeued_tasks=tuple(requeued_by_id[key] for key in sorted(requeued_by_id)),
            alerts=tuple(alerts),
            hold_after_current_task=hold_after_current_task,
            leader_claim=leader_claim,
        )