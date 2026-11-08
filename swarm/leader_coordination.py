"""Bus-backed leader monitoring, failover election, and hold/resume signaling."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from contracts import (
    AlertNotice,
    Bus,
    Clock,
    DroneState,
    Heartbeat,
    LeaderClaim,
    Subscription,
    SwarmMessage,
    Task,
)

from .communication import SwarmCommunicator
from .failure_handling import FailureMonitor, FailureReport
from .leader_election import elect_leader, leader_score, select_leader_claim

__all__ = ["LeaderCoordinator"]


class LeaderCoordinator:
    """Observe heartbeats and claims, publish leader changes, and report failures.

    The application supplies callbacks for autonomy actions: the swarm can request
    a hold, but only the per-drone controller can safely execute it. Call ``tick()``
    from the deterministic simulation/application loop.
    """

    def __init__(
        self,
        bus: Bus,
        clock: Clock,
        local_state_provider: Callable[[], DroneState],
        *,
        initial_leader_id: str | None = None,
        epoch: int | None = None,
        heartbeat_period_s: float = 0.5,
        missed_heartbeat_threshold: int = 6,
        leader_timeout_s: float | None = None,
        claim_period_s: float = 1.0,
        task_provider: Callable[[], Iterable[Task]] | None = None,
        active_task_provider: Callable[[], bool] | None = None,
        on_hold: Callable[[], None] | None = None,
        on_resume: Callable[[], None] | None = None,
        on_leader_change: Callable[[str, int], None] | None = None,
        on_failure_report: Callable[[FailureReport], None] | None = None,
    ) -> None:
        local_state = local_state_provider()
        if not local_state.drone_id:
            raise ValueError("local drone_id must not be empty")
        if not math.isfinite(claim_period_s) or claim_period_s <= 0:
            raise ValueError("claim_period_s must be finite and > 0")
        if leader_timeout_s is not None and (
            not math.isfinite(leader_timeout_s) or leader_timeout_s <= 0
        ):
            raise ValueError("leader_timeout_s must be finite and > 0")

        self._bus = bus
        self._clock = clock
        self._local_state_provider = local_state_provider
        self._local_id = local_state.drone_id
        self._communicator = SwarmCommunicator(bus, clock, self._local_id)
        self._monitor = FailureMonitor(
            heartbeat_period_s=heartbeat_period_s,
            missed_heartbeat_threshold=missed_heartbeat_threshold,
            heartbeat_timeout_s=leader_timeout_s,
        )
        self._timeout_s = max(
            heartbeat_period_s * missed_heartbeat_threshold,
            leader_timeout_s or 0.0,
        )
        self._claim_period_s = claim_period_s
        self._task_provider = task_provider or (lambda: ())
        self._active_task_provider = active_task_provider or (lambda: False)
        self._on_hold = on_hold or (lambda: None)
        self._on_resume = on_resume or (lambda: None)
        self._on_leader_change = on_leader_change or (lambda *_: None)
        self._on_failure_report = on_failure_report or (lambda *_: None)
        self._states: dict[str, DroneState] = {self._local_id: local_state.model_copy(deep=True)}
        self._epoch = max(local_state.epoch, epoch or 0)
        self._leader_id = initial_leader_id or (
            local_state.drone_id if local_state.role.value == "LEADER" else local_state.leader_id
        )
        self._current_claim: LeaderClaim | None = None
        self._last_claim_at: float | None = None
        self._last_published_claim_at: float | None = None
        self._last_published_claim: tuple[str, int] | None = None
        self._election_in_progress = False
        self._election_started_at: float | None = None
        self._hold_requested = False
        self._hold_executed = False
        self._subscriptions: list[Subscription] = []
        self._started = False
        now = self._now()
        if self._leader_id is not None:
            self._last_claim_at = now
        if local_state.role.value == "LEADER":
            self._leader_id = self._local_id
            self._last_claim_at = now
            self._current_claim = LeaderClaim(
                candidate_id=self._local_id,
                epoch=self._epoch,
                score=leader_score(local_state),
            )

    @property
    def leader_id(self) -> str | None:
        return self._leader_id

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def hold_requested(self) -> bool:
        """Whether autonomy should hold after its current task completes."""
        return self._hold_requested

    def start(self) -> None:
        """Subscribe to swarm heartbeats/claims and announce local leadership."""
        if self._started:
            raise RuntimeError("leader coordinator is already started")
        self._started = True
        self._subscriptions = [
            self._communicator.subscribe("drone/*/heartbeat", self._on_message),
            self._communicator.subscribe("swarm/leader", self._on_message),
        ]
        if self._leader_id == self._local_id:
            self._publish_local_claim(self._now())

    def _now(self) -> float:
        now = self._clock.now()
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite time")
        return now

    def _on_message(self, message: SwarmMessage) -> None:
        now = self._now()
        if message.is_expired(now):
            return
        body = message.body
        if isinstance(body, Heartbeat):
            state = body.state
            if state.drone_id != message.sender_id:
                return
            if self._monitor.observe_heartbeat(state, received_at=now):
                self._states[state.drone_id] = state.model_copy(deep=True)
                if self._leader_id is None and state.role.value == "LEADER":
                    self._leader_id = state.drone_id
                    self._epoch = max(self._epoch, state.epoch)
                    self._last_claim_at = now
                    self._on_leader_change(self._leader_id, self._epoch)
            return

        if not isinstance(body, LeaderClaim):
            return
        if body.candidate_id != message.sender_id or message.epoch != body.epoch:
            return
        if body.epoch < self._epoch:
            return
        current = self._current_claim
        if body.candidate_id == self._leader_id and body.epoch == self._epoch:
            self._last_claim_at = now
        if current is not None and body.epoch < current.epoch:
            return
        winner = select_leader_claim([body] if current is None else [current, body])
        if winner == body:
            self._accept_claim(body, now)

    def _accept_claim(self, claim: LeaderClaim, now: float) -> None:
        changed = claim.candidate_id != self._leader_id or claim.epoch != self._epoch
        was_holding = self._hold_requested
        self._leader_id = claim.candidate_id
        self._epoch = claim.epoch
        self._last_claim_at = now
        self._current_claim = claim
        self._election_in_progress = False
        self._election_started_at = None
        if changed:
            self._on_leader_change(self._leader_id, self._epoch)
        if was_holding:
            if self._active_task_provider():
                self._hold_requested = False
                self._hold_executed = False
            elif self._hold_executed:
                self._hold_requested = False
                self._hold_executed = False
                self._on_resume()

    def _request_hold(self) -> None:
        if self._hold_requested:
            return
        self._hold_requested = True
        if not self._active_task_provider():
            self._on_hold()
            self._hold_executed = True

    def task_finished(self) -> None:
        """Complete the deferred hold once the active task has safely ended."""
        if self._hold_requested and not self._active_task_provider():
            self._on_hold()
            self._hold_executed = True

    def _publish_local_claim(self, now: float, claim: LeaderClaim | None = None) -> None:
        local_state = self._local_state_provider()
        self._states[self._local_id] = local_state.model_copy(
            update={"epoch": self._epoch},
            deep=True,
        )
        selected = claim or LeaderClaim(
            candidate_id=self._local_id,
            epoch=self._epoch,
            score=leader_score(local_state),
        )
        marker = (selected.candidate_id, selected.epoch)
        if self._last_published_claim == marker and self._last_published_claim_at is not None:
            if now - self._last_published_claim_at < self._claim_period_s:
                return
        self._communicator.publish(selected, epoch=selected.epoch)
        self._last_published_claim = marker
        self._last_published_claim_at = now
        self._accept_claim(selected, now)

    def _begin_election(self, now: float, proposed: LeaderClaim | None = None) -> LeaderClaim | None:
        self._request_hold()
        claim = proposed
        if claim is not None and claim.epoch <= self._epoch:
            candidate = self._states.get(claim.candidate_id)
            if (
                candidate is not None
                and candidate.drone_id not in self._monitor.lost_drone_ids
                and candidate.status.value not in ("FAULT", "LOST")
            ):
                claim = LeaderClaim(
                    candidate_id=claim.candidate_id,
                    epoch=self._epoch + 1,
                    score=leader_score(candidate),
                )
            else:
                claim = None
        if claim is None:
            local_state = self._local_state_provider()
            self._states[self._local_id] = local_state.model_copy(
                update={"epoch": self._epoch},
                deep=True,
            )
            lost_ids = set(self._monitor.lost_drone_ids)
            candidates = [
                state for drone_id, state in self._states.items()
                if drone_id not in lost_ids
                and drone_id != self._leader_id
                and state.status.value not in ("FAULT", "LOST")
            ]
            if not candidates:
                candidates = [
                    state for drone_id, state in self._states.items()
                    if drone_id not in lost_ids and state.status.value not in ("FAULT", "LOST")
                ]
            if candidates:
                next_epoch = max(self._epoch, *(state.epoch for state in candidates)) + 1
                claim = elect_leader(candidates, epoch=next_epoch)
        if claim is not None and claim.candidate_id == self._local_id:
            self._epoch = max(self._epoch, claim.epoch)
            self._publish_local_claim(now, claim)
        return claim

    def tick(self) -> FailureReport:
        """Check heartbeat/claim timeouts, publish alerts, and initiate failover."""
        now = self._now()
        report = self._monitor.check_timeouts(now, self._task_provider())
        self._states.update({
            state.drone_id: state.model_copy(deep=True)
            for state in report.lost_states
        })
        if report.hold_after_current_task:
            self._request_hold()
        if report.newly_lost:
            self._on_failure_report(report)
            for alert in report.alerts:
                self._communicator.publish(AlertNotice(alert=alert), epoch=self._epoch)

        leader_was_lost = self._leader_id in report.newly_lost
        claim_silent = (
            self._leader_id is not None
            and self._last_claim_at is not None
            and now - self._last_claim_at >= self._timeout_s
        )
        election_due = (
            not self._election_in_progress
            or self._election_started_at is None
            or now - self._election_started_at >= self._timeout_s
        )
        if (leader_was_lost or claim_silent or self._leader_id is None) and election_due:
            self._election_in_progress = True
            self._election_started_at = now
            self._begin_election(now, report.leader_claim if leader_was_lost else None)

        if self._leader_id == self._local_id:
            self._publish_local_claim(now)
        return report

    def close(self) -> None:
        """Cancel subscriptions and stop publishing claims."""
        for subscription in self._subscriptions:
            subscription.cancel()
        self._subscriptions.clear()
        self._started = False
