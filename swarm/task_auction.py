"""Clock-driven, bus-based task announcements, bidding, and assignment."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from contracts import (
    Bus,
    Clock,
    DroneState,
    Position,
    SwarmMessage,
    Subscription,
    Task,
    TaskAnnounce,
    TaskAssign,
    TaskBid,
    TaskStatus,
)

from .communication import SwarmCommunicator
from .task_allocation import SSIAuctionRound, bid_for_task, target_position_for

__all__ = ["AuctionOutcome", "TaskAuctionCoordinator", "TaskBidResponder"]


@dataclass(frozen=True)
class AuctionOutcome:
    """Result emitted when an auction deadline is reached."""

    task: Task
    winner_id: str | None
    assignment: Task | None


class TaskAuctionCoordinator:
    """Run one SSI task auction at a time over the shared Bus.

    The simulation or application loop must call `tick()` regularly. The Bus
    implementation is responsible for delivery timing, range, and packet loss.
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
    ) -> None:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        if not math.isfinite(bid_window_s) or bid_window_s <= 0:
            raise ValueError("bid_window_s must be finite and > 0")

        self._clock = clock
        self._sender_id = sender_id
        self._epoch = epoch
        self._bid_window_s = bid_window_s
        self._target_positions = target_positions
        self._communicator = SwarmCommunicator(bus, clock, sender_id)
        self._round: SSIAuctionRound | None = None
        self._task: Task | None = None
        self._subscription = self._communicator.subscribe(
            "swarm/tasks/bids",
            self._on_message,
        )

    @property
    def active_task(self) -> Task | None:
        """The task currently collecting bids, if a round is open."""
        return self._task

    def start_auction(self, task: Task) -> SwarmMessage:
        """Publish a task offer and start its bid window."""
        if self._round is not None:
            raise RuntimeError("an auction round is already active")
        if task.status is not TaskStatus.PENDING:
            raise ValueError("only PENDING tasks can be auctioned")
        if task.priority <= 0:
            raise ValueError("task priority must be > 0 to run an auction")
        target_position_for(task, self._target_positions)

        now = self._clock.now()
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite time")
        deadline = now + self._bid_window_s
        offered_task = task.model_copy(deep=True)
        offered_task.epoch = self._epoch
        self._task = offered_task
        self._round = SSIAuctionRound(task.task_id, bid_deadline=deadline)
        return self._communicator.publish(
            TaskAnnounce(task=offered_task, bid_deadline=deadline),
            epoch=self._epoch,
        )

    def _on_message(self, message: SwarmMessage) -> None:
        if self._round is None or message.epoch != self._epoch:
            return
        if message.is_expired(self._clock.now()):
            return
        body = message.body
        if not isinstance(body, TaskBid):
            return
        if body.bidder_id != message.sender_id:
            return
        self._round.submit_bid(body, received_at=self._clock.now())

    def tick(self) -> AuctionOutcome | None:
        """Close a due round and publish a directed assignment to its winner."""
        if self._round is None or self._task is None:
            return None

        now = self._clock.now()
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite time")
        if now < self._round.bid_deadline:
            return None
        winner = self._round.close(now)

        task = self._task
        self._round = None
        self._task = None
        if winner is None:
            return AuctionOutcome(task=task, winner_id=None, assignment=None)

        assignment = task.model_copy(deep=True)
        assignment.status = TaskStatus.ASSIGNED
        assignment.assigned_to = winner.bidder_id
        assignment.epoch = self._epoch
        self._communicator.publish(
            TaskAssign(task=assignment, epoch=self._epoch),
            receiver_id=winner.bidder_id,
            epoch=self._epoch,
        )
        return AuctionOutcome(
            task=task,
            winner_id=winner.bidder_id,
            assignment=assignment,
        )

    def close(self) -> None:
        """Cancel the bid subscription and discard any active round."""
        self._subscription.cancel()
        self._round = None
        self._task = None


class TaskBidResponder:
    """Answer task announcements using the drone's latest state and task cost model."""

    def __init__(
        self,
        bus: Bus,
        clock: Clock,
        state_provider: Callable[[], DroneState],
        target_positions: Mapping[str, Position],
        *,
        reserve_s: float = 60.0,
        cruise_speed_mps: float = 10.0,
    ) -> None:
        self._clock = clock
        self._state_provider = state_provider
        self._target_positions = target_positions
        self._reserve_s = reserve_s
        self._cruise_speed_mps = cruise_speed_mps
        self._responded: set[tuple[str, int | None, str, float]] = set()
        self._communicator: SwarmCommunicator | None = None
        self._subscription: Subscription | None = None
        self._bus = bus

    def start(self) -> None:
        """Subscribe once to the shared task-announcement topic."""
        if self._subscription is not None:
            raise RuntimeError("bid responder is already started")
        state = self._state_provider()
        self._communicator = SwarmCommunicator(self._bus, self._clock, state.drone_id)
        self._subscription = self._communicator.subscribe(
            "swarm/tasks/announce",
            self._on_message,
        )

    def _on_message(self, message: SwarmMessage) -> None:
        if self._communicator is None or message.is_expired(self._clock.now()):
            return
        body = message.body
        if not isinstance(body, TaskAnnounce):
            return
        if self._clock.now() > body.bid_deadline:
            return

        state = self._state_provider()
        key = (
            message.sender_id,
            message.epoch,
            body.task.task_id,
            body.bid_deadline,
        )
        if key in self._responded:
            return
        self._responded.add(key)

        if message.epoch is not None and message.epoch < state.epoch:
            return
        target = target_position_for(body.task, self._target_positions)
        bid = bid_for_task(
            body.task,
            state,
            target,
            now=self._clock.now(),
            reserve_s=self._reserve_s,
            cruise_speed_mps=self._cruise_speed_mps,
        )
        self._communicator.publish(bid, epoch=message.epoch)

    def close(self) -> None:
        """Cancel the task-announcement subscription."""
        if self._subscription is not None:
            self._subscription.cancel()
            self._subscription = None
