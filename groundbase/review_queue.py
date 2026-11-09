"""In-memory ground-base queue for perception candidates requiring human review.

Decision delivery (opt-in, `publish_decisions=True`): resolving a review publishes a GroundCommand on
`ground/commands`, which the swarm leader consumes:
    FIRE      -> CONFIRM_FIRE          (target_id = the candidate's hotspot id)
    NOT_FIRE  -> MARK_FALSE_POSITIVE
    REVISIT   -> nothing (the leader keeps its own autonomous timeout; ask a seeker to look again)
The swarm never WAITS for this: if no decision arrives in time the leader decides on its own (hard rule 4 spirit:
the ground link is a bonus, not a dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from contracts import (
    SUB_GROUND_REVIEWS,
    Bus,
    CommandKind,
    GroundCommand,
    ReviewReport,
    SwarmMessage,
    topic_for,
)

__all__ = ["ReviewDisposition", "ReviewRecord", "GroundReviewQueue"]


class ReviewDisposition(StrEnum):
    FIRE = "FIRE"
    NOT_FIRE = "NOT_FIRE"
    REVISIT = "REVISIT"


@dataclass(frozen=True)
class ReviewRecord:
    report: ReviewReport
    sender_id: str
    received_at: float
    disposition: ReviewDisposition | None = None


class GroundReviewQueue:
    """Subscribe to ground review reports, deduplicate by review ID, and track operator decisions."""

    def __init__(self, bus: Bus, now: Callable[[], float], publish_decisions: bool = False,
                 sender_id: str = "ground"):
        self._now = now
        self._bus = bus
        self.publish_decisions = publish_decisions
        self.sender_id = sender_id
        self._seq = 0
        self._pending: dict[str, ReviewRecord] = {}
        self._resolved: dict[str, ReviewRecord] = {}
        self.expired_reports = 0
        self.subscription = bus.subscribe(SUB_GROUND_REVIEWS, self.handle_message)

    def handle_message(self, message: SwarmMessage) -> bool:
        """Accept a valid, current review report; duplicate reports update rather than multiply queue items."""
        if not isinstance(message.body, ReviewReport):
            raise TypeError(f"Expected ReviewReport, received {type(message.body).__name__}")
        now = self._now()
        if message.is_expired(now):
            self.expired_reports += 1
            return False

        report = message.body
        if report.review_id != f"{message.sender_id}:{report.hotspot.hotspot_id}":
            raise ValueError("review_id must identify its sender and hotspot")
        if report.review_id in self._resolved:
            return True
        record = ReviewRecord(report=report, sender_id=message.sender_id, received_at=now)
        existing = self._pending.get(report.review_id)
        if existing is not None:
            record = ReviewRecord(
                report=report,
                sender_id=message.sender_id,
                received_at=existing.received_at,
            )
        self._pending[report.review_id] = record
        return True

    def pending(self) -> list[ReviewRecord]:
        return list(self._pending.values())

    def resolve(self, review_id: str, disposition: ReviewDisposition) -> ReviewRecord:
        """Record an operator decision. Decision delivery back to the drone is a separate integration."""
        try:
            record = self._pending.pop(review_id)
        except KeyError as exc:
            raise KeyError(f"No pending review with id {review_id!r}") from exc
        resolved = ReviewRecord(
            report=record.report,
            sender_id=record.sender_id,
            received_at=record.received_at,
            disposition=disposition,
        )
        self._resolved[review_id] = resolved
        if self.publish_decisions:
            self._deliver(resolved)
        return resolved

    def _deliver(self, record: ReviewRecord) -> None:
        kind = {ReviewDisposition.FIRE: CommandKind.CONFIRM_FIRE,
                ReviewDisposition.NOT_FIRE: CommandKind.MARK_FALSE_POSITIVE}.get(record.disposition)
        if kind is None:
            return
        body = GroundCommand(command=kind, target_id=record.report.hotspot.hotspot_id,
                             params={"review_id": record.report.review_id}, issued_by=self.sender_id)
        self._seq += 1
        msg = SwarmMessage.create(self.sender_id, body, self._now(), seq=self._seq)
        self._bus.publish(topic_for(msg), msg)

    def resolved(self) -> list[ReviewRecord]:
        return list(self._resolved.values())

    def close(self) -> None:
        self.subscription.cancel()
