"""In-memory ground-base queue for perception candidates requiring human review."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from contracts import Bus, ReviewReport, SUB_GROUND_REVIEWS, SwarmMessage

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

    def __init__(self, bus: Bus, now: Callable[[], float]):
        self._now = now
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
        return resolved

    def resolved(self) -> list[ReviewRecord]:
        return list(self._resolved.values())

    def close(self) -> None:
        self.subscription.cancel()
