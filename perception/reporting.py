"""Publish tracked hotspots and route REVIEW-band candidates to the ground base."""
from __future__ import annotations

from contracts import Body, Bus, Hotspot, HotspotReport, ReviewReport, SwarmMessage, ThermalFrame
from contracts.topics import topic_for

from .frame_context import FrameContext
from .pipeline import SeekerPerception
from .triage import Triage, TriageBands

__all__ = ["PerceptionReporter"]


class PerceptionReporter:
    """Bus integration for a seeker's perception pipeline.

    `triage_bands` must be calibrated for the tracker's fused `Hotspot.confidence` score,
    not copied from per-frame model thresholds without validation.
    """

    def __init__(self, perception: SeekerPerception, bus: Bus, triage_bands: TriageBands):
        self.perception = perception
        self.bus = bus
        self.triage_bands = triage_bands
        self._seq = 0
        self._published_reviews: set[str] = set()

    def process(self, frame: ThermalFrame, ctx: FrameContext | None = None) -> list[Hotspot]:
        """Process a frame, publish hotspot reports, and publish each REVIEW candidate once."""
        hotspots = self.perception.process(frame, ctx)
        for hotspot in hotspots:
            if self.triage_bands.classify(hotspot.confidence) is Triage.REVIEW:
                review_id = f"{self.perception.drone_id}:{hotspot.hotspot_id}"
                if review_id not in self._published_reviews:
                    report = ReviewReport(
                        review_id=review_id,
                        hotspot=hotspot,
                        score=hotspot.confidence,
                        not_fire_below=self.triage_bands.not_fire_below,
                        fire_at_or_above=self.triage_bands.fire_at_or_above,
                    )
                    self._publish(report, frame.timestamp)
                    self._published_reviews.add(review_id)
            self._publish(HotspotReport(hotspot=hotspot), frame.timestamp)
        return hotspots

    def _publish(self, body: Body, timestamp: float) -> None:
        message = SwarmMessage.create(
            self.perception.drone_id,
            body,
            now=timestamp,
            seq=self._seq,
        )
        self._seq += 1
        self.bus.publish(topic_for(message), message)
