"""Review triage publishing and ground-base queue integration tests."""
from __future__ import annotations

import numpy as np
import pytest

from contracts import (
    CameraModel,
    Detection,
    DetectorKind,
    Hotspot,
    HotspotReport,
    MessageHandler,
    Position,
    Pose,
    ReviewReport,
    SwarmMessage,
    ThermalFrame,
    topic_matches,
)
from groundbase.review_queue import GroundReviewQueue, ReviewDisposition
from perception.pipeline import SeekerPerception
from perception.reporting import PerceptionReporter
from perception.triage import TriageBands


class FakeSubscription:
    def cancel(self) -> None:
        pass


class FakeBus:
    def __init__(self):
        self.published: list[tuple[str, SwarmMessage]] = []
        self.subscribers: list[tuple[str, MessageHandler]] = []

    def publish(self, topic: str, message: SwarmMessage) -> None:
        self.published.append((topic, message))
        for pattern, handler in self.subscribers:
            if topic_matches(pattern, topic):
                handler(message)

    def subscribe(self, pattern: str, handler: MessageHandler):
        self.subscribers.append((pattern, handler))
        return FakeSubscription()


class FixedScoreDetector:
    def __init__(self, score: float):
        self.score = score

    def detect(self, frame: ThermalFrame) -> list[Detection]:
        return [
            Detection(
                u=32,
                v=32,
                area_px=4,
                peak_temp_c=180,
                confidence=self.score,
                source=DetectorKind.THRESHOLD,
            )
        ]


def build_reporter(score: float, bands: TriageBands, bus: FakeBus) -> PerceptionReporter:
    perception = SeekerPerception("seeker-1", FixedScoreDetector(score))
    return PerceptionReporter(perception, bus, bands)


def make_frame(timestamp: float) -> ThermalFrame:
    return ThermalFrame(
        drone_id="seeker-1",
        timestamp=timestamp,
        temp_c=np.zeros((64, 64), dtype=np.float32),
        pose=Pose(position=Position(x=0, y=0, z=80), yaw_deg=0),
        camera=CameraModel(width_px=64, height_px=64, hfov_deg=32),
    )


def test_review_candidate_is_published_to_ground_and_added_to_queue():
    bus = FakeBus()
    now = [1.0]
    queue = GroundReviewQueue(bus, now=lambda: now[0])
    reporter = build_reporter(0.5, TriageBands(0.2, 0.8), bus)

    hotspots = reporter.process(make_frame(1.0))

    assert len(hotspots) == 1
    assert any(topic == "drone/seeker-1/hotspots" and isinstance(msg.body, HotspotReport)
               for topic, msg in bus.published)
    review_messages = [(topic, msg) for topic, msg in bus.published if isinstance(msg.body, ReviewReport)]
    assert len(review_messages) == 1
    topic, message = review_messages[0]
    assert topic == "ground/reviews"
    assert message.reliable
    assert message.body.review_id == f"seeker-1:{hotspots[0].hotspot_id}"
    assert message.body.score == pytest.approx(0.5)
    assert [record.report.review_id for record in queue.pending()] == [message.body.review_id]


def test_duplicate_review_reports_update_single_queue_item_and_resolved_id_stays_closed():
    bus = FakeBus()
    now = [1.0]
    queue = GroundReviewQueue(bus, now=lambda: now[0])
    reporter = build_reporter(0.5, TriageBands(0.2, 0.8), bus)

    reporter.process(make_frame(1.0))
    reporter.process(make_frame(2.0))

    reports = [msg for _, msg in bus.published if isinstance(msg.body, ReviewReport)]
    assert len(reports) == 1
    assert len(queue.pending()) == 1

    resolved = queue.resolve(reports[0].body.review_id, ReviewDisposition.REVISIT)
    assert resolved.disposition is ReviewDisposition.REVISIT
    assert queue.pending() == []
    assert queue.resolved() == [resolved]
    assert queue.handle_message(reports[0])
    assert queue.pending() == []


@pytest.mark.parametrize(
    ("score", "bands"),
    [
        (0.9, TriageBands(0.2, 0.8)),
        (0.1, TriageBands(0.2, 0.8)),
    ],
)
def test_fire_and_not_fire_candidates_do_not_enter_ground_review_queue(score, bands):
    bus = FakeBus()
    queue = GroundReviewQueue(bus, now=lambda: 1.0)
    reporter = build_reporter(score, bands, bus)

    reporter.process(make_frame(1.0))

    assert queue.pending() == []
    assert not any(isinstance(message.body, ReviewReport) for _, message in bus.published)


def test_expired_review_report_is_not_added_to_queue():
    bus = FakeBus()
    now = [400.0]
    queue = GroundReviewQueue(bus, now=lambda: now[0])
    reporter = build_reporter(0.5, TriageBands(0.2, 0.8), bus)
    reporter.process(make_frame(1.0))

    assert queue.pending() == []
    assert queue.expired_reports == 1


def test_review_report_rejects_score_outside_review_band():
    hotspot = Hotspot(
        hotspot_id="h1",
        position=Position(x=1, y=2, z=0),
        peak_temp_c=180,
        confidence=0.9,
        first_seen=1,
        last_seen=1,
    )
    with pytest.raises(ValueError):
        ReviewReport(
            review_id="seeker-1:h1",
            hotspot=hotspot,
            score=0.9,
            not_fire_below=0.2,
            fire_at_or_above=0.8,
        )
