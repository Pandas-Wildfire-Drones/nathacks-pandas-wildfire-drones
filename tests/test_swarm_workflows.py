"""Integration tests for the bus-backed swarm workflows."""

from __future__ import annotations

from contracts import (
    DroneKind,
    DroneState,
    DroneStatus,
    ExtinguishingAgent,
    Heartbeat,
    Hotspot,
    HotspotReport,
    HotspotStatus,
    MessageType,
    Position,
    Powertrain,
    RefillRequest,
    RefillStation,
    RefillTargetKind,
    SwarmRole,
    TaskAck,
    TaskComplete,
    TaskStatus,
    WaterSource,
    WaterSourceKind,
    WaterSourceOrigin,
    WaterSourceStatus,
    decode,
    encode,
    topic_matches,
)
from swarm import (
    LeaderCoordinator,
    RefillAssignmentCoordinator,
    SwarmCommunicator,
    TaskBidResponder,
    TaskLifecycleCoordinator,
)


class MutableClock:
    def __init__(self, now: float = 0.0) -> None:
        self.current_time = now

    def now(self) -> float:
        return self.current_time


class LoopbackSubscription:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class LoopbackBus:
    """Synchronous bus exercising serialization and topic routing at each delivery."""

    def __init__(self) -> None:
        self._subscriptions = []
        self.published = []

    def publish(self, topic, msg) -> None:
        self.published.append((topic, msg))
        raw = encode(msg)
        for pattern, handler, subscription in list(self._subscriptions):
            if not subscription.cancelled and topic_matches(pattern, topic):
                handler(decode(raw))

    def subscribe(self, pattern, handler):
        subscription = LoopbackSubscription()
        self._subscriptions.append((pattern, handler, subscription))
        return subscription


def make_drone(
    drone_id: str,
    *,
    role: SwarmRole = SwarmRole.FOLLOWER,
    kind: DroneKind = DroneKind.SEEKER,
    x: float = 0.0,
    epoch: int = 0,
    water_remaining_l: float | None = None,
) -> DroneState:
    hybrid = kind is DroneKind.EXTINGUISHER
    return DroneState(
        drone_id=drone_id,
        kind=kind,
        role=role,
        position=Position(x=x, y=0.0),
        powertrain=Powertrain.HYBRID if hybrid else Powertrain.ELECTRIC,
        battery_pct=80.0,
        endurance_remaining_s=3600.0,
        fuel_remaining_kg=20.0 if hybrid else None,
        fuel_pct=80.0 if hybrid else None,
        water_remaining_l=water_remaining_l,
        epoch=epoch,
        sensors_ok={"thermal": True, "lidar": True},
        link_quality=0.9,
        status=DroneStatus.IDLE,
        timestamp=0.0,
    )


def test_leader_coordinator_takes_over_and_emits_hold_resume_transition():
    bus = LoopbackBus()
    clock = MutableClock()
    local = make_drone("d2")
    events = []
    coordinator = LeaderCoordinator(
        bus,
        clock,
        lambda: local,
        initial_leader_id="d1",
        epoch=5,
        heartbeat_period_s=0.5,
        missed_heartbeat_threshold=2,
        leader_timeout_s=1.0,
        on_hold=lambda: events.append("hold"),
        on_resume=lambda: events.append("resume"),
    )
    coordinator.start()
    sender = SwarmCommunicator(bus, clock, "d1")
    sender.publish(
        Heartbeat(state=make_drone("d1", role=SwarmRole.LEADER)),
        epoch=0,
    )

    clock.current_time = 1.0
    report = coordinator.tick()

    assert report.newly_lost == ("d1",)
    assert coordinator.leader_id == "d2"
    assert coordinator.epoch == 6
    assert events == ["hold", "resume"]
    assert any(
        message.msg_type is MessageType.LEADER_CLAIM
        and message.body.candidate_id == "d2"
        and message.body.epoch == 6
        for _, message in bus.published
    )
    coordinator.close()


def test_task_lifecycle_creates_auctions_and_tracks_ack_completion():
    bus = LoopbackBus()
    clock = MutableClock()
    target = Position(x=100.0, y=0.0)
    lifecycle = TaskLifecycleCoordinator(
        bus,
        clock,
        "leader",
        {},
        epoch=3,
        bid_window_s=2.0,
    )
    drone = make_drone(
        "ext1",
        kind=DroneKind.EXTINGUISHER,
        water_remaining_l=100.0,
    )
    responder = TaskBidResponder(
        bus,
        clock,
        lambda: drone,
        {"h1": target},
        reserve_s=0.0,
    )
    lifecycle.start()
    responder.start()
    reporter = SwarmCommunicator(bus, clock, "seek1")
    reporter.publish(
        HotspotReport(
            hotspot=Hotspot(
                hotspot_id="h1",
                position=target,
                peak_temp_c=600.0,
                est_area_m2=10.0,
                confidence=0.9,
                status=HotspotStatus.CONFIRMED,
                reported_by="seek1",
                first_seen=0.0,
                last_seen=0.0,
            )
        )
    )

    assert len(lifecycle.tasks) == 1
    lifecycle.tick()
    clock.current_time = 2.0
    assigned = lifecycle.tick()
    assert len(assigned) == 1
    assert assigned[0].status is TaskStatus.ASSIGNED
    assert assigned[0].assigned_to == "ext1"

    drone_comms = SwarmCommunicator(bus, clock, "ext1")
    drone_comms.publish(
        TaskAck(
            task_id=assigned[0].task_id,
            drone_id="ext1",
            accepted=True,
            epoch=3,
        ),
        epoch=3,
    )
    assert lifecycle.tasks[0].status is TaskStatus.IN_PROGRESS

    drone_comms.publish(
        TaskComplete(
            task_id=assigned[0].task_id,
            drone_id="ext1",
            success=True,
            result_status=HotspotStatus.SUPPRESSED,
        ),
        epoch=3,
    )
    assert lifecycle.tasks[0].status is TaskStatus.DONE
    assert lifecycle.hotspots[0].status is HotspotStatus.SUPPRESSED
    lifecycle.close()
    responder.close()


def test_refill_assignment_prefers_nearest_validated_water_and_reserves_station_bays():
    bus = LoopbackBus()
    clock = MutableClock()
    station_a = RefillStation(
        station_id="station-a",
        position=Position(x=10.0, y=0.0),
        water_available_l=500.0,
        fill_rate_lpm=100.0,
        total_bays=1,
        free_bays=1,
        agents_available=[ExtinguishingAgent.WATER],
        fuel_available_kg=100.0,
    )
    station_b = station_a.model_copy(
        update={
            "station_id": "station-b",
            "position": Position(x=20.0, y=0.0),
            "battery_swap_s": 90.0,
        },
        deep=True,
    )
    pond = WaterSource(
        source_id="pond",
        position=Position(x=2.0, y=0.0),
        kind=WaterSourceKind.POND,
        origin=WaterSourceOrigin.DETECTED,
        status=WaterSourceStatus.VALIDATED,
        est_volume_l=10000.0,
    )
    allocator = RefillAssignmentCoordinator(
        bus,
        clock,
        "leader",
        [station_a, station_b],
        [pond],
        epoch=4,
    )

    water_request = RefillRequest(
        drone_id="e1",
        position=Position(x=0.0, y=0.0),
        water_remaining_l=0.0,
    )
    water_decision = allocator.assign(water_request)
    assert water_decision.grant is not None
    assert water_decision.grant.target_kind is RefillTargetKind.WATER_SOURCE
    assert water_decision.grant.target_id == "pond"

    fuel_request = RefillRequest(
        drone_id="e2",
        position=Position(x=0.0, y=0.0),
        water_remaining_l=0.0,
        needs_fuel=True,
    )
    fuel_decision = allocator.assign(fuel_request)
    assert fuel_decision.grant is not None
    assert fuel_decision.grant.target_kind is RefillTargetKind.STATION
    assert fuel_decision.grant.target_id == "station-a"
    assert fuel_decision.grant.epoch == 4
    second_fuel_decision = allocator.assign(
        RefillRequest(
            drone_id="e3",
            position=Position(x=10.0, y=0.0),
            water_remaining_l=0.0,
            needs_fuel=True,
        )
    )
    assert second_fuel_decision.grant is not None
    assert second_fuel_decision.grant.target_id == "station-b"
    assert allocator.release_reservation("e2")
    assert allocator.release_reservation("e3")
    assert not allocator.release_reservation("e2")
    battery_decision = allocator.assign_battery_swap(
        "s1",
        Position(x=0.0, y=0.0),
    )
    assert battery_decision.grant is not None
    assert battery_decision.grant.target_kind is RefillTargetKind.STATION
    assert battery_decision.grant.target_id == "station-b"
