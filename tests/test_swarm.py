"""Tests for the pure-Python swarm coordination package."""

from __future__ import annotations

from contracts import (
    AlertKind,
    DroneKind,
    DroneState,
    DroneStatus,
    Hotspot,
    HotspotStatus,
    LeaderClaim,
    MessageType,
    Position,
    Powertrain,
    SwarmRole,
    Task,
    TaskBid,
    TaskStatus,
    TaskType,
    decode,
    encode,
    topic_matches,
)
from swarm import (
    TaskAuctionCoordinator,
    TaskBidResponder,
    FailureMonitor,
    SSIAuctionRound,
    SwarmCommunicator,
    allocate_tasks,
    bid_for_task,
    create_tasks_from_hotspots,
    elect_leader,
    prioritize_hotspots,
    score_hotspot,
    select_leader_claim,
)


def make_drone(
    drone_id: str,
    *,
    kind: DroneKind = DroneKind.SEEKER,
    x: float = 0.0,
    status: DroneStatus = DroneStatus.IDLE,
    role: SwarmRole = SwarmRole.FOLLOWER,
    current_task_id: str | None = None,
    water_remaining_l: float | None = None,
    endurance_remaining_s: float | None = 3600.0,
    epoch: int = 0,
) -> DroneState:
    return DroneState(
        drone_id=drone_id,
        kind=kind,
        powertrain=Powertrain.HYBRID if kind is DroneKind.EXTINGUISHER else Powertrain.ELECTRIC,
        role=role,
        position=Position(x=x, y=0.0, z=0.0),
        battery_pct=80.0,
        endurance_remaining_s=endurance_remaining_s,
        fuel_remaining_kg=20.0 if kind is DroneKind.EXTINGUISHER else None,
        fuel_pct=80.0 if kind is DroneKind.EXTINGUISHER else None,
        water_remaining_l=water_remaining_l,
        current_task_id=current_task_id,
        epoch=epoch,
        sensors_ok={"thermal": True, "lidar": True},
        link_quality=0.9,
        status=status,
        timestamp=0.0,
    )


def make_hotspot(
    hotspot_id: str,
    *,
    confidence: float = 0.8,
    last_seen: float = 10.0,
    status: HotspotStatus = HotspotStatus.CONFIRMED,
) -> Hotspot:
    return Hotspot(
        hotspot_id=hotspot_id,
        position=Position(x=1.0, y=0.0, z=0.0),
        peak_temp_c=500.0,
        est_area_m2=10.0,
        confidence=confidence,
        status=status,
        first_seen=0.0,
        last_seen=last_seen,
    )


def make_task(
    task_id: str,
    *,
    task_type: TaskType = TaskType.INVESTIGATE,
    target_id: str = "h1",
    priority: float = 0.5,
    status: TaskStatus = TaskStatus.PENDING,
    assigned_to: str | None = None,
) -> Task:
    target_field = (
        {"target_zone_id": target_id}
        if task_type is TaskType.SCAN_ZONE
        else {"target_hotspot_id": target_id}
    )
    return Task(
        task_id=task_id,
        type=task_type,
        priority=priority,
        status=status,
        assigned_to=assigned_to,
        created_at=0.0,
        **target_field,
    )


def test_prioritization_scores_staleness_and_terminal_status():
    fresh = make_hotspot("fresh", last_seen=100.0)
    stale = make_hotspot("stale", last_seen=0.0)
    inactive = make_hotspot("out", status=HotspotStatus.VERIFIED_OUT)

    ranked = prioritize_hotspots([stale, inactive, fresh], now=100.0)

    assert [hotspot.hotspot_id for hotspot in ranked] == ["fresh", "stale", "out"]
    assert ranked[-1].priority_score == 0.0
    assert fresh.priority_score == 0.0
    assert score_hotspot(fresh, now=100.0) > score_hotspot(stale, now=100.0)


def test_task_creation_maps_hotspot_status_and_avoids_active_duplicates():
    detected = make_hotspot("h1", status=HotspotStatus.DETECTED, last_seen=12.0)
    confirmed = make_hotspot("h2", status=HotspotStatus.CONFIRMED, last_seen=13.0)
    inactive = make_hotspot("h3", status=HotspotStatus.SUPPRESSED)
    existing = make_task("active-investigation", target_id="h1")

    tasks = create_tasks_from_hotspots(
        [detected, confirmed, inactive],
        now=20.0,
        existing_tasks=[existing],
        deadline_after_s=30.0,
    )

    assert len(tasks) == 1
    assert tasks[0].type is TaskType.EXTINGUISH
    assert tasks[0].target_hotspot_id == "h2"
    assert 0.0 <= tasks[0].priority <= 1.0
    assert tasks[0].deadline == 50.0


def test_election_excludes_faulted_drones_and_advances_epoch():
    healthy = make_drone("d01", epoch=2)
    better = make_drone("d02", epoch=2, endurance_remaining_s=7200.0)
    faulted = make_drone("d99", status=DroneStatus.FAULT, endurance_remaining_s=72000.0)

    claim = elect_leader([healthy, better, faulted])

    assert claim.candidate_id == "d02"
    assert claim.epoch == 3
    assert select_leader_claim([
        LeaderClaim(candidate_id="z", epoch=3, score=0.1),
        LeaderClaim(candidate_id="a", epoch=3, score=0.9),
        LeaderClaim(candidate_id="x", epoch=2, score=1.0),
    ]).candidate_id == "a"


def test_task_allocation_respects_kind_priority_and_one_task_per_drone():
    seeker = make_drone("s01", x=0.0)
    second_seeker = make_drone("s02", x=10.0)
    investigate = make_task("t-investigate", priority=0.9)
    scan = make_task(
        "t-scan",
        task_type=TaskType.SCAN_ZONE,
        target_id="z1",
        priority=0.5,
    )
    assigned = allocate_tasks(
        [scan, investigate],
        [seeker, second_seeker],
        {"h1": Position(x=1.0, y=0.0), "z1": Position(x=20.0, y=0.0)},
        now=0.0,
        epoch=4,
        reserve_s=0.0,
        cruise_speed_mps=10.0,
    )

    assert [(task.task_id, task.assigned_to) for task in assigned] == [
        ("t-investigate", "s01"),
        ("t-scan", "s02"),
    ]
    assert all(task.status is TaskStatus.ASSIGNED and task.epoch == 4 for task in assigned)
    assert investigate.status is TaskStatus.PENDING


def test_extinguish_bid_checks_extinguisher_water_and_endurance():
    task = make_task("drop", task_type=TaskType.EXTINGUISH)
    target = Position(x=100.0, y=0.0)
    seeker_bid = bid_for_task(
        task,
        make_drone("s1"),
        target,
        now=0.0,
        reserve_s=0.0,
    )
    empty_extinguisher_bid = bid_for_task(
        task,
        make_drone("e1", kind=DroneKind.EXTINGUISHER, water_remaining_l=0.0),
        target,
        now=0.0,
        reserve_s=0.0,
    )
    capable_bid = bid_for_task(
        task,
        make_drone("e2", kind=DroneKind.EXTINGUISHER, water_remaining_l=50.0),
        target,
        now=0.0,
        reserve_s=0.0,
    )

    assert not seeker_bid.feasible
    assert not empty_extinguisher_bid.feasible
    assert capable_bid.feasible


def test_bid_is_normalized_marginal_travel_time_divided_by_priority():
    task = make_task("t1", priority=0.5)
    bid = bid_for_task(
        task,
        make_drone("d1"),
        Position(x=100.0, y=0.0),
        now=0.0,
        reserve_s=0.0,
        cruise_speed_mps=10.0,
        travel_time_normalizer_s=100.0,
    )

    assert bid.feasible
    assert bid.eta_s == 10.0
    assert bid.cost == 10.0 / 110.0 / 0.5


def test_zero_priority_task_is_not_auctioned():
    task = make_task("t1", priority=0.0)
    bid = bid_for_task(
        task,
        make_drone("d1"),
        Position(x=10.0, y=0.0),
        now=0.0,
    )

    assert not bid.feasible


def test_task_bid_includes_a_return_leg_and_endurance_reserve():
    task = make_task("t1")
    drone = make_drone("d1", endurance_remaining_s=25.0)

    bid = bid_for_task(
        task,
        drone,
        Position(x=100.0, y=0.0),
        now=0.0,
        reserve_s=10.0,
        cruise_speed_mps=10.0,
    )

    assert not bid.feasible


def test_ssi_round_accepts_one_bid_per_drone_before_deadline_and_breaks_ties():
    auction = SSIAuctionRound("t1", bid_deadline=5.0)
    bids = [
        TaskBid(task_id="t1", bidder_id="d2", cost=0.4),
        TaskBid(task_id="t1", bidder_id="d1", cost=0.4),
    ]

    assert auction.submit_bid(bids[0], received_at=3.0)
    assert not auction.submit_bid(
        TaskBid(task_id="t1", bidder_id="d2", cost=0.1),
        received_at=4.0,
    )
    assert auction.submit_bid(bids[1], received_at=5.0)
    assert not auction.submit_bid(
        TaskBid(task_id="t1", bidder_id="d3", cost=0.1),
        received_at=5.1,
    )
    winner = auction.close(now=5.0)
    assert winner is not None
    assert winner.bidder_id == "d1"
    assert not auction.submit_bid(
        TaskBid(task_id="t1", bidder_id="d4", cost=0.0),
        received_at=5.0,
    )


def test_failure_monitor_requeues_active_work_once_and_detects_leader_loss():
    monitor = FailureMonitor(heartbeat_timeout_s=3.0)
    leader = make_drone(
        "d1",
        role=SwarmRole.LEADER,
        status=DroneStatus.TRANSIT,
        current_task_id="t1",
    )
    task = make_task(
        "t1",
        status=TaskStatus.IN_PROGRESS,
        assigned_to="d1",
    )
    follower = make_drone("d2", epoch=2)
    assert monitor.observe_heartbeat(leader, received_at=10.0)
    assert monitor.observe_heartbeat(follower, received_at=12.0)

    first = monitor.check_timeouts(now=13.0, tasks=[task])
    repeated = monitor.check_timeouts(now=14.0, tasks=[task])

    assert first.newly_lost == ("d1",)
    assert first.hold_after_current_task
    assert first.alerts[0].kind is AlertKind.LEADER_LOST
    assert first.requeued_tasks[0].status is TaskStatus.PENDING
    assert first.requeued_tasks[0].assigned_to is None
    assert first.leader_claim is not None
    assert first.leader_claim.candidate_id == "d2"
    assert first.leader_claim.epoch == 3
    assert repeated.newly_lost == ()
    assert monitor.lost_drone_ids == ("d1",)
    assert monitor.observe_heartbeat(leader, received_at=15.0)
    assert monitor.lost_drone_ids == ()


def test_failure_monitor_uses_multiple_missed_heartbeat_intervals():
    monitor = FailureMonitor(
        heartbeat_period_s=0.5,
        missed_heartbeat_threshold=3,
    )
    drone = make_drone("d1")
    monitor.observe_heartbeat(drone, received_at=1.0)

    assert monitor.check_timeouts(now=2.4).newly_lost == ()
    assert monitor.check_timeouts(now=2.5).newly_lost == ("d1",)


class FakeClock:
    def now(self) -> float:
        return 12.5


class FakeSubscription:
    def cancel(self) -> None:
        pass


class FakeBus:
    def __init__(self) -> None:
        self.published = []
        self.subscribers = []

    def publish(self, topic, msg) -> None:
        self.published.append((topic, msg))

    def subscribe(self, pattern, handler):
        self.subscribers.append((pattern, handler))
        return FakeSubscription()


def test_communicator_routes_messages_with_increasing_sequence_and_type_filter():
    from contracts import Heartbeat

    bus = FakeBus()
    communicator = SwarmCommunicator(bus, FakeClock(), "d1")
    first = communicator.publish(Heartbeat(state=make_drone("d1")))
    second = communicator.publish(Heartbeat(state=make_drone("d1")))

    received = []
    communicator.subscribe(
        "drone/*/heartbeat",
        received.append,
        message_types=[MessageType.HEARTBEAT],
    )
    bus.subscribers[0][1](first)

    assert [first.seq, second.seq] == [0, 1]
    assert bus.published[0][0] == "drone/d1/heartbeat"
    assert received == [first]


class MutableClock:
    def __init__(self, now: float = 0.0) -> None:
        self.current_time = now

    def now(self) -> float:
        return self.current_time


class LoopbackBus:
    """Synchronous test bus that still serializes and validates every delivery."""

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


class LoopbackSubscription:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def test_bus_auction_announces_collects_a_bid_and_assigns_winner():
    bus = LoopbackBus()
    clock = MutableClock()
    target = Position(x=100.0, y=0.0)
    drone = make_drone("s1", endurance_remaining_s=3600.0)
    responder = TaskBidResponder(
        bus,
        clock,
        lambda: drone,
        {"h1": target},
        reserve_s=0.0,
        cruise_speed_mps=10.0,
    )
    responder.start()
    coordinator = TaskAuctionCoordinator(
        bus,
        clock,
        "leader",
        {"h1": target},
        epoch=1,
        bid_window_s=2.0,
    )

    announced = coordinator.start_auction(make_task("t1", priority=0.8))
    assert announced.msg_type is MessageType.TASK_ANNOUNCE
    assert coordinator.tick() is None

    clock.current_time = 2.0
    outcome = coordinator.tick()

    assert outcome is not None
    assert outcome.winner_id == "s1"
    assert outcome.assignment is not None
    assert outcome.assignment.status is TaskStatus.ASSIGNED
    assert bus.published[-1][0] == "drone/s1/cmd"
    assert bus.published[-1][1].msg_type is MessageType.TASK_ASSIGN

    coordinator.close()
    responder.close()
