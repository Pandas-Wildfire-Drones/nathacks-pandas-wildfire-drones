from __future__ import annotations

import pytest

from contracts import (
    DroneKind,
    DroneState,
    Position,
    Powertrain,
    SwarmMessage,
    TaskComplete,
    TaskStatus,
    TaskType,
    decode,
    encode,
    topic_matches,
)
from swarm import (
    SwarmCommunicator,
    TaskBidResponder,
    TaskLifecycleCoordinator,
    form_clusters_for_area,
    generate_grid_zones,
)


class MutableClock:
    def __init__(self) -> None:
        self.current_time = 0.0

    def now(self) -> float:
        return self.current_time


class LoopbackSubscription:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class LoopbackBus:
    def __init__(self) -> None:
        self.subscriptions = []
        self.published: list[tuple[str, SwarmMessage]] = []

    def publish(self, topic: str, message: SwarmMessage) -> None:
        self.published.append((topic, message))
        raw = encode(message)
        for pattern, handler, subscription in list(self.subscriptions):
            if not subscription.cancelled and topic_matches(pattern, topic):
                handler(decode(raw))

    def subscribe(self, pattern, handler):
        subscription = LoopbackSubscription()
        self.subscriptions.append((pattern, handler, subscription))
        return subscription


def make_seeker(drone_id: str, x: float = 0.0) -> DroneState:
    return DroneState(
        drone_id=drone_id,
        kind=DroneKind.SEEKER,
        powertrain=Powertrain.ELECTRIC,
        position=Position(x=x, y=0.0),
        battery_pct=90.0,
        endurance_remaining_s=3600.0,
        status="IDLE",
        timestamp=0.0,
    )


def test_generate_grid_zones_tiles_rectangle_and_clips_edges():
    zones = generate_grid_zones(
        Position(x=0.0, y=0.0, z=5.0),
        Position(x=25.0, y=18.0, z=5.0),
        zone_width_m=10.0,
        zone_height_m=10.0,
    )

    assert len(zones) == 6
    assert [zone.zone_id for zone in zones] == [
        "zone-r0000-c0000",
        "zone-r0000-c0001",
        "zone-r0000-c0002",
        "zone-r0001-c0000",
        "zone-r0001-c0001",
        "zone-r0001-c0002",
    ]
    areas = [
        (max(point.x for point in zone.boundary) - min(point.x for point in zone.boundary))
        * (max(point.y for point in zone.boundary) - min(point.y for point in zone.boundary))
        for zone in zones
    ]
    assert sum(areas) == pytest.approx(25.0 * 18.0)
    assert all(point.z == 5.0 for zone in zones for point in zone.boundary)


@pytest.mark.parametrize(
    ("southwest", "northeast", "kwargs"),
    [
        (Position(x=0, y=0), Position(x=10, y=10), {"zone_width_m": 0}),
        (
            Position(x=0, y=0),
            Position(x=10, y=10),
            {"zone_width_m": 5, "zone_height_m": -1},
        ),
        (Position(x=1, y=0), Position(x=0, y=10), {"zone_width_m": 5}),
        (
            Position(x=0, y=0, z=1),
            Position(x=10, y=10, z=2),
            {"zone_width_m": 5},
        ),
    ],
)
def test_generate_grid_zones_rejects_invalid_input(southwest, northeast, kwargs):
    with pytest.raises(ValueError):
        generate_grid_zones(southwest, northeast, **kwargs)


def test_form_clusters_for_area_generates_and_assigns_zones():
    formation = form_clusters_for_area(
        [
            make_seeker("d0", 0),
            make_seeker("d1", 1),
            make_seeker("d2", 99),
            make_seeker("d3", 100),
        ],
        Position(x=0, y=0),
        Position(x=100, y=10),
        max_members=2,
        zone_width_m=50,
        now=0,
    )

    assert len(formation.clusters) == 2
    assert all(len(cluster.member_ids) <= 2 for cluster in formation.clusters)
    assert {zone.zone_id for zone in formation.zones} == {
        "zone-r0000-c0000",
        "zone-r0000-c0001",
    }
    assert all(zone.cluster_id is not None for zone in formation.zones)
    assert all(
        zone.zone_id in cluster.zone_ids
        for zone in formation.zones
        for cluster in formation.clusters
        if zone.cluster_id == cluster.cluster_id
    )


def test_zone_scan_tasks_are_bid_assigned_and_marked_complete():
    bus = LoopbackBus()
    clock = MutableClock()
    zones = generate_grid_zones(
        Position(x=0, y=0),
        Position(x=10, y=10),
        zone_width_m=10,
    )
    lifecycle = TaskLifecycleCoordinator(
        bus,
        clock,
        "leader",
        {},
        epoch=3,
        bid_window_s=2.0,
        zones=zones,
    )
    seeker = make_seeker("seeker")
    responder = TaskBidResponder(
        bus,
        clock,
        lambda: seeker,
        {},
        reserve_s=0.0,
        zones=zones,
    )
    attacker = DroneState(
        drone_id="attacker",
        kind=DroneKind.EXTINGUISHER,
        powertrain=Powertrain.HYBRID,
        position=Position(x=0.0, y=0.0),
        battery_pct=90.0,
        endurance_remaining_s=3600.0,
        fuel_remaining_kg=10.0,
        fuel_pct=90.0,
        water_remaining_l=100.0,
        timestamp=0.0,
    )
    attacker_responder = TaskBidResponder(
        bus,
        clock,
        lambda: attacker,
        {},
        reserve_s=0.0,
        zones=zones,
    )
    lifecycle.start()
    responder.start()
    attacker_responder.start()

    scan_tasks = [task for task in lifecycle.tasks if task.type is TaskType.SCAN_ZONE]
    assert len(scan_tasks) == 1
    assert scan_tasks[0].target_zone_id == zones[0].zone_id

    lifecycle.tick()
    clock.current_time = 2.0
    assigned = lifecycle.tick()
    assert len(assigned) == 1
    assert assigned[0].assigned_to == "seeker"
    assert lifecycle.zones[0].assigned_to == "seeker"

    SwarmCommunicator(bus, clock, "seeker").publish(
        TaskComplete(
            task_id=assigned[0].task_id,
            drone_id="seeker",
            success=True,
        ),
        epoch=3,
    )

    assert lifecycle.tasks[0].status is TaskStatus.DONE
    assert lifecycle.zones[0].last_scanned == clock.current_time
    assert lifecycle.zones[0].assigned_to is None

    recovered_lifecycle = TaskLifecycleCoordinator(
        bus,
        clock,
        "new-leader",
        {},
        epoch=4,
        zones=lifecycle.zones,
    )
    assert len(recovered_lifecycle.tasks) == 1
    assert recovered_lifecycle.tasks[0].status is TaskStatus.DONE

    recovered_lifecycle.close()
    lifecycle.close()
    responder.close()
    attacker_responder.close()


def test_successfully_scanned_zone_does_not_get_a_new_scan_task():
    bus = LoopbackBus()
    clock = MutableClock()
    scanned = generate_grid_zones(
        Position(x=0, y=0),
        Position(x=10, y=10),
        zone_width_m=10,
    )[0].model_copy(update={"last_scanned": 4.0})

    lifecycle = TaskLifecycleCoordinator(
        bus,
        clock,
        "leader",
        {},
        epoch=1,
        zones=[scanned],
    )

    assert len(lifecycle.tasks) == 1
    assert lifecycle.tasks[0].type is TaskType.SCAN_ZONE
    assert lifecycle.tasks[0].status is TaskStatus.DONE
    lifecycle.close()
