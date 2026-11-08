import pytest

import autonomy.mission_coordinator as mission_coordinator
from autonomy.cbs_planner import PathRequest
from autonomy.drone_controller import DroneController
from autonomy.mission_coordinator import MissionPlanningContext
from autonomy.swarm_adapter import SwarmAutonomyAdapter
from autonomy.water_drop import WaterDropLimits
from contracts.config import default_extinguisher_spec
from contracts.entities import DroneState, Hotspot, Task
from contracts.enums import (
    DroneKind,
    DroneStatus,
    ExtinguishingAgent,
    HotspotStatus,
    Powertrain,
    TaskStatus,
    TaskType,
)
from contracts.geometry import GridSpec, Position
from contracts.messages import PathClaim, SwarmMessage, TaskAssign, TaskComplete


class FakeBus:
    def __init__(self, events=None):
        self.published = []
        self.subscriptions = []
        self.events = events if events is not None else []

    def subscribe(self, pattern, handler):
        self.subscriptions.append((pattern, handler))

    def publish(self, topic, message):
        self.published.append((topic, message))
        if topic == "swarm/path_claims":
            self.events.append("path_claim")


class FakeAutopilot:
    def __init__(self, events):
        self.events = events
        self.arrived = False
        self.goto_targets = []
        self.hold_calls = 0
        self.return_calls = 0

    def position(self):
        return Position(x=0, y=0, z=20)

    def heading_deg(self):
        return 0

    def speed_mps(self):
        return 0

    def battery_pct(self):
        return 80

    def endurance_remaining_s(self):
        return 3600

    def goto(self, target, speed_mps=None):
        self.goto_targets.append(target)
        self.events.append("begin_extinguish")

    def hold(self):
        self.hold_calls += 1

    def return_to_launch(self):
        self.return_calls += 1

    def at_target(self, tolerance_m=2.0):
        return self.arrived


class FakePayload:
    def __init__(self):
        self.release_calls = []

    def water_remaining_l(self):
        return 100

    def agent(self):
        return ExtinguishingAgent.WATER

    def release(self, volume_l, at):
        self.release_calls.append((volume_l, at))
        return volume_l

    def fill(self, volume_l, agent=ExtinguishingAgent.WATER):
        return volume_l


def make_drone(**updates):
    values = dict(
        drone_id="d1",
        kind=DroneKind.EXTINGUISHER,
        position=Position(x=0, y=0, z=20),
        speed_mps=0,
        powertrain=Powertrain.HYBRID,
        battery_pct=80,
        endurance_remaining_s=3600,
        fuel_remaining_kg=200,
        fuel_pct=95,
        water_remaining_l=100,
        agent=ExtinguishingAgent.WATER,
        status=DroneStatus.IDLE,
        epoch=4,
        sensors_ok={"thermal": True, "lidar": True},
        link_quality=1,
        timestamp=0,
    )
    values.update(updates)
    return DroneState(**values)


def make_hotspot(**updates):
    values = dict(
        hotspot_id="h1",
        position=Position(x=12, y=10, z=20),
        peak_temp_c=500,
        confidence=0.95,
        status=HotspotStatus.CONFIRMED,
        assigned_to="d1",
        first_seen=0,
        last_seen=0,
    )
    values.update(updates)
    return Hotspot(**values)


def make_task(**updates):
    values = dict(
        task_id="t1",
        type=TaskType.EXTINGUISH,
        status=TaskStatus.ASSIGNED,
        assigned_to="d1",
        target_hotspot_id="h1",
        created_at=0,
        epoch=4,
        volume_l=8,
    )
    values.update(updates)
    return Task(**values)


def make_context(occupancy_grid=None):
    grid_spec = GridSpec(
        origin=Position(x=-0.5, y=-0.5, z=0),
        resolution_m=1,
        rows=32,
        cols=32,
    )
    return MissionPlanningContext(
        occupancy_grid=occupancy_grid or [[0] * 32 for _ in range(32)],
        grid_spec=grid_spec,
        base_position=Position(x=0, y=0, z=20),
        platform_spec=default_extinguisher_spec(),
        minimum_separation_m=2,
        time_step_s=1,
        altitude_band_m=(20, 30),
    )


def make_adapter(drone=None, *, context=None, bus=None, events=None):
    drone = drone or make_drone()
    events = events if events is not None else []
    bus = bus or FakeBus()
    autopilot = FakeAutopilot(events)
    payload = FakePayload()
    controller = DroneController(
        WaterDropLimits(
            min_confidence=0.7,
            max_distance_m=100,
            min_altitude_agl_m=10,
            max_altitude_agl_m=100,
            max_speed_mps=20,
        ),
        autopilot,
        payload,
    )
    adapter = SwarmAutonomyAdapter(
        drone,
        controller,
        bus,
        planning_context=context or make_context(),
        altitude_agl_provider=lambda _drone: 15,
    )
    adapter.state.current_hotspot = make_hotspot()
    return adapter, bus, controller, autopilot, payload


def make_assignment(task=None):
    return SwarmMessage.create(
        sender_id="leader",
        body=TaskAssign(
            task=task or make_task(),
            epoch=4,
        ),
        now=0,
        epoch=4,
    )


def test_assignment_publishes_cbs_claim_before_controller_transit():
    events = []
    bus = FakeBus(events)
    adapter, _, _, autopilot, _ = make_adapter(
        bus=bus,
        events=events,
    )

    result = adapter.handle_message(make_assignment())

    assert result.status == "TRANSIT"
    assert events[:2] == ["path_claim", "begin_extinguish"]
    claim = next(
        message.body
        for topic, message in bus.published
        if topic == "swarm/path_claims"
    )
    assert isinstance(claim, PathClaim)
    assert claim.waypoints[0] == Position(x=0, y=0, z=25)
    assert claim.waypoints[-1] == make_hotspot().position
    assert claim.altitude_band_m == (20, 30)
    assert autopilot.goto_targets == [claim.waypoints[0]]


@pytest.mark.parametrize(
    "drone_updates",
    [
        {"battery_pct": 10},
        {"water_remaining_l": 1},
        {"fuel_remaining_kg": 1},
        {"endurance_remaining_s": 1},
    ],
    ids=["battery", "water", "hybrid-fuel", "endurance"],
)
def test_resource_failure_blocks_cbs_and_mission(
    monkeypatch,
    drone_updates,
):
    def unexpected_planning(*args, **kwargs):
        raise AssertionError("CBS must not run after resource failure")

    monkeypatch.setattr(mission_coordinator, "plan_paths", unexpected_planning)
    adapter, bus, _, autopilot, payload = make_adapter(
        drone=make_drone(**drone_updates),
    )

    result = adapter.handle_message(make_assignment())

    assert result is None
    assert autopilot.events == []
    assert payload.release_calls == []
    assert not any(
        topic == "swarm/path_claims"
        for topic, _ in bus.published
    )
    assert any(topic == "ground/alerts" for topic, _ in bus.published)
    assert adapter.state.current_task is None


def test_conflicting_remote_claim_blocks_publication_and_execution():
    adapter, bus, _, autopilot, _ = make_adapter()
    adapter.handle_path_claim(
        make_assignment(),
        PathClaim(
            drone_id="d2",
            waypoints=[
                Position(x=0, y=0, z=25),
                Position(x=12, y=10, z=25),
            ],
            valid_from=0,
            valid_until=20,
            altitude_band_m=(20, 30),
        ),
    )

    result = adapter.handle_message(make_assignment())

    assert result is None
    assert autopilot.events == []
    assert not any(
        topic == "swarm/path_claims"
        for topic, _ in bus.published
    )
    assert adapter.state.current_task is None


def test_path_claim_publication_failure_blocks_controller_execution():
    class FailingClaimBus(FakeBus):
        def publish(self, topic, message):
            if topic == "swarm/path_claims":
                raise RuntimeError("bus unavailable")
            super().publish(topic, message)

    bus = FailingClaimBus()
    adapter, _, _, autopilot, _ = make_adapter(bus=bus)

    result = adapter.handle_message(make_assignment())

    assert result is None
    assert autopilot.events == []
    assert any(topic == "ground/alerts" for topic, _ in bus.published)
    assert adapter.state.current_task is None


def test_invalid_hotspot_assignment_blocks_planning():
    adapter, bus, _, autopilot, _ = make_adapter()
    adapter.state.current_hotspot = make_hotspot(assigned_to="d2")

    result = adapter.handle_message(make_assignment())

    assert result is None
    assert autopilot.events == []
    assert not any(
        topic == "swarm/path_claims"
        for topic, _ in bus.published
    )
    assert any(topic == "ground/alerts" for topic, _ in bus.published)


def test_cbs_failure_publishes_alert_without_starting_mission():
    blocked = [[0] * 32 for _ in range(32)]
    for row in blocked:
        row[5] = 1
    adapter, bus, _, autopilot, _ = make_adapter(
        context=make_context(blocked),
    )

    result = adapter.handle_message(make_assignment())

    assert result is None
    assert autopilot.events == []
    assert not any(
        topic == "swarm/path_claims"
        for topic, _ in bus.published
    )
    assert any(topic == "ground/alerts" for topic, _ in bus.published)
    assert adapter.state.current_task is None


@pytest.mark.parametrize(
    ("drone_updates", "expected_action"),
    [
        ({"status": DroneStatus.FAULT}, "hold"),
        ({"status": DroneStatus.LOST}, "hold"),
        ({"link_quality": 0.1}, "return"),
    ],
    ids=["fault", "lost", "degraded-link"],
)
def test_health_failure_blocks_assignment_before_planning(
    drone_updates,
    expected_action,
):
    adapter, bus, _, autopilot, _ = make_adapter(
        drone=make_drone(**drone_updates),
    )

    result = adapter.handle_message(make_assignment())

    assert result is None
    assert autopilot.events == []
    assert not any(
        topic == "swarm/path_claims"
        for topic, _ in bus.published
    )
    assert getattr(autopilot, f"{expected_action}_calls") == 1


def test_successful_mission_reaches_drop_and_publishes_task_complete():
    adapter, bus, _, autopilot, payload = make_adapter()

    result = adapter.handle_message(make_assignment())
    assert result.status == "TRANSIT"

    claim = next(
        message.body
        for topic, message in bus.published
        if topic == "swarm/path_claims"
    )
    assert isinstance(claim, PathClaim)
    assert autopilot.goto_targets == [claim.waypoints[0]]
    assert payload.release_calls == []

    for _ in claim.waypoints[1:]:
        autopilot.arrived = True
        result = adapter.step()
        assert autopilot.goto_targets == claim.waypoints[
            : len(autopilot.goto_targets)
        ]
        assert result.status == "TRANSIT"
        assert payload.release_calls == []

    result = adapter.step()
    assert result.status == "DONE"
    assert autopilot.goto_targets == claim.waypoints
    assert payload.release_calls
    completes = [
        message.body
        for _, message in bus.published
        if isinstance(message.body, TaskComplete)
    ]
    assert len(completes) == 1
    assert completes[0].success


def test_missing_agl_measurement_blocks_final_release():
    adapter, bus, _, autopilot, payload = make_adapter()
    adapter.altitude_agl_provider = None
    result = adapter.handle_message(make_assignment())
    assert result.status == "TRANSIT"
    autopilot.arrived = True

    result = adapter.step()

    assert result is None
    assert payload.release_calls == []
    assert autopilot.hold_calls == 1
    assert any(topic == "ground/alerts" for topic, _ in bus.published)


def test_dynamic_replanning_uses_existing_conflict_group_replanner():
    adapter, _, _, _, _ = make_adapter()
    grid = [[0, 0, 0] for _ in range(3)]
    grid_spec = GridSpec(
        origin=Position(x=0, y=0, z=0),
        resolution_m=1,
        rows=3,
        cols=3,
    )

    def cell(row, col):
        return Position(x=col + 0.5, y=row + 0.5, z=0)

    requests = [
        PathRequest(
            drone_id="east",
            start=cell(1, 0),
            goal=cell(1, 2),
            speed_mps=1,
        ),
        PathRequest(
            drone_id="west",
            start=cell(1, 2),
            goal=cell(1, 0),
            speed_mps=1,
        ),
    ]
    existing = {
        "east": [cell(1, 0), cell(1, 1), cell(1, 2)],
        "west": [cell(1, 2), cell(1, 1), cell(1, 0)],
    }

    replanned = adapter.replan_conflicting_groups(
        requests,
        existing,
        grid,
        grid_spec,
        minimum_separation_m=1.1,
        conflict_group_threshold_m=2,
        time_step_s=1,
        max_timesteps=20,
    )

    assert replanned is not None
    assert replanned["east"][-1] == cell(1, 2)
    assert replanned["west"][-1] == cell(1, 0)
    assert replanned != existing


def test_new_conflicting_remote_claim_replans_and_republishes_active_route():
    context = MissionPlanningContext(
        occupancy_grid=[[0] * 3 for _ in range(3)],
        grid_spec=GridSpec(
            origin=Position(x=-0.5, y=-0.5, z=0),
            resolution_m=1,
            rows=3,
            cols=3,
        ),
        base_position=Position(x=0, y=0, z=20),
        platform_spec=default_extinguisher_spec(),
        minimum_separation_m=1.5,
        time_step_s=1,
        altitude_band_m=(20, 30),
    )
    events = []
    bus = FakeBus(events)
    adapter, _, controller, autopilot, _ = make_adapter(
        context=context,
        bus=bus,
        events=events,
    )
    adapter.state.current_hotspot = make_hotspot(
        position=Position(x=2, y=2, z=20),
    )
    result = adapter.handle_message(make_assignment())
    assert result.status == "TRANSIT"
    original_path = controller.active_path
    assert original_path is not None

    remote_claim = PathClaim(
        drone_id="d2",
        waypoints=[
            Position(x=2, y=0, z=25),
            Position(x=2, y=1, z=25),
            Position(x=1, y=2, z=25),
            Position(x=0, y=2, z=25),
        ],
        valid_from=0,
        valid_until=3,
        altitude_band_m=(20, 30),
    )
    adapter.state.remote_path_claims.append(remote_claim)
    assert adapter.check_path_conflicts(
        list(original_path),
        context.minimum_separation_m,
        valid_from=adapter.now,
        time_step_s=context.time_step_s,
        altitude_band_m=context.altitude_band_m,
    )
    adapter.state.remote_path_claims.clear()

    adapter.handle_path_claim(
        make_assignment(),
        remote_claim,
    )

    claims = [
        message.body
        for topic, message in bus.published
        if topic == "swarm/path_claims"
    ]
    assert len(claims) == 2
    replacement_claim = claims[-1]
    assert isinstance(replacement_claim, PathClaim)
    assert replacement_claim.waypoints != list(original_path)
    assert controller.active_path == tuple(replacement_claim.waypoints)
    assert autopilot.hold_calls >= 1
    assert autopilot.goto_targets[-1] == replacement_claim.waypoints[0]
    assert events[-2:] == ["path_claim", "begin_extinguish"]
    assert adapter.check_path_conflicts(
        replacement_claim.waypoints,
        adapter.planning_context.minimum_separation_m,
        valid_from=adapter.now,
        time_step_s=adapter.planning_context.time_step_s,
        altitude_band_m=adapter.planning_context.altitude_band_m,
    ) == []
