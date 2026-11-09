import pytest

from autonomy.drone_controller import DroneController
from autonomy.mission_coordinator import MissionPlanningContext
from autonomy.swarm_adapter import SwarmAutonomyAdapter
from autonomy.water_drop import WaterDropLimits
from contracts.config import default_extinguisher_spec
from contracts.entities import (
    DroneState,
    Hotspot,
    RefillStation,
    Task,
    WaterSource,
)
from contracts.enums import (
    DroneKind,
    DroneStatus,
    ExtinguishingAgent,
    HotspotStatus,
    Powertrain,
    RefillTargetKind,
    TaskStatus,
    TaskType,
    WaterSourceKind,
    WaterSourceOrigin,
    WaterSourceStatus,
)
from contracts.geometry import GridSpec, Position
from contracts.messages import (
    PathClaim,
    RefillGrant,
    RefillRequest,
    SwarmMessage,
    TaskAssign,
    TaskComplete,
)


class FakeBus:
    def __init__(self):
        self.published = []
        self.subscriptions = []

    def subscribe(self, topic, handler):
        self.subscriptions.append((topic, handler))

    def publish(self, topic, message):
        self.published.append((topic, message))


class FakeAutopilot:
    def __init__(self):
        self.arrived = False
        self.goto_targets = []
        self.hold_calls = 0
        self.return_calls = 0
        self.fail_navigation = False

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
        if self.fail_navigation:
            raise RuntimeError("navigation unavailable")
        self.goto_targets.append(target)

    def hold(self):
        self.hold_calls += 1

    def return_to_launch(self):
        self.return_calls += 1

    def at_target(self, tolerance_m=2.0):
        return self.arrived


class FakePayload:
    def __init__(self, water_l=0, actual_fill_l=7):
        self.water_l = water_l
        self.actual_fill_l = actual_fill_l
        self.fill_calls = []
        self.release_calls = []

    def water_remaining_l(self):
        return self.water_l

    def agent(self):
        return ExtinguishingAgent.WATER

    def release(self, volume_l, at):
        self.release_calls.append((volume_l, at))
        released = min(volume_l, self.water_l)
        self.water_l -= released
        return released

    def fill(self, volume_l, agent=ExtinguishingAgent.WATER):
        self.fill_calls.append((volume_l, agent))
        added = min(volume_l, self.actual_fill_l)
        self.water_l += added
        return added


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
        water_remaining_l=0,
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
        volume_l=None,
    )
    values.update(updates)
    return Task(**values)


def make_source(**updates):
    values = dict(
        source_id="src1",
        position=Position(x=10, y=10, z=0),
        kind=WaterSourceKind.POND,
        origin=WaterSourceOrigin.DETECTED,
        status=WaterSourceStatus.VALIDATED,
        est_volume_l=1000,
        confidence=0.9,
    )
    values.update(updates)
    return WaterSource(**values)


def make_station(**updates):
    values = dict(
        station_id="station1",
        position=Position(x=10, y=10, z=0),
        water_available_l=1000,
        fill_rate_lpm=100,
        total_bays=2,
        free_bays=1,
        agents_available=[ExtinguishingAgent.WATER],
    )
    values.update(updates)
    return RefillStation(**values)


def make_grant(
    *,
    drone_id="d1",
    epoch=4,
    kind=RefillTargetKind.WATER_SOURCE,
    target_id="src1",
    position=None,
):
    return RefillGrant(
        drone_id=drone_id,
        target_kind=kind,
        target_id=target_id,
        position=position or make_source().position,
        epoch=epoch,
    )


def make_message(body, *, epoch=4, timestamp=0, receiver_id=None):
    return SwarmMessage.create(
        sender_id="leader",
        body=body,
        now=timestamp,
        epoch=epoch,
        receiver_id=receiver_id,
    )


def make_adapter(
    *,
    drone=None,
    task=None,
    source=None,
    station=None,
    payload=None,
    autopilot=None,
    target_resolver=None,
):
    drone = drone or make_drone()
    task = task or make_task()
    source = source or make_source()
    station = station or make_station()
    bus = FakeBus()
    autopilot = autopilot or FakeAutopilot()
    payload = payload or FakePayload(drone.water_remaining_l or 0)
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
    grid_spec = GridSpec(
        origin=Position(x=-0.5, y=-0.5, z=0),
        resolution_m=1,
        rows=32,
        cols=32,
    )
    context = MissionPlanningContext(
        occupancy_grid=[[0] * 32 for _ in range(32)],
        grid_spec=grid_spec,
        base_position=Position(x=0, y=0, z=20),
        platform_spec=default_extinguisher_spec(),
        minimum_separation_m=2,
        time_step_s=1,
        altitude_band_m=(20, 30),
    )
    if target_resolver is None:
        target_resolver = lambda grant: (
            source
            if grant.target_kind is RefillTargetKind.WATER_SOURCE
            and grant.target_id == source.source_id
            else station
            if grant.target_kind is RefillTargetKind.STATION
            and grant.target_id == station.station_id
            else None
        )
    adapter = SwarmAutonomyAdapter(
        drone,
        controller,
        bus,
        planning_context=context,
        altitude_agl_provider=lambda _drone: 15,
        refill_target_resolver=target_resolver,
    )
    adapter.state.current_hotspot = make_hotspot()
    return adapter, bus, controller, autopilot, payload, task, source, station


def assign(adapter, task):
    return adapter.handle_message(
        make_message(
            TaskAssign(task=task, epoch=4),
        )
    )


def requests(bus):
    return [
        message.body
        for topic, message in bus.published
        if topic == "swarm/refill/requests"
    ]


def task_completions(bus):
    return [
        message.body
        for _, message in bus.published
        if isinstance(message.body, TaskComplete)
    ]


def test_low_water_publishes_one_request_with_current_drone_state():
    adapter, bus, _, _, _, task, _, _ = make_adapter(
        drone=make_drone(position=Position(x=3, y=4, z=20), water_remaining_l=0),
    )
    result = assign(adapter, task)

    assert result.status == "REFILL_REQUESTED"
    request, = requests(bus)
    assert isinstance(request, RefillRequest)
    assert request.drone_id == "d1"
    assert request.position == adapter.drone.position
    assert request.water_remaining_l == 0
    assert request.agent is ExtinguishingAgent.WATER
    assert request.fuel_remaining_kg == 200
    assert not request.needs_fuel

    for _ in range(4):
        assert adapter.step().status == "REFILL_REQUESTED"
    assert len(requests(bus)) == 1


def test_hybrid_low_water_request_flags_fuel_need():
    adapter, bus, _, _, _, task, _, _ = make_adapter(
        drone=make_drone(fuel_remaining_kg=0.1, endurance_remaining_s=10),
    )

    assign(adapter, task)

    request, = requests(bus)
    assert request.needs_fuel


@pytest.mark.parametrize(
    ("grant", "receiver_id"),
    [
        (make_grant(drone_id="d2"), "d1"),
        (make_grant(), "d2"),
        (make_grant(epoch=3), "d1"),
    ],
    ids=["wrong-drone-body", "wrong-drone-envelope", "stale-epoch"],
)
def test_wrong_or_stale_grants_are_rejected(grant, receiver_id):
    adapter, bus, _, autopilot, payload, task, _, _ = make_adapter()
    assign(adapter, task)

    adapter.handle_message(
        make_message(grant, receiver_id=receiver_id),
    )

    assert adapter.state.latest_refill_grant is None
    assert not autopilot.goto_targets
    assert not payload.fill_calls
    if grant.drone_id == "d1":
        assert any(topic == "ground/alerts" for topic, _ in bus.published)


def test_expired_refill_grant_is_rejected():
    adapter, bus, _, autopilot, payload, task, _, _ = make_adapter()
    assign(adapter, task)
    adapter.now = 100

    adapter.handle_message(
        make_message(
            make_grant(),
            timestamp=0,
            receiver_id="d1",
        ),
    )

    assert adapter.state.latest_refill_grant is None
    assert not autopilot.goto_targets
    assert not payload.fill_calls
    assert adapter.state.refill_request_outstanding
    assert any(topic == "ground/alerts" for topic, _ in bus.published)


def test_valid_grant_navigates_and_refill_waits_for_arrival():
    adapter, bus, controller, autopilot, payload, task, source, _ = make_adapter()
    assign(adapter, task)
    grant = make_grant()

    adapter.handle_message(
        make_message(grant, receiver_id="d1"),
    )

    assert adapter.state.latest_refill_grant == grant
    assert autopilot.goto_targets == [source.position]
    assert not payload.fill_calls
    assert adapter.step().status == "REFILL_TRANSIT"
    assert not payload.fill_calls

    autopilot.arrived = True
    result = adapter.step()

    assert result.status == "TRANSIT"
    assert result.success
    assert payload.fill_calls == [
        (default_extinguisher_spec().water_capacity_l, ExtinguishingAgent.WATER)
    ]
    assert payload.water_remaining_l() == 7
    assert adapter.drone.water_remaining_l == 7
    assert adapter.state.last_refill_water_added_l == 7
    assert adapter.state.latest_refill_grant is None
    assert not adapter.state.refill_request_outstanding
    assert not controller.ready_for_mission
    assert any(
        topic == "swarm/path_claims"
        for topic, _ in bus.published
    )

    autopilot.arrived = True
    for _ in range(len(controller.active_path or ()) + 2):
        if adapter.state.current_task is None:
            break
        adapter.step()

    assert adapter.state.current_task is None
    assert len(payload.release_calls) == 1
    assert task_completions(bus)[0].success


def test_zero_volume_refill_fails_without_resuming_mission():
    adapter, bus, controller, _, payload, task, _, _ = make_adapter()
    assign(adapter, task)
    adapter.state.refill_request_volume_l = 0

    adapter.handle_message(
        make_message(make_grant(), receiver_id="d1"),
    )

    assert not payload.fill_calls
    assert adapter.state.current_task is None
    assert controller.ready_for_mission
    assert task_completions(bus)[0].success is False


@pytest.mark.parametrize(
    ("source", "station", "grant"),
    [
        (
            make_source(status=WaterSourceStatus.CANDIDATE),
            make_station(),
            make_grant(),
        ),
        (
            make_source(),
            make_station(free_bays=0),
            make_grant(
                kind=RefillTargetKind.STATION,
                target_id="station1",
            ),
        ),
    ],
    ids=["invalid-source", "unavailable-station"],
)
def test_invalid_or_unavailable_refill_target_fails_safely(
    source,
    station,
    grant,
):
    adapter, bus, controller, autopilot, payload, task, _, _ = make_adapter(
        source=source,
        station=station,
    )
    assign(adapter, task)

    adapter.handle_message(
        make_message(grant, receiver_id="d1"),
    )

    assert not payload.fill_calls
    assert adapter.state.current_task is None
    assert controller.ready_for_mission
    assert autopilot.hold_calls >= 1
    assert task_completions(bus)[0].success is False


def test_unavailable_target_resolver_fails_grant():
    adapter, bus, _, autopilot, payload, task, _, _ = make_adapter(
        target_resolver=lambda _grant: None,
    )
    assign(adapter, task)

    adapter.handle_message(
        make_message(make_grant(), receiver_id="d1"),
    )

    assert not autopilot.goto_targets
    assert not payload.fill_calls
    assert adapter.state.current_task is None
    assert task_completions(bus)[0].success is False


def test_navigation_failure_during_refill_fails_without_filling():
    autopilot = FakeAutopilot()
    autopilot.fail_navigation = True
    adapter, bus, controller, _, payload, task, _, _ = make_adapter(
        autopilot=autopilot,
    )
    assign(adapter, task)

    adapter.handle_message(
        make_message(make_grant(), receiver_id="d1"),
    )

    assert not payload.fill_calls
    assert adapter.state.current_task is None
    assert controller.ready_for_mission
    assert task_completions(bus)[0].success is False


def test_payload_adding_zero_water_fails_without_resuming():
    payload = FakePayload(water_l=0, actual_fill_l=0)
    adapter, bus, controller, autopilot, _, task, _, _ = make_adapter(
        payload=payload,
    )
    assign(adapter, task)
    adapter.handle_message(
        make_message(make_grant(), receiver_id="d1"),
    )
    autopilot.arrived = True

    result = adapter.step()

    assert result.status == "FAILED"
    assert len(payload.fill_calls) == 1
    assert adapter.drone.water_remaining_l == 0
    assert controller.ready_for_mission
    assert task_completions(bus)[0].success is False


def test_epoch_change_during_refill_prevents_mission_resume():
    adapter, bus, controller, _, payload, task, _, _ = make_adapter()
    assign(adapter, task)
    adapter.handle_message(
        make_message(make_grant(), receiver_id="d1"),
    )
    adapter.drone.epoch = 5

    result = adapter.step()

    assert result.status == "FAILED"
    assert not payload.fill_calls
    assert not any(
        isinstance(message.body, PathClaim)
        for _, message in bus.published
    )
    assert controller.ready_for_mission


@pytest.mark.parametrize(
    "status",
    [DroneStatus.FAULT, DroneStatus.LOST],
)
def test_fault_or_lost_drone_during_refill_fails_safely(status):
    adapter, bus, controller, autopilot, payload, task, _, _ = make_adapter()
    assign(adapter, task)
    adapter.handle_message(
        make_message(make_grant(), receiver_id="d1"),
    )
    adapter.drone.status = status

    result = adapter.step()

    assert result.status == "FAILED"
    assert not payload.fill_calls
    assert adapter.state.current_task is None
    assert controller.ready_for_mission
    assert autopilot.hold_calls >= 1
    assert task_completions(bus)[0].success is False


def test_partial_refill_that_remains_insufficient_does_not_resume():
    adapter, bus, controller, autopilot, payload, task, _, _ = make_adapter(
        task=make_task(volume_l=20),
        payload=FakePayload(water_l=0, actual_fill_l=7),
    )
    assign(adapter, task)
    adapter.handle_message(
        make_message(make_grant(), receiver_id="d1"),
    )
    autopilot.arrived = True

    result = adapter.step()

    assert result.status == "FAILED"
    assert payload.water_remaining_l() == 7
    assert adapter.drone.water_remaining_l == 7
    assert controller.ready_for_mission
    assert adapter.state.current_task is None
    assert payload.release_calls == []
    assert not any(
        isinstance(message.body, PathClaim)
        for _, message in bus.published
    )
    assert task_completions(bus)[0].success is False
