from autonomy.deconfliction import Conflict
from autonomy.drone_controller import DroneController, MissionExecutionResult
from autonomy.water_drop import WaterDropLimits
from autonomy.swarm_adapter import SwarmAutonomyAdapter
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
from contracts.geometry import Position
from contracts.messages import PathClaim, SwarmMessage


class FakeBus:
    def __init__(self):
        self.messages = []
        self.subscriptions = []

    def publish(self, topic, msg):
        self.messages.append((topic, msg))

    def subscribe(self, pattern, handler):
        self.subscriptions.append((pattern, handler))


class FakeAutopilot:
    def __init__(self):
        self.hold_count = 0
        self.return_count = 0
        self.goto_calls = []
        self.arrived = False

    def position(self):
        return Position(x=0, y=0, z=20)

    def heading_deg(self):
        return 0

    def speed_mps(self):
        return 0

    def battery_pct(self):
        return 80

    def endurance_remaining_s(self):
        return 1000

    def goto(self, target, speed_mps=None):
        self.goto_calls.append((target, speed_mps))

    def hold(self):
        self.hold_count += 1

    def return_to_launch(self):
        self.return_count += 1

    def at_target(self, tolerance_m=2.0):
        return self.arrived


class FakePayload:
    def __init__(self):
        self.water_remaining = 20
        self.release_calls = []

    def water_remaining_l(self):
        return self.water_remaining

    def agent(self):
        return ExtinguishingAgent.WATER

    def release(self, volume_l, at):
        self.release_calls.append((volume_l, at))
        self.water_remaining -= volume_l
        return volume_l

    def fill(self, volume_l, agent):
        self.water_remaining += volume_l
        return volume_l


def make_drone(**updates):
    values = dict(
        drone_id="drone_02",
        kind=DroneKind.EXTINGUISHER,
        position=Position(x=0, y=0, z=20),
        heading_deg=0,
        speed_mps=0,
        powertrain=Powertrain.HYBRID,
        battery_pct=80,
        fuel_remaining_kg=100,
        fuel_pct=50,
        water_remaining_l=20,
        sensors_ok={"thermal": True, "lidar": True},
        link_quality=1.0,
        status=DroneStatus.IDLE,
        timestamp=100,
        epoch=1,
    )
    values.update(updates)
    return DroneState(**values)


def make_hotspot(**updates):
    values = dict(
        hotspot_id="hotspot_01",
        position=Position(x=10, y=0, z=20),
        peak_temp_c=500,
        est_area_m2=100,
        confidence=0.95,
        status=HotspotStatus.CONFIRMED,
        priority_score=1.0,
        observations=1,
        growth_rate_m2_per_min=1.0,
        assigned_to="drone_02",
        reported_by="drone_02",
        first_seen=100,
        last_seen=100,
    )
    values.update(updates)
    return Hotspot(**values)


def make_task(**updates):
    values = dict(
        task_id="task_01",
        type=TaskType.EXTINGUISH,
        priority=1,
        status=TaskStatus.ASSIGNED,
        assigned_to="drone_02",
        target_hotspot_id="hotspot_01",
        created_at=100,
        deadline=200,
        epoch=1,
        volume_l=5,
    )
    values.update(updates)
    return Task(**values)


def make_message(body):
    return SwarmMessage.create(
        sender_id="leader_01",
        body=body,
        now=100,
        epoch=1,
    )


def test_faulted_drone_is_blocked_before_mission_step():
    autopilot = FakeAutopilot()
    payload = FakePayload()
    controller = DroneController(
        WaterDropLimits(
            min_confidence=0.70,
            max_distance_m=100.0,
            min_altitude_agl_m=10.0,
            max_altitude_agl_m=100.0,
            max_speed_mps=20.0,
        ),
        autopilot,
        payload,
    )

    drone = make_drone(status=DroneStatus.FAULT)
    adapter = SwarmAutonomyAdapter(drone, controller, FakeBus())

    adapter.state.current_task = make_task()
    adapter.state.current_hotspot = make_hotspot()

    result = adapter.step()

    assert result is None
    assert autopilot.hold_count == 1
    assert payload.release_calls == []


def test_lost_drone_is_blocked_before_mission_step():
    autopilot = FakeAutopilot()
    payload = FakePayload()
    controller = DroneController(
        WaterDropLimits(
            min_confidence=0.70,
            max_distance_m=100.0,
            min_altitude_agl_m=10.0,
            max_altitude_agl_m=100.0,
            max_speed_mps=20.0,
        ),
        autopilot,
        payload,
    )

    drone = make_drone(status=DroneStatus.LOST)
    adapter = SwarmAutonomyAdapter(drone, controller, FakeBus())

    adapter.state.current_task = make_task()
    adapter.state.current_hotspot = make_hotspot()

    result = adapter.step()

    assert result is None
    assert autopilot.hold_count == 1
    assert payload.release_calls == []


def test_failed_sensor_blocks_mission_step():
    autopilot = FakeAutopilot()
    payload = FakePayload()
    controller = DroneController(
        WaterDropLimits(
            min_confidence=0.70,
            max_distance_m=100.0,
            min_altitude_agl_m=10.0,
            max_altitude_agl_m=100.0,
            max_speed_mps=20.0,
        ),
        autopilot,
        payload,
    )

    drone = make_drone(
        sensors_ok={"thermal": False, "lidar": True}
    )
    adapter = SwarmAutonomyAdapter(drone, controller, FakeBus())

    adapter.state.current_task = make_task()
    adapter.state.current_hotspot = make_hotspot()

    result = adapter.step()

    assert result is None
    assert autopilot.hold_count == 1
    assert payload.release_calls == []


def test_degraded_link_blocks_mission_step():
    autopilot = FakeAutopilot()
    payload = FakePayload()
    controller = DroneController(
        WaterDropLimits(
            min_confidence=0.70,
            max_distance_m=100.0,
            min_altitude_agl_m=10.0,
            max_altitude_agl_m=100.0,
            max_speed_mps=20.0,
        ),
        autopilot,
        payload,
    )

    drone = make_drone(link_quality=0.1)
    adapter = SwarmAutonomyAdapter(drone, controller, FakeBus())

    adapter.state.current_task = make_task()
    adapter.state.current_hotspot = make_hotspot()

    result = adapter.step()

    assert result is None
    assert autopilot.return_count == 1
    assert autopilot.hold_count == 0
    assert payload.release_calls == []


def test_healthy_drone_can_continue_to_controller():
    autopilot = FakeAutopilot()
    payload = FakePayload()
    controller = DroneController(
        WaterDropLimits(
            min_confidence=0.70,
            max_distance_m=100.0,
            min_altitude_agl_m=10.0,
            max_altitude_agl_m=100.0,
            max_speed_mps=20.0,
        ),
        autopilot,
        payload,
    )

    drone = make_drone()
    adapter = SwarmAutonomyAdapter(
        drone,
        controller,
        FakeBus(),
        altitude_agl_provider=lambda _drone: 20.0,
    )

    task = make_task()
    hotspot = make_hotspot()

    adapter.state.current_task = task
    adapter.state.current_hotspot = hotspot

    begin_result = controller.begin_extinguish(
        drone,
        hotspot,
        task,
    )

    assert begin_result.status == "TRANSIT"
    assert begin_result.success
    assert autopilot.goto_calls

    result = adapter.step()

    assert result.status == "TRANSIT"
    assert result.success
    assert autopilot.hold_count == 0


def test_other_drone_path_claim_is_retained_for_deconfliction():
    bus = FakeBus()
    adapter = SwarmAutonomyAdapter(
        make_drone(),
        DroneController(FakeAutopilot(), FakePayload()),
        bus,
    )

    claim = PathClaim(
        drone_id="drone_03",
        waypoints=[
            Position(x=0, y=0, z=20),
            Position(x=10, y=0, z=20),
        ],
        valid_from=100,
        valid_until=120,
        altitude_band_m=(15, 25),
    )

    adapter.handle_path_claim(
        make_message(claim),
        claim,
    )

    assert len(adapter.state.remote_path_claims) == 1
    assert adapter.state.remote_path_claims[0].drone_id == "drone_03"


def test_own_path_claim_is_not_added_as_remote_claim():
    bus = FakeBus()
    adapter = SwarmAutonomyAdapter(
        make_drone(),
        DroneController(FakeAutopilot(), FakePayload()),
        bus,
    )

    claim = PathClaim(
        drone_id="drone_02",
        waypoints=[
            Position(x=0, y=0, z=20),
            Position(x=10, y=0, z=20),
        ],
        valid_from=100,
        valid_until=120,
        altitude_band_m=(15, 25),
    )

    adapter.handle_path_claim(
        make_message(claim),
        claim,
    )

    assert adapter.state.latest_path_claim == claim
    assert adapter.state.remote_path_claims == []


def test_conflicting_remote_path_is_detected():
    bus = FakeBus()
    adapter = SwarmAutonomyAdapter(
        make_drone(),
        DroneController(FakeAutopilot(), FakePayload()),
        bus,
    )

    claim = PathClaim(
        drone_id="drone_03",
        waypoints=[
            Position(x=0, y=1, z=20),
            Position(x=10, y=1, z=20),
        ],
        valid_from=100,
        valid_until=101,
        altitude_band_m=(15, 25),
    )

    adapter.handle_path_claim(
        make_message(claim),
        claim,
    )

    proposed = [
        Position(x=0, y=0, z=20),
        Position(x=10, y=0, z=20),
    ]

    conflicts = adapter.check_path_conflicts(
        proposed,
        minimum_separation_m=2,
    )

    assert len(conflicts) == 2
    assert all(isinstance(conflict, Conflict) for conflict in conflicts)


def test_safe_remote_path_has_no_conflict():
    bus = FakeBus()
    adapter = SwarmAutonomyAdapter(
        make_drone(),
        DroneController(FakeAutopilot(), FakePayload()),
        bus,
    )

    claim = PathClaim(
        drone_id="drone_03",
        waypoints=[
            Position(x=0, y=10, z=20),
            Position(x=10, y=10, z=20),
        ],
        valid_from=100,
        valid_until=120,
        altitude_band_m=(15, 25),
    )

    adapter.handle_path_claim(
        make_message(claim),
        claim,
    )

    proposed = [
        Position(x=0, y=0, z=20),
        Position(x=10, y=0, z=20),
    ]

    conflicts = adapter.check_path_conflicts(
        proposed,
        minimum_separation_m=2,
    )

    assert conflicts == []
