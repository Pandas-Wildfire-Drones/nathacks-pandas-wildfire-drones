from types import SimpleNamespace

import pytest

from autonomy.swarm_adapter import SwarmAutonomyAdapter
from contracts.entities import DroneState, Hotspot, Position, Task
from contracts.enums import (
    AlertKind,
    DroneKind,
    DroneStatus,
    ExtinguishingAgent,
    Powertrain,
    Severity,
    SwarmRole,
    TaskStatus,
    TaskType,
)
from contracts.messages import SwarmMessage, TaskAssign


class FakeBus:
    def __init__(self):
        self.subscriptions = []
        self.published = []

    def subscribe(self, topic, handler):
        self.subscriptions.append((topic, handler))

    def publish(self, topic, message):
        self.published.append((topic, message))


class FakeController:
    def __init__(self):
        self.autopilot = SimpleNamespace(
            hold_calls=0,
        )
        self.autopilot.hold = self._hold
        self.begin_calls = []
        self.step_calls = []

    def _hold(self):
        self.autopilot.hold_calls += 1

    def begin_extinguish(self, drone, hotspot, task, cruise_speed_mps=None):
        self.begin_calls.append((drone, hotspot, task, cruise_speed_mps))

        from autonomy.drone_controller import MissionExecutionResult

        return MissionExecutionResult(
            status="TRANSIT",
            success=False,
            water_used_l=0.0,
            reason="Navigation requested",
        )

    def step_extinguish(self, drone, hotspot, task, *, altitude_agl_m):
        self.step_calls.append((drone, hotspot, task, altitude_agl_m))

        from autonomy.drone_controller import MissionExecutionResult

        return MissionExecutionResult(
            status="DONE",
            success=True,
            water_used_l=8.0,
            reason="Water released successfully",
        )


def make_drone():
    return DroneState(
        drone_id="drone_02",
        kind=DroneKind.EXTINGUISHER,
        role=SwarmRole.FOLLOWER,
        position=Position(x=0, y=0, z=20),
        heading_deg=0,
        speed_mps=0,
        powertrain=Powertrain.HYBRID,
        battery_pct=90,
        endurance_remaining_s=3600,
        fuel_remaining_kg=200,
        fuel_pct=95.24,
        water_remaining_l=100,
        agent=ExtinguishingAgent.WATER,
        status=DroneStatus.IDLE,
        epoch=2,
        sensors_ok={"thermal": True, "lidar": True},
        link_quality=1.0,
        timestamp=0,
    )


def make_hotspot():
    return Hotspot(
        hotspot_id="hs_001",
        position=Position(x=12, y=10, z=10),
        peak_temp_c=500,
        confidence=0.95,
        status="CONFIRMED",
        priority_score=0.8,
        observations=1,
        first_seen=0,
        last_seen=1,
    )


def make_task():
    return Task(
        task_id="task_001",
        type=TaskType.EXTINGUISH,
        priority=0.8,
        status=TaskStatus.ASSIGNED,
        assigned_to="drone_02",
        target_hotspot_id="hs_001",
        created_at=0,
        epoch=2,
        volume_l=8,
    )


def make_message(body, epoch=2):
    return SwarmMessage.create(
        sender_id="leader_01",
        body=body,
        now=0,
        epoch=epoch,
    )


def test_adapter_subscribes_to_swarm_messages():
    bus = FakeBus()
    controller = FakeController()

    SwarmAutonomyAdapter(make_drone(), controller, bus)

    assert len(bus.subscriptions) >= 5


def test_task_assign_for_this_drone_starts_mission():
    bus = FakeBus()
    controller = FakeController()

    adapter = SwarmAutonomyAdapter(make_drone(), controller, bus)
    adapter.state.current_hotspot = make_hotspot()

    assignment = TaskAssign(
        task=make_task(),
        epoch=2,
    )

    result = adapter.handle_message(
        make_message(assignment),
    )

    assert result.status == "TRANSIT"
    assert len(controller.begin_calls) == 1


def test_task_assign_for_another_drone_is_ignored():
    bus = FakeBus()
    controller = FakeController()

    adapter = SwarmAutonomyAdapter(make_drone(), controller, bus)

    task = make_task().model_copy(
        update={"assigned_to": "drone_99"}
    )

    adapter.handle_message(
        make_message(
            TaskAssign(
                task=task,
                epoch=2,
            )
        )
    )

    assert controller.begin_calls == []


def test_stale_task_assignment_is_not_executed():
    bus = FakeBus()
    controller = FakeController()

    adapter = SwarmAutonomyAdapter(make_drone(), controller, bus)
    adapter.state.current_hotspot = make_hotspot()

    adapter.handle_message(
        make_message(
            TaskAssign(
                task=make_task(),
                epoch=1,
            ),
            epoch=1,
        )
    )

    assert controller.begin_calls == []
    assert len(bus.published) == 1


def test_unknown_hotspot_blocks_task_execution():
    bus = FakeBus()
    controller = FakeController()

    adapter = SwarmAutonomyAdapter(make_drone(), controller, bus)

    adapter.handle_message(
        make_message(
            TaskAssign(
                task=make_task(),
                epoch=2,
            )
        )
    )

    assert controller.begin_calls == []
    assert len(bus.published) == 1


def test_step_publishes_task_completion():
    bus = FakeBus()
    controller = FakeController()

    adapter = SwarmAutonomyAdapter(make_drone(), controller, bus)
    adapter.state.current_task = make_task()
    adapter.state.current_hotspot = make_hotspot()

    result = adapter.step()

    assert result.status == "DONE"
    assert len(controller.step_calls) == 1
    assert len(bus.published) == 1


def test_unhealthy_drone_holds_instead_of_executing():
    bus = FakeBus()
    controller = FakeController()

    drone = make_drone().model_copy(
        update={"sensors_ok": {"thermal": False}}
    )

    adapter = SwarmAutonomyAdapter(drone, controller, bus)
    adapter.state.current_task = make_task()
    adapter.state.current_hotspot = make_hotspot()

    result = adapter.step()

    assert result is None
    assert controller.autopilot.hold_calls == 1
    assert controller.step_calls == []
    assert len(bus.published) == 1
