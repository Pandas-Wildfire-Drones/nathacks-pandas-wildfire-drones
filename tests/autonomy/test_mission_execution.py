import pytest

from autonomy.drone_controller import DroneController
from autonomy.water_drop import WaterDropLimits
from contracts.entities import DroneState, Hotspot, Task
from contracts.enums import (
    DroneKind,
    HotspotStatus,
    Powertrain,
    TaskStatus,
    TaskType,
)
from contracts.geometry import Position


class FakeAutopilot:
    def __init__(self, arrived=True):
        self.arrived = arrived
        self.current_position = Position(x=12, y=10, z=10)
        self.current_speed = 1.0
        self.goto_calls = []
        self.hold_calls = 0

    def position(self):
        return self.current_position

    def heading_deg(self):
        return 0.0

    def speed_mps(self):
        return self.current_speed

    def battery_pct(self):
        return 80.0

    def endurance_remaining_s(self):
        return 500.0

    def goto(self, target, speed_mps=None):
        self.goto_calls.append((target, speed_mps))

    def hold(self):
        self.hold_calls += 1

    def return_to_launch(self):
        pass

    def at_target(self, tolerance_m=2.0):
        return self.arrived


class FakePayload:
    def __init__(self, water=20.0, release_amount=None):
        self.water = water
        self.release_amount = release_amount
        self.release_calls = []

    def water_remaining_l(self):
        return self.water

    def agent(self):
        from contracts.enums import ExtinguishingAgent
        return ExtinguishingAgent.WATER

    def release(self, volume_l, at):
        self.release_calls.append((volume_l, at))

        amount = (
            volume_l
            if self.release_amount is None
            else min(volume_l, self.release_amount)
        )

        self.water -= amount
        return amount

    def fill(self, volume_l, agent=None):
        amount = min(volume_l, 20.0 - self.water)
        self.water += amount
        return amount


def make_drone(**overrides):
    values = dict(
        drone_id="drone_02",
        kind=DroneKind.EXTINGUISHER,
        position=Position(x=10, y=10, z=100),
        powertrain=Powertrain.HYBRID,
        battery_pct=80,
        fuel_remaining_kg=5,
        fuel_pct=50,
        water_remaining_l=20,
        speed_mps=1,
        epoch=2,
        timestamp=0,
    )
    values.update(overrides)
    return DroneState(**values)


def make_hotspot(**overrides):
    values = dict(
        hotspot_id="hs_001",
        position=Position(x=12, y=10, z=10),
        peak_temp_c=500,
        confidence=0.95,
        status=HotspotStatus.CONFIRMED,
        first_seen=0,
        last_seen=5,
    )
    values.update(overrides)
    return Hotspot(**values)


def make_task(**overrides):
    values = dict(
        task_id="task_001",
        type=TaskType.EXTINGUISH,
        status=TaskStatus.ASSIGNED,
        assigned_to="drone_02",
        target_hotspot_id="hs_001",
        created_at=0,
        epoch=2,
        volume_l=8,
    )
    values.update(overrides)
    return Task(**values)


def make_controller(arrived=True, water=20.0, release_amount=None):
    autopilot = FakeAutopilot(arrived=arrived)
    payload = FakePayload(
        water=water,
        release_amount=release_amount,
    )

    controller = DroneController(
        drop_limits=WaterDropLimits(
            min_confidence=0.8,
            max_distance_m=5,
            min_altitude_agl_m=5,
            max_altitude_agl_m=20,
            max_speed_mps=2,
        ),
        autopilot=autopilot,
        payload=payload,
    )

    return controller, autopilot, payload


def test_begin_extinguish_requests_navigation():
    controller, autopilot, _ = make_controller()

    result = controller.begin_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
    )

    assert result.status == "TRANSIT"
    assert len(autopilot.goto_calls) == 1


def test_begin_extinguish_passes_cruise_speed():
    controller, autopilot, _ = make_controller()

    controller.begin_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
        cruise_speed_mps=5,
    )

    assert autopilot.goto_calls[0][1] == 5


def test_rejects_wrong_drone_assignment():
    controller, autopilot, _ = make_controller()

    with pytest.raises(ValueError, match="assigned"):
        controller.begin_extinguish(
            make_drone(),
            make_hotspot(),
            make_task(assigned_to="drone_99"),
        )

    assert not autopilot.goto_calls


def test_rejects_stale_task_epoch_before_navigation():
    controller, autopilot, _ = make_controller()

    with pytest.raises(ValueError, match="epoch"):
        controller.begin_extinguish(
            make_drone(epoch=2),
            make_hotspot(),
            make_task(epoch=1),
        )

    assert not autopilot.goto_calls


def test_waits_until_autopilot_reports_arrival():
    controller, autopilot, payload = make_controller(arrived=False)

    controller.begin_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
    )

    result = controller.step_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
        altitude_agl_m=10,
    )

    assert result.status == "TRANSIT"
    assert not payload.release_calls
    assert autopilot.hold_calls == 0


def test_releases_water_after_arrival():
    controller, _, payload = make_controller()

    controller.begin_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
    )

    result = controller.step_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
        altitude_agl_m=10,
    )

    assert result.status == "DONE"
    assert result.success
    assert result.water_used_l == 8
    assert payload.water == 12


def test_does_not_release_twice_after_completion():
    controller, _, payload = make_controller()

    controller.begin_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
    )

    first = controller.step_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
        altitude_agl_m=10,
    )

    second = controller.step_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
        altitude_agl_m=10,
    )

    assert first == second
    assert len(payload.release_calls) == 1


def test_rejects_unsafe_altitude_without_releasing():
    controller, autopilot, payload = make_controller()

    controller.begin_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
    )

    result = controller.step_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
        altitude_agl_m=30,
    )

    assert result.status == "REJECTED"
    assert not result.success
    assert not payload.release_calls
    assert autopilot.hold_calls == 1


def test_reports_failed_release_when_payload_releases_nothing():
    controller, _, payload = make_controller(release_amount=0)

    controller.begin_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
    )

    result = controller.step_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
        altitude_agl_m=10,
    )

    assert result.status == "FAILED"
    assert not result.success
    assert result.water_used_l == 0


def test_rejects_second_mission_while_one_is_active():
    controller, _, _ = make_controller(arrived=False)

    controller.begin_extinguish(
        make_drone(),
        make_hotspot(),
        make_task(),
    )

    another_task = make_task(task_id="task_002")

    with pytest.raises(RuntimeError, match="already active"):
        controller.begin_extinguish(
            make_drone(),
            make_hotspot(),
            another_task,
        )
