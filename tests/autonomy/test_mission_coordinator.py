import pytest

from autonomy.mission_coordinator import MissionCoordinator, MissionPlan
from autonomy.path_claim_manager import PathClaimManager
from contracts.entities import DroneState, Task
from contracts.enums import (
    DroneKind,
    Powertrain,
    SwarmRole,
    TaskStatus,
    TaskType,
)
from contracts.geometry import GridSpec, Position


class FakeBus:
    def __init__(self):
        self.published = []

    def publish(self, topic, message):
        self.published.append((topic, message))


def make_drone():
    return DroneState(
        drone_id="d1",
        kind=DroneKind.EXTINGUISHER,
        role=SwarmRole.FOLLOWER,
        position=Position(x=0.5, y=0.5, z=10),
        speed_mps=5,
        powertrain=Powertrain.ELECTRIC,
        battery_pct=90,
        endurance_remaining_s=1000,
        water_remaining_l=20,
        timestamp=0,
        epoch=3,
    )


def make_task():
    return Task(
        task_id="task-1",
        type=TaskType.EXTINGUISH,
        status=TaskStatus.ASSIGNED,
        assigned_to="d1",
        target_hotspot_id="h1",
        created_at=0,
        epoch=3,
        volume_l=5,
    )


def make_grid():
    return GridSpec(
        origin=Position(x=0, y=0, z=0),
        resolution_m=1,
        rows=3,
        cols=3,
    )


def test_coordinator_rejects_wrong_drone():
    drone = make_drone()
    task = make_task().model_copy(update={"assigned_to": "d2"})

    coordinator = MissionCoordinator(
        drone,
        PathClaimManager(FakeBus()),
    )

    result = coordinator.plan_single_task(
        task,
        Position(x=2.5, y=0.5, z=10),
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        make_grid(),
        minimum_separation_m=1,
        time_step_s=1,
        cruise_speed_mps=5,
    )

    assert not result.accepted
    assert "another drone" in result.reason


def test_coordinator_rejects_stale_epoch():
    drone = make_drone()
    task = make_task().model_copy(update={"epoch": 2})

    coordinator = MissionCoordinator(
        drone,
        PathClaimManager(FakeBus()),
    )

    result = coordinator.plan_single_task(
        task,
        Position(x=2.5, y=0.5, z=10),
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        make_grid(),
        minimum_separation_m=1,
        time_step_s=1,
        cruise_speed_mps=5,
    )

    assert not result.accepted
    assert "epoch" in result.reason


def test_coordinator_plans_route():
    drone = make_drone()
    task = make_task()

    coordinator = MissionCoordinator(
        drone,
        PathClaimManager(FakeBus()),
    )

    result = coordinator.plan_single_task(
        task,
        Position(x=2.5, y=0.5, z=10),
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        make_grid(),
        minimum_separation_m=1,
        time_step_s=1,
        cruise_speed_mps=5,
    )

    assert result.accepted
    assert result.plan is not None
    assert result.plan.task_id == "task-1"
    assert result.plan.drone_id == "d1"
    assert result.plan.waypoints[0].x == pytest.approx(0.5)
    assert result.plan.waypoints[-1].x == pytest.approx(2.5)


def test_coordinator_respects_blocked_grid():
    drone = make_drone()
    task = make_task()

    coordinator = MissionCoordinator(
        drone,
        PathClaimManager(FakeBus()),
    )

    result = coordinator.plan_single_task(
        task,
        Position(x=2.5, y=0.5, z=10),
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        make_grid(),
        minimum_separation_m=1,
        time_step_s=1,
        cruise_speed_mps=5,
    )

    assert not result.accepted


def test_coordinator_resource_gate_detects_insufficient_water():
    drone = make_drone().model_copy(
        update={"water_remaining_l": 2}
    )
    task = make_task()

    coordinator = MissionCoordinator(
        drone,
        PathClaimManager(FakeBus()),
    )

    result = coordinator.check_resources_for_task(
        task,
        outbound_distance_m=100,
        return_distance_m=100,
        cruise_speed_mps=5,
        battery_use_per_second=0.001,
    )

    assert not result.can_complete_mission
    assert result.water_margin_liters == pytest.approx(-3)


def test_coordinator_publishes_cbs_plan_as_claim():
    drone = make_drone()
    task = make_task()
    bus = FakeBus()

    coordinator = MissionCoordinator(
        drone,
        PathClaimManager(bus),
    )

    result = coordinator.plan_single_task(
        task,
        Position(x=2.5, y=0.5, z=10),
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        make_grid(),
        minimum_separation_m=1,
        time_step_s=1,
        cruise_speed_mps=5,
    )

    assert result.accepted
    assert result.plan is not None

    message = coordinator.publish_plan(
        result.plan,
        other_claims=[],
        minimum_separation_m=1,
    )

    assert message.body.drone_id == "d1"
    assert message.body.waypoints == result.plan.waypoints
    assert bus.published
    assert bus.published[0][0] == "swarm/path_claims"


def test_coordinator_rejects_route_outside_assigned_altitude_band():
    drone = make_drone()
    coordinator = MissionCoordinator(
        drone,
        PathClaimManager(FakeBus()),
    )
    plan = MissionPlan(
        task_id="task-1",
        drone_id="d1",
        waypoints=[Position(x=0.5, y=0.5, z=10)],
        duration_s=1,
        time_step_s=1,
        water_required_l=0,
    )

    with pytest.raises(ValueError, match="assigned altitude band"):
        coordinator.publish_plan(
            plan,
            other_claims=[],
            minimum_separation_m=1,
            altitude_band_m=(20, 30),
        )
