from autonomy.drone_controller import DroneController
from autonomy.water_drop import WaterDropLimits
from contracts.entities import DroneState, Hotspot, Task, Zone
from contracts.enums import (
    DroneKind,
    HotspotStatus,
    Powertrain,
    TaskStatus,
    TaskType,
)
from contracts.geometry import GridSpec, Position


def make_controller():
    return DroneController(
        drop_limits=WaterDropLimits(
            min_confidence=0.8,
            max_distance_m=5,
            min_altitude_agl_m=5,
            max_altitude_agl_m=20,
            max_speed_mps=2,
        )
    )


def make_drone():
    return DroneState(
        drone_id="drone_02",
        kind=DroneKind.EXTINGUISHER,
        position=Position(x=10, y=10, z=100),
        powertrain=Powertrain.HYBRID,
        battery_pct=80,
        fuel_remaining_kg=5,
        fuel_pct=50,
        water_remaining_l=20,
        speed_mps=1,
        timestamp=0,
    )


def test_controller_plans_route():
    controller = make_controller()
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    spec = GridSpec(
        origin=Position(x=0, y=0, z=0),
        resolution_m=1,
        rows=3,
        cols=3,
    )
    route = controller.plan_route(
        grid,
        spec,
        Position(x=0.5, y=0.5),
        Position(x=2.5, y=2.5),
    )
    assert route is not None
    assert len(route) >= 2


def test_controller_generates_coverage():
    controller = make_controller()
    zone = Zone(
        zone_id="zone_01",
        boundary=[
            Position(x=0, y=0),
            Position(x=10, y=0),
            Position(x=10, y=10),
            Position(x=0, y=10),
        ],
    )
    waypoints = controller.plan_coverage(zone, spacing_m=2)
    assert waypoints
    assert all(zone.contains(point) for point in waypoints)


def test_controller_detects_path_conflict():
    controller = make_controller()
    conflicts = controller.check_path_conflicts(
        [(0, 0, 10), (1, 0, 10)],
        [[(0, 0, 10), (1, 0, 10)]],
        minimum_separation_m=3,
    )
    assert len(conflicts) == 2


def test_controller_checks_water_drop():
    controller = make_controller()
    hotspot = Hotspot(
        hotspot_id="hs_001",
        position=Position(x=12, y=10, z=0),
        peak_temp_c=500,
        confidence=0.95,
        status=HotspotStatus.CONFIRMED,
        first_seen=0,
        last_seen=5,
    )
    task = Task(
        task_id="task_001",
        type=TaskType.EXTINGUISH,
        status=TaskStatus.ASSIGNED,
        assigned_to="drone_02",
        target_hotspot_id="hs_001",
        volume_l=8,
        created_at=0,
    )
    result = controller.check_drop(
        make_drone(),
        hotspot,
        task,
        altitude_agl_m=10,
    )
    assert result.should_drop
    assert result.water_to_release_liters == 8
