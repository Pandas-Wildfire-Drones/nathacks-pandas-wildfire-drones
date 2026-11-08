from autonomy.coverage_search import generate_coverage_waypoints
from autonomy.deconfliction import detect_conflicts
from autonomy.health_manager import evaluate_drone_health
from autonomy.path_claim_manager import PathClaimManager
from autonomy.path_planning import find_path
from autonomy.refill_manager import execute_water_refill
from autonomy.water_drop import WaterDropLimits

from contracts.entities import (
    DroneState,
    Hotspot,
    Task,
    WaterSource,
    Zone,
)
from contracts.enums import (
    DroneKind,
    DroneStatus,
    HotspotStatus,
    Powertrain,
    TaskStatus,
    TaskType,
    WaterSourceKind,
    WaterSourceOrigin,
    WaterSourceStatus,
)
from contracts.geometry import GridSpec, Position


def make_drone():
    return DroneState(
        drone_id="drone_02",
        kind=DroneKind.EXTINGUISHER,
        position=Position(x=0, y=0, z=20),
        powertrain=Powertrain.HYBRID,
        battery_pct=80,
        fuel_remaining_kg=100,
        fuel_pct=50,
        water_remaining_l=20,
        sensors_ok={"thermal": True, "lidar": True},
        link_quality=1.0,
        status=DroneStatus.IDLE,
        timestamp=0,
    )


def test_phase2_path_and_coverage_stack():
    zone = Zone(
        zone_id="zone_01",
        boundary=[
            Position(x=0, y=0),
            Position(x=10, y=0),
            Position(x=10, y=10),
            Position(x=0, y=10),
        ],
    )

    waypoints = generate_coverage_waypoints(
        zone,
        spacing_m=2,
        z_m=20,
    )

    assert waypoints

    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    spec = GridSpec(
        origin=Position(x=0, y=0, z=0),
        resolution_m=5,
        rows=3,
        cols=3,
    )

    route = find_path(
        grid,
        spec,
        Position(x=2.5, y=2.5),
        Position(x=12.5, y=12.5),
    )

    assert route


def test_phase2_deconfliction_detects_shared_route():
    route = [
        Position(x=0, y=0, z=20),
        Position(x=5, y=0, z=20),
        Position(x=10, y=0, z=20),
    ]

    conflicts = detect_conflicts(
        route,
        [route],
        minimum_separation_m=3,
    )

    assert len(conflicts) == 3


def test_phase2_health_blocks_unhealthy_drone():
    drone = make_drone().model_copy(
        update={"sensors_ok": {"thermal": False}}
    )

    decision = evaluate_drone_health(drone)

    assert not decision.safe
    assert decision.action == "HOLD"


def test_phase2_path_claim_can_be_created():
    class Bus:
        def __init__(self):
            self.messages = []

        def publish(self, topic, msg):
            self.messages.append((topic, msg))

        def subscribe(self, pattern, handler):
            raise NotImplementedError

    bus = Bus()
    manager = PathClaimManager(bus)

    msg = manager.publish_claim(
        make_drone(),
        [
            Position(x=0, y=0, z=20),
            Position(x=10, y=0, z=20),
        ],
        valid_from=0,
        valid_until=20,
        altitude_band_m=(15, 25),
    )

    assert msg.body.drone_id == "drone_02"
    assert bus.messages


def test_phase2_validated_water_source_can_support_refill():
    class Autopilot:
        def __init__(self):
            self.arrived = True
            self.hold_count = 0

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
            pass

        def hold(self):
            self.hold_count += 1

        def return_to_launch(self):
            pass

        def at_target(self, tolerance_m=2.0):
            return self.arrived

    class Payload:
        def water_remaining_l(self):
            return 5

        def agent(self):
            from contracts.enums import ExtinguishingAgent
            return ExtinguishingAgent.WATER

        def release(self, volume_l, at):
            return volume_l

        def fill(self, volume_l, agent):
            return volume_l

    source = WaterSource(
        source_id="src_01",
        position=Position(x=50, y=50, z=0),
        kind=WaterSourceKind.POND,
        origin=WaterSourceOrigin.DETECTED,
        status=WaterSourceStatus.VALIDATED,
        est_volume_l=10000,
        confidence=0.95,
    )

    result = execute_water_refill(
        make_drone(),
        Autopilot(),
        Payload(),
        source.position,
        20,
        source=source,
    )

    assert result.success
    assert result.water_added_l == 20
