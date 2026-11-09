from autonomy.path_claim_manager import PathClaimManager
from contracts.entities import DroneState
from contracts.enums import DroneKind, Powertrain
from contracts.geometry import Position


class FakeBus:
    def __init__(self):
        self.published = []

    def publish(self, topic, msg):
        self.published.append((topic, msg))

    def subscribe(self, pattern, handler):
        raise NotImplementedError


def make_drone():
    return DroneState(
        drone_id="drone_02",
        kind=DroneKind.EXTINGUISHER,
        position=Position(x=0, y=0, z=10),
        powertrain=Powertrain.HYBRID,
        battery_pct=80,
        fuel_remaining_kg=100,
        fuel_pct=50,
        timestamp=0,
    )


def test_creates_path_claim():
    manager = PathClaimManager(FakeBus())

    msg = manager.create_claim(
        make_drone(),
        [
            Position(x=0, y=0, z=20),
            Position(x=10, y=0, z=20),
        ],
        valid_from=10,
        valid_until=20,
        altitude_band_m=(15, 25),
    )

    assert msg.sender_id == "drone_02"
    assert msg.body.drone_id == "drone_02"
    assert len(msg.body.waypoints) == 2
    assert msg.body.valid_from == 10
    assert msg.body.valid_until == 20


def test_publish_sends_to_swarm_path_claim_topic():
    bus = FakeBus()
    manager = PathClaimManager(bus)

    manager.publish_claim(
        make_drone(),
        [
            Position(x=0, y=0, z=20),
            Position(x=10, y=0, z=20),
        ],
        valid_from=10,
        valid_until=20,
    )

    assert len(bus.published) == 1

    topic, message = bus.published[0]

    assert topic == "swarm/path_claims"
    assert message.body.drone_id == "drone_02"


def test_rejects_empty_path():
    manager = PathClaimManager(FakeBus())

    try:
        manager.create_claim(
            make_drone(),
            [],
            valid_from=10,
            valid_until=20,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "at least one waypoint" in str(exc)


def test_rejects_invalid_time_window():
    manager = PathClaimManager(FakeBus())

    try:
        manager.create_claim(
            make_drone(),
            [Position(x=0, y=0, z=20)],
            valid_from=20,
            valid_until=10,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "valid_until" in str(exc)


def test_sequence_increases():
    manager = PathClaimManager(FakeBus())

    first = manager.create_claim(
        make_drone(),
        [Position(x=0, y=0, z=20)],
        valid_from=0,
        valid_until=10,
    )

    second = manager.create_claim(
        make_drone(),
        [Position(x=1, y=0, z=20)],
        valid_from=10,
        valid_until=20,
    )

    assert second.seq > first.seq


def test_publish_from_planned_path_creates_claim_from_cbs_route():
    from autonomy.cbs_planner import PathRequest, plan_paths
    from contracts.geometry import GridSpec, Position

    def cell(row, col):
        return Position(
            x=col + 0.5,
            y=row + 0.5,
            z=0,
        )

    drone = make_drone()

    grid = [[0, 0, 0] for _ in range(3)]
    grid_spec = GridSpec(
        origin=Position(x=0, y=0, z=0),
        resolution_m=1,
        rows=3,
        cols=3,
    )

    request = PathRequest(
        drone_id=drone.drone_id,
        start=cell(0, 0),
        goal=cell(0, 2),
        speed_mps=10,
        start_time=100.0,
        altitude_band_m=(20, 30),
    )

    paths = plan_paths(
        [request],
        grid,
        grid_spec,
        minimum_separation_m=1,
        time_step_s=0.1,
    )

    assert paths is not None

    bus = FakeBus()
    manager = PathClaimManager(bus)

    message = manager.publish_from_planned_path(
        drone=drone,
        waypoints=paths[drone.drone_id],
        valid_from=request.start_time,
        time_step_s=0.1,
        altitude_band_m=request.altitude_band_m,
    )

    assert message.sender_id == drone.drone_id
    assert message.body.drone_id == drone.drone_id
    assert message.body.waypoints == paths[drone.drone_id]
    assert message.body.valid_from == 100.0
    assert message.body.valid_until > message.body.valid_from
    assert message.body.altitude_band_m == (20, 30)

    assert len(bus.published) == 1

    topic, published_message = bus.published[0]
    assert topic == "swarm/path_claims"
    assert published_message == message