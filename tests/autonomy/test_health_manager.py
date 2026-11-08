from autonomy.health_manager import evaluate_drone_health
from contracts.entities import DroneState
from contracts.enums import (
    DroneKind,
    DroneStatus,
    Powertrain,
)
from contracts.geometry import Position


def make_drone(**updates):
    values = dict(
        drone_id="drone_02",
        kind=DroneKind.EXTINGUISHER,
        position=Position(x=0, y=0, z=20),
        powertrain=Powertrain.HYBRID,
        battery_pct=80,
        fuel_remaining_kg=100,
        fuel_pct=50,
        water_remaining_l=50,
        sensors_ok={"thermal": True, "lidar": True},
        link_quality=1.0,
        status=DroneStatus.IDLE,
        timestamp=0,
    )

    values.update(updates)
    return DroneState(**values)


def test_healthy_drone_can_continue():
    result = evaluate_drone_health(make_drone())

    assert result.safe
    assert result.action == "CONTINUE"


def test_sensor_failure_holds_drone():
    result = evaluate_drone_health(
        make_drone(sensors_ok={"thermal": False})
    )

    assert not result.safe
    assert result.action == "HOLD"


def test_bad_link_returns_drone():
    result = evaluate_drone_health(
        make_drone(link_quality=0.1)
    )

    assert not result.safe
    assert result.action == "RETURN"


def test_fault_drone_returns_or_holds():
    result = evaluate_drone_health(
        make_drone(status=DroneStatus.FAULT)
    )

    assert not result.safe
    assert result.action == "RETURN_OR_HOLD"


def test_lost_drone_is_not_safe():
    result = evaluate_drone_health(
        make_drone(status=DroneStatus.LOST)
    )

    assert not result.safe
    assert result.action == "LOST"


def test_invalid_link_threshold_rejected():
    try:
        evaluate_drone_health(
            make_drone(),
            minimum_link_quality=2,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass
