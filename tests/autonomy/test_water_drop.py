import pytest

from autonomy.water_drop import WaterDropLimits, evaluate_water_drop
from contracts.entities import DroneState, Hotspot, Task
from contracts.enums import (
    DroneKind,
    HotspotStatus,
    Powertrain,
    TaskStatus,
    TaskType,
)
from contracts.geometry import Position


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
        timestamp=0,
    )
    values.update(overrides)
    return DroneState(**values)


def make_hotspot(**overrides):
    values = dict(
        hotspot_id="hs_001",
        position=Position(x=12, y=10, z=0),
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
        volume_l=8,
    )
    values.update(overrides)
    return Task(**values)


def make_limits(**overrides):
    values = dict(
        min_confidence=0.8,
        max_distance_m=5,
        min_altitude_agl_m=5,
        max_altitude_agl_m=20,
        max_speed_mps=2,
    )
    values.update(overrides)
    return WaterDropLimits(**values)


def evaluate(**overrides):
    values = dict(
        drone=make_drone(),
        hotspot=make_hotspot(),
        task=make_task(),
        altitude_agl_m=10,
        limits=make_limits(),
    )
    values.update(overrides)
    return evaluate_water_drop(**values)


def test_allows_valid_drop():
    result = evaluate()
    assert result.should_drop
    assert result.water_to_release_liters == 8


def test_uses_all_available_water_when_task_has_no_volume():
    result = evaluate(task=make_task(volume_l=None))
    assert result.should_drop
    assert result.water_to_release_liters == 20


def test_never_releases_more_than_available():
    result = evaluate(
        drone=make_drone(water_remaining_l=3),
    )
    assert result.should_drop
    assert result.water_to_release_liters == 3


def test_rejects_seeker_drone():
    # A seeker is electric and must not carry hybrid fuel fields.
    from contracts.enums import Powertrain

    seeker = DroneState(
        drone_id="drone_01",
        kind=DroneKind.SEEKER,
        position=Position(x=10, y=10, z=100),
        powertrain=Powertrain.ELECTRIC,
        battery_pct=90,
        timestamp=0,
    )
    result = evaluate(drone=seeker)
    assert not result.should_drop
    assert "extinguisher" in result.reason.lower()


def test_rejects_unconfirmed_hotspot():
    result = evaluate(
        hotspot=make_hotspot(status=HotspotStatus.DETECTED),
    )
    assert not result.should_drop


def test_rejects_low_confidence():
    result = evaluate(
        hotspot=make_hotspot(confidence=0.3),
    )
    assert not result.should_drop


def test_rejects_drone_too_far_away():
    result = evaluate(
        drone=make_drone(position=Position(x=100, y=100, z=100)),
    )
    assert not result.should_drop


def test_rejects_altitude_below_limit():
    result = evaluate(altitude_agl_m=2)
    assert not result.should_drop


def test_rejects_altitude_above_limit():
    result = evaluate(altitude_agl_m=30)
    assert not result.should_drop


def test_rejects_excessive_speed():
    result = evaluate(drone=make_drone(speed_mps=5))
    assert not result.should_drop


def test_rejects_empty_tank():
    result = evaluate(drone=make_drone(water_remaining_l=0))
    assert not result.should_drop


def test_rejects_wrong_task_type():
    result = evaluate(
        task=make_task(type=TaskType.INVESTIGATE),
    )
    assert not result.should_drop


def test_rejects_task_for_different_hotspot():
    result = evaluate(
        task=make_task(target_hotspot_id="hs_999"),
    )
    assert not result.should_drop


def test_rejects_task_assigned_to_another_drone():
    result = evaluate(
        task=make_task(assigned_to="drone_99"),
    )
    assert not result.should_drop


def test_rejects_inactive_task():
    result = evaluate(
        task=make_task(status=TaskStatus.DONE),
    )
    assert not result.should_drop


def test_rejects_invalid_agl_altitude():
    with pytest.raises(ValueError):
        evaluate(altitude_agl_m=-1)


def test_rejects_invalid_drop_limits():
    with pytest.raises(ValueError):
        make_limits(min_altitude_agl_m=30, max_altitude_agl_m=10)
