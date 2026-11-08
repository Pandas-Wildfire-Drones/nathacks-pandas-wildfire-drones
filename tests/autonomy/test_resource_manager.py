import pytest

from autonomy.resource_manager import assess_mission_resources
from contracts.entities import DroneState
from contracts.enums import DroneKind, Powertrain
from contracts.geometry import Position


def make_electric_drone(
    battery_pct=85,
    water_remaining_l=14,
    endurance_remaining_s=1000,
):
    return DroneState(
        drone_id="drone_01",
        kind=DroneKind.SEEKER,
        position=Position(x=0, y=0, z=10),
        powertrain=Powertrain.ELECTRIC,
        battery_pct=battery_pct,
        endurance_remaining_s=endurance_remaining_s,
        water_remaining_l=water_remaining_l,
        timestamp=0,
    )


def make_hybrid_drone(
    battery_pct=80,
    fuel_remaining_kg=5,
    water_remaining_l=100,
    endurance_remaining_s=1000,
):
    return DroneState(
        drone_id="drone_02",
        kind=DroneKind.EXTINGUISHER,
        position=Position(x=0, y=0, z=10),
        powertrain=Powertrain.HYBRID,
        battery_pct=battery_pct,
        endurance_remaining_s=endurance_remaining_s,
        fuel_remaining_kg=fuel_remaining_kg,
        fuel_pct=50,
        water_remaining_l=water_remaining_l,
        timestamp=0,
    )


def test_electric_drone_has_sufficient_resources():
    drone = make_electric_drone()

    result = assess_mission_resources(
        drone,
        outbound_distance_m=500,
        return_distance_m=500,
        cruise_speed_mps=10,
        battery_use_per_second=0.001,
        hover_seconds=10,
        hover_battery_use_per_second=0.002,
        required_drop_liters=5,
        reserve_fraction=0.20,
    )

    assert result.can_complete_mission
    assert not result.should_return
    assert result.battery_required_pct == pytest.approx(12)
    assert result.battery_margin_pct == pytest.approx(53)
    assert result.water_margin_liters == pytest.approx(9)


def test_electric_drone_has_insufficient_battery():
    drone = make_electric_drone(battery_pct=20)

    result = assess_mission_resources(
        drone,
        outbound_distance_m=500,
        return_distance_m=500,
        cruise_speed_mps=10,
        battery_use_per_second=0.001,
        hover_seconds=10,
        hover_battery_use_per_second=0.002,
        required_drop_liters=5,
        reserve_fraction=0.20,
    )

    assert not result.can_complete_mission
    assert result.should_return
    assert any("battery" in reason.lower() for reason in result.reasons)


def test_detects_insufficient_water():
    drone = make_electric_drone(water_remaining_l=1)

    result = assess_mission_resources(
        drone,
        outbound_distance_m=100,
        return_distance_m=100,
        cruise_speed_mps=10,
        battery_use_per_second=0.001,
        required_drop_liters=5,
    )

    assert not result.can_complete_mission
    assert result.water_margin_liters == pytest.approx(-4)
    assert any("water" in reason.lower() for reason in result.reasons)


def test_hybrid_drone_has_sufficient_fuel_and_battery_reserve():
    drone = make_hybrid_drone()

    result = assess_mission_resources(
        drone,
        outbound_distance_m=500,
        return_distance_m=500,
        cruise_speed_mps=10,
        hover_seconds=10,
        fuel_use_kg_per_second=0.01,
        fuel_reserve_kg=0.5,
        required_drop_liters=50,
        reserve_fraction=0.20,
    )

    assert result.can_complete_mission
    assert result.fuel_required_kg == pytest.approx(1.1)
    assert result.fuel_margin_kg == pytest.approx(3.4)
    assert result.battery_margin_pct == pytest.approx(60)


def test_hybrid_drone_returns_when_fuel_is_insufficient():
    drone = make_hybrid_drone(fuel_remaining_kg=0.5)

    result = assess_mission_resources(
        drone,
        outbound_distance_m=500,
        return_distance_m=500,
        cruise_speed_mps=10,
        fuel_use_kg_per_second=0.01,
        fuel_reserve_kg=0.5,
    )

    assert not result.can_complete_mission
    assert result.should_return
    assert any("fuel" in reason.lower() for reason in result.reasons)


def test_hybrid_drone_preserves_emergency_battery_reserve():
    drone = make_hybrid_drone(battery_pct=10)

    result = assess_mission_resources(
        drone,
        outbound_distance_m=100,
        return_distance_m=100,
        cruise_speed_mps=10,
        fuel_use_kg_per_second=0.001,
        reserve_fraction=0.20,
    )

    assert not result.can_complete_mission
    assert result.battery_margin_pct == pytest.approx(-10)
    assert any("battery" in reason.lower() for reason in result.reasons)


def test_detects_insufficient_remaining_endurance():
    drone = make_electric_drone(endurance_remaining_s=50)

    result = assess_mission_resources(
        drone,
        outbound_distance_m=500,
        return_distance_m=500,
        cruise_speed_mps=10,
        battery_use_per_second=0.001,
    )

    assert not result.can_complete_mission
    assert any("endurance" in reason.lower() for reason in result.reasons)


def test_seeker_does_not_need_water_for_search():
    drone = make_electric_drone(water_remaining_l=None)

    result = assess_mission_resources(
        drone,
        outbound_distance_m=100,
        return_distance_m=100,
        cruise_speed_mps=10,
        battery_use_per_second=0.001,
        required_drop_liters=0,
    )

    assert result.can_complete_mission
    assert result.water_available_liters == 0


def test_rejects_zero_cruise_speed():
    drone = make_electric_drone()

    with pytest.raises(ValueError):
        assess_mission_resources(
            drone,
            outbound_distance_m=100,
            return_distance_m=100,
            cruise_speed_mps=0,
            battery_use_per_second=0.001,
        )


def test_rejects_negative_distance():
    drone = make_electric_drone()

    with pytest.raises(ValueError):
        assess_mission_resources(
            drone,
            outbound_distance_m=-100,
            return_distance_m=100,
            cruise_speed_mps=10,
            battery_use_per_second=0.001,
        )


def test_rejects_invalid_battery_reserve():
    drone = make_electric_drone()

    with pytest.raises(ValueError):
        assess_mission_resources(
            drone,
            outbound_distance_m=100,
            return_distance_m=100,
            cruise_speed_mps=10,
            battery_use_per_second=0.001,
            reserve_fraction=1.5,
        )


def test_electric_drone_requires_battery_consumption_estimate():
    drone = make_electric_drone()

    with pytest.raises(ValueError):
        assess_mission_resources(
            drone,
            outbound_distance_m=100,
            return_distance_m=100,
            cruise_speed_mps=10,
        )


def test_hybrid_drone_requires_fuel_consumption_estimate():
    drone = make_hybrid_drone()

    with pytest.raises(ValueError):
        assess_mission_resources(
            drone,
            outbound_distance_m=100,
            return_distance_m=100,
            cruise_speed_mps=10,
        )
