
from autonomy.resource_manager import assess_mission_resources


def sample_inputs():
    return dict(
        battery_level=0.85,
        water_level=0.70,
        tank_capacity_liters=20,
        outbound_distance_m=500,
        return_distance_m=500,
        cruise_speed_mps=10,
        battery_use_per_second=0.001,
        hover_seconds=10,
        hover_battery_use_per_second=0.002,
        required_drop_liters=5,
        reserve_fraction=0.20,
    )


def test_sufficient_resources():
    result = assess_mission_resources(**sample_inputs())
    assert result.can_complete_mission
    assert result.battery_margin > 0
    assert result.water_margin_liters > 0


def test_insufficient_battery():
    inputs = sample_inputs()
    inputs["battery_level"] = 0.20
    result = assess_mission_resources(**inputs)
    assert not result.can_complete_mission
    assert any("battery" in reason.lower() for reason in result.reasons)


def test_insufficient_water():
    inputs = sample_inputs()
    inputs["water_level"] = 0.10
    result = assess_mission_resources(**inputs)
    assert not result.can_complete_mission
    assert any("water" in reason.lower() for reason in result.reasons)
