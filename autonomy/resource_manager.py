"""Resource management for drone missions."""
from dataclasses import dataclass


@dataclass
class ResourceAssessment:
    can_complete_mission: bool
    battery_required: float
    battery_margin: float
    water_available_liters: float
    water_required_liters: float
    water_margin_liters: float
    reasons: list[str]


def assess_mission_resources(
    battery_level,
    water_level,
    tank_capacity_liters,
    outbound_distance_m,
    return_distance_m,
    cruise_speed_mps,
    battery_use_per_second,
    hover_seconds,
    hover_battery_use_per_second,
    required_drop_liters,
    reserve_fraction,
):
    """Estimate whether a drone has enough resources for a mission."""

    for name, value in [
        ("battery_level", battery_level),
        ("water_level", water_level),
        ("reserve_fraction", reserve_fraction),
    ]:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    if tank_capacity_liters <= 0 or cruise_speed_mps <= 0:
        raise ValueError("Tank capacity and speed must be positive")

    values = [
        outbound_distance_m, return_distance_m,
        battery_use_per_second, hover_seconds,
        hover_battery_use_per_second, required_drop_liters,
    ]
    if any(value < 0 for value in values):
        raise ValueError("Distances, rates, times and water cannot be negative")

    flight_seconds = (
        outbound_distance_m + return_distance_m
    ) / cruise_speed_mps

    battery_required = (
        flight_seconds * battery_use_per_second
        + hover_seconds * hover_battery_use_per_second
    )

    water_available = water_level * tank_capacity_liters
    battery_margin = battery_level - battery_required - reserve_fraction
    water_margin = water_available - required_drop_liters

    reasons = []
    if battery_margin < 0:
        reasons.append("Insufficient battery for mission and safety reserve")
    if water_margin < 0:
        reasons.append("Insufficient water for planned drop")

    return ResourceAssessment(
        can_complete_mission=not reasons,
        battery_required=battery_required,
        battery_margin=battery_margin,
        water_available_liters=water_available,
        water_required_liters=required_drop_liters,
        water_margin_liters=water_margin,
        reasons=reasons,
    )
