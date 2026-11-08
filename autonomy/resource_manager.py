"""Assess whether a drone has enough resources to complete a mission."""

from dataclasses import dataclass
from math import isfinite

from contracts.entities import DroneState
from contracts.enums import Powertrain


@dataclass
class ResourceAssessment:
    """Resource estimate and recommendation for an individual drone."""

    can_complete_mission: bool
    battery_required_pct: float
    battery_margin_pct: float
    fuel_required_kg: float
    fuel_margin_kg: float
    water_available_liters: float
    water_required_liters: float
    water_margin_liters: float
    mission_duration_s: float
    reasons: list[str]

    @property
    def should_return(self) -> bool:
        """Whether the drone should return instead of continuing."""
        return not self.can_complete_mission


def assess_mission_resources(
    drone: DroneState,
    outbound_distance_m: float,
    return_distance_m: float,
    cruise_speed_mps: float,
    *,
    battery_use_per_second: float = 0.0,
    hover_seconds: float = 0.0,
    hover_battery_use_per_second: float = 0.0,
    required_drop_liters: float = 0.0,
    reserve_fraction: float = 0.20,
    fuel_use_kg_per_second: float = 0.0,
    fuel_reserve_kg: float = 0.0,
) -> ResourceAssessment:
    """Estimate whether a drone can complete a mission and return safely.

    Args:
        drone: Current telemetry snapshot using the shared DroneState contract.
        outbound_distance_m: Distance to the mission destination.
        return_distance_m: Distance from the destination back to base.
        cruise_speed_mps: Assumed average travel speed.
        battery_use_per_second: Electric battery fraction consumed per second.
            For example, 0.001 means 0.1 percentage points per second.
        hover_seconds: Estimated time spent hovering during the mission.
        hover_battery_use_per_second: Additional battery fraction consumed
            per second while hovering.
        required_drop_liters: Water required for the planned drop.
        reserve_fraction: Minimum battery reserve as a fraction from 0 to 1.
        fuel_use_kg_per_second: Hybrid fuel consumption in kg per second.
        fuel_reserve_kg: Minimum fuel that must remain after the mission.

    Notes:
        HYBRID drones use fuel for their normal sortie. Their battery is
        treated as an emergency reserve rather than normal flight energy.
        Consumption rates must come from the aircraft performance model.
    """
    fraction_values = {
        "reserve_fraction": reserve_fraction,
    }

    for name, value in fraction_values.items():
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")

    non_negative_values = {
        "outbound_distance_m": outbound_distance_m,
        "return_distance_m": return_distance_m,
        "hover_seconds": hover_seconds,
        "battery_use_per_second": battery_use_per_second,
        "hover_battery_use_per_second": hover_battery_use_per_second,
        "required_drop_liters": required_drop_liters,
        "fuel_use_kg_per_second": fuel_use_kg_per_second,
        "fuel_reserve_kg": fuel_reserve_kg,
    }

    for name, value in non_negative_values.items():
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    if not isfinite(cruise_speed_mps) or cruise_speed_mps <= 0:
        raise ValueError("Cruise speed must be positive and finite")

    flight_seconds = (
        outbound_distance_m + return_distance_m
    ) / cruise_speed_mps

    mission_duration_s = flight_seconds + hover_seconds

    battery_required_fraction = 0.0
    fuel_required_kg = 0.0

    if drone.powertrain == Powertrain.ELECTRIC:
        if mission_duration_s > 0 and (
            battery_use_per_second + hover_battery_use_per_second <= 0
        ):
            raise ValueError(
                "Electric drones require a positive battery consumption rate"
            )

        battery_required_fraction = (
            flight_seconds * battery_use_per_second
            + hover_seconds * hover_battery_use_per_second
        )

        battery_required_pct = battery_required_fraction * 100
        battery_margin_pct = (
            drone.battery_pct
            - battery_required_pct
            - reserve_fraction * 100
        )

        # Fuel is not used by electric drones.
        fuel_margin_kg = 0.0

    elif drone.powertrain == Powertrain.HYBRID:
        if mission_duration_s > 0 and fuel_use_kg_per_second <= 0:
            raise ValueError(
                "Hybrid drones require a positive fuel consumption rate"
            )

        fuel_required_kg = mission_duration_s * fuel_use_kg_per_second

        # Hybrid battery is an emergency reserve, not normal sortie energy.
        battery_required_pct = 0.0
        battery_margin_pct = drone.battery_pct - reserve_fraction * 100

        fuel_margin_kg = (
            drone.fuel_remaining_kg - fuel_required_kg - fuel_reserve_kg
        )

    else:
        raise ValueError(f"Unsupported powertrain: {drone.powertrain}")

    water_available = drone.water_remaining_l or 0.0
    water_margin = water_available - required_drop_liters

    reasons = []

    if battery_margin_pct < 0:
        reasons.append(
            "Insufficient battery for mission and emergency reserve"
        )

    if drone.powertrain == Powertrain.HYBRID and fuel_margin_kg < 0:
        reasons.append(
            "Insufficient fuel for mission and fuel reserve"
        )

    if water_margin < 0:
        reasons.append("Insufficient water for planned drop")

    if (
        drone.endurance_remaining_s is not None
        and mission_duration_s > drone.endurance_remaining_s
    ):
        reasons.append("Mission exceeds remaining endurance")

    return ResourceAssessment(
        can_complete_mission=not reasons,
        battery_required_pct=battery_required_pct,
        battery_margin_pct=battery_margin_pct,
        fuel_required_kg=fuel_required_kg,
        fuel_margin_kg=fuel_margin_kg,
        water_available_liters=water_available,
        water_required_liters=required_drop_liters,
        water_margin_liters=water_margin,
        mission_duration_s=mission_duration_s,
        reasons=reasons,
    )
