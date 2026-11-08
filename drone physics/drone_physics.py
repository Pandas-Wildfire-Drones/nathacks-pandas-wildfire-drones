"""NatHacks Team E: preliminary drone and suppression calculations.

All inputs use SI units unless stated otherwise. No aircraft specifications
are assumed. Outputs are estimates, not flight-safety certifications.
"""
from __future__ import annotations
import math

G = 9.80665  # m/s^2


def _positive(**values):
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and > 0')


def _nonnegative(**values):
    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and >= 0')


def total_mass(empty_mass_kg: float, fuel_mass_kg: float, water_mass_kg: float,
               other_payload_kg: float = 0.0) -> float:
    """Takeoff mass (kg); empty mass must include battery/generator hardware."""
    _positive(empty_mass_kg=empty_mass_kg)
    _nonnegative(fuel_mass_kg=fuel_mass_kg, water_mass_kg=water_mass_kg,
                 other_payload_kg=other_payload_kg)
    return empty_mass_kg + fuel_mass_kg + water_mass_kg + other_payload_kg


def rotor_disk_area(rotor_count: int, diameter_m: float) -> float:
    """Sum of rotor disk areas (m^2), assuming equal, non-overlapping disks.

    Coaxial/overlapping rotors require a corrected effective area/model.
    """
    if isinstance(rotor_count, bool) or not isinstance(rotor_count, int) or rotor_count < 1:
        raise ValueError('rotor_count must be a positive integer')
    _positive(diameter_m=diameter_m)
    return rotor_count * math.pi * (diameter_m / 2) ** 2


def ideal_hover_power(mass_kg: float, disk_area_m2: float,
                      air_density_kg_m3: float = 1.225) -> float:
    """Ideal induced hover power in W (actuator-disk momentum theory)."""
    _positive(mass_kg=mass_kg, disk_area_m2=disk_area_m2,
              air_density_kg_m3=air_density_kg_m3)
    return (mass_kg * G) ** 1.5 / math.sqrt(2 * air_density_kg_m3 * disk_area_m2)


def estimated_hover_power(mass_kg: float, disk_area_m2: float,
                          overall_factor: float, auxiliary_power_w: float = 0.0,
                          air_density_kg_m3: float = 1.225) -> float:
    """Approximate electrical hover power (W).

    overall_factor is a calibrated ideal-induced-to-electrical factor in (0,1].
    It lumps rotor/profile, motor, ESC, interference and other losses; not a
    universal efficiency. Prefer measured aircraft data when available.
    """
    _positive(overall_factor=overall_factor)
    if overall_factor > 1:
        raise ValueError('overall_factor must be <= 1')
    _nonnegative(auxiliary_power_w=auxiliary_power_w)
    return (ideal_hover_power(mass_kg, disk_area_m2, air_density_kg_m3)
            / overall_factor + auxiliary_power_w)


def power_margin(generator_electric_w: float, required_electric_w: float) -> float:
    """Continuous electrical power surplus/deficit (W); negative = deficit."""
    _nonnegative(generator_electric_w=generator_electric_w,
                 required_electric_w=required_electric_w)
    return generator_electric_w - required_electric_w


def battery_endurance(usable_battery_wh: float, average_electric_power_w: float) -> float:
    """Battery-only endurance in hours at constant average electrical load."""
    _nonnegative(usable_battery_wh=usable_battery_wh)
    _positive(average_electric_power_w=average_electric_power_w)
    return usable_battery_wh / average_electric_power_w


def fuel_consumption(generator_electric_kw: float, hours: float,
                     specific_fuel_consumption_kg_per_kwh: float) -> float:
    """Fuel mass (kg) from *electrical-output-based* specific consumption.

    SFC must be based on generator electrical output, not engine shaft output.
    """
    _nonnegative(generator_electric_kw=generator_electric_kw, hours=hours)
    _positive(specific_fuel_consumption_kg_per_kwh=specific_fuel_consumption_kg_per_kwh)
    return generator_electric_kw * hours * specific_fuel_consumption_kg_per_kwh


def travel_time(distance_m: float, ground_speed_m_s: float) -> float:
    """Travel time (seconds), assuming constant ground speed, no acceleration."""
    _nonnegative(distance_m=distance_m)
    _positive(ground_speed_m_s=ground_speed_m_s)
    return distance_m / ground_speed_m_s


def stopping_distance(speed_m_s: float, deceleration_m_s2: float,
                      response_delay_s: float = 0.0, safety_margin_m: float = 0.0) -> float:
    """1D stopping distance (m), constant deceleration, still-air simplification."""
    _nonnegative(speed_m_s=speed_m_s, response_delay_s=response_delay_s,
                 safety_margin_m=safety_margin_m)
    _positive(deceleration_m_s2=deceleration_m_s2)
    return (speed_m_s * response_delay_s
            + speed_m_s ** 2 / (2 * deceleration_m_s2) + safety_margin_m)


def water_required(fire_area_m2: float, application_rate_l_m2: float,
                   delivery_fraction: float) -> float:
    """Scenario water demand (L); NOT a validated extinguishment predictor."""
    _nonnegative(fire_area_m2=fire_area_m2)
    _positive(application_rate_l_m2=application_rate_l_m2,
              delivery_fraction=delivery_fraction)
    if delivery_fraction > 1:
        raise ValueError('delivery_fraction must be <= 1')
    return fire_area_m2 * application_rate_l_m2 / delivery_fraction


def circular_fire_area(initial_radius_m: float, spread_rate_m_s: float,
                       elapsed_s: float) -> float:
    """Toy circular fire-growth model; constant radial spread, no wind/terrain."""
    _nonnegative(initial_radius_m=initial_radius_m,
                 spread_rate_m_s=spread_rate_m_s, elapsed_s=elapsed_s)
    return math.pi * (initial_radius_m + spread_rate_m_s * elapsed_s) ** 2


def update_mission_mass(empty_mass_kg: float, fuel_mass_kg: float,
                        water_mass_kg: float, water_dropped_kg: float = 0.0,
                        fuel_burned_kg: float = 0.0,
                        other_payload_kg: float = 0.0) -> dict:
    """Apply a mission event and return updated masses (kg).

    Water/fuel cannot be consumed beyond the available amount. Recompute
    hover power with result['total_mass_kg'] after each event.
    """
    total_mass(empty_mass_kg, fuel_mass_kg, water_mass_kg, other_payload_kg)
    _nonnegative(water_dropped_kg=water_dropped_kg,
                 fuel_burned_kg=fuel_burned_kg)
    if water_dropped_kg > water_mass_kg:
        raise ValueError('water_dropped_kg exceeds remaining water')
    if fuel_burned_kg > fuel_mass_kg:
        raise ValueError('fuel_burned_kg exceeds remaining fuel')
    remaining_water = water_mass_kg - water_dropped_kg
    remaining_fuel = fuel_mass_kg - fuel_burned_kg
    return {
        'fuel_mass_kg': remaining_fuel,
        'water_mass_kg': remaining_water,
        'total_mass_kg': total_mass(empty_mass_kg, remaining_fuel,
                                    remaining_water, other_payload_kg),
    }
