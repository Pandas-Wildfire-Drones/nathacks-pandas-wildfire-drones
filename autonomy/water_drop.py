"""drone should only release water when the target is sufficiently reliable and the drone is within configured release limits"""
from dataclasses import dataclass


@dataclass
class DropDecision:
    should_drop: bool
    water_to_release_liters: float
    reason: str


def evaluate_water_drop(
    target_confidence,
    distance_to_target_m,
    altitude_m,
    speed_mps,
    water_available_liters,
    requested_drop_liters,
    min_confidence,
    max_distance_m,
    min_altitude_m,
    max_altitude_m,
    max_speed_mps,
):
    """Evaluate whether release conditions are satisfied."""

    fractions = [target_confidence]
    if any(not 0 <= value <= 1 for value in fractions):
        raise ValueError("Target confidence must be between 0 and 1")

    values = [
        distance_to_target_m, altitude_m, speed_mps,
        water_available_liters, requested_drop_liters,
        min_confidence, max_distance_m, min_altitude_m,
        max_altitude_m, max_speed_mps,
    ]
    if any(value < 0 for value in values):
        raise ValueError("Drop inputs and limits cannot be negative")
    if min_altitude_m > max_altitude_m:
        raise ValueError("Minimum altitude exceeds maximum altitude")

    if target_confidence < min_confidence:
        return DropDecision(False, 0, "Target confidence too low")
    if distance_to_target_m > max_distance_m:
        return DropDecision(False, 0, "Drone is too far from target")
    if not min_altitude_m <= altitude_m <= max_altitude_m:
        return DropDecision(False, 0, "Altitude outside release limits")
    if speed_mps > max_speed_mps:
        return DropDecision(False, 0, "Drone is moving too fast")
    if water_available_liters <= 0:
        return DropDecision(False, 0, "No water available")
    if requested_drop_liters <= 0:
        return DropDecision(False, 0, "Requested drop must be positive")

    amount = min(water_available_liters, requested_drop_liters)

    return DropDecision(True, amount, "Release conditions satisfied")
