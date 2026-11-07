
from autonomy.water_drop import evaluate_water_drop


def sample_inputs():
    return dict(
        target_confidence=0.95,
        distance_to_target_m=2,
        altitude_m=10,
        speed_mps=1,
        water_available_liters=10,
        requested_drop_liters=5,
        min_confidence=0.80,
        max_distance_m=5,
        min_altitude_m=5,
        max_altitude_m=15,
        max_speed_mps=2,
    )


def test_allows_valid_drop():
    result = evaluate_water_drop(**sample_inputs())
    assert result.should_drop
    assert result.water_to_release_liters == 5


def test_rejects_low_confidence():
    inputs = sample_inputs()
    inputs["target_confidence"] = 0.3
    result = evaluate_water_drop(**inputs)
    assert not result.should_drop


def test_never_releases_more_than_available():
    inputs = sample_inputs()
    inputs["water_available_liters"] = 2
    result = evaluate_water_drop(**inputs)
    assert result.should_drop
    assert result.water_to_release_liters == 2
