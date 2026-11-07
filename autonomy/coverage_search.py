"""Coverage search algorithm for generating waypoints in a rectangular area."""

def generate_coverage_waypoints(
    x_min, y_min, x_max, y_max, spacing_m
):
    """Generate lawnmower waypoints over a rectangular search area."""
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Area bounds must have positive dimensions")
    if spacing_m <= 0:
        raise ValueError("Spacing must be positive")

    waypoints = []
    y = y_min
    row = 0

    while y <= y_max:
        if row % 2 == 0:
            waypoints.extend([(x_min, y), (x_max, y)])
        else:
            waypoints.extend([(x_max, y), (x_min, y)])

        y += spacing_m
        row += 1

    # Ensure the top boundary is covered if spacing doesn't land on it.
    if waypoints[-1][1] < y_max:
        if row % 2 == 0:
            waypoints.extend([(x_min, y_max), (x_max, y_max)])
        else:
            waypoints.extend([(x_max, y_max), (x_min, y_max)])

    return waypoints
