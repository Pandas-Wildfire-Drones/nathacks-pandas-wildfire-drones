
"""Generate lawnmower coverage waypoints inside an assigned Zone."""

from math import isfinite

from contracts.entities import Zone
from contracts.geometry import Position


def _scanline_intersections(
    boundary: list[Position],
    y: float,
) -> list[float]:
    """Find the x-coordinates where a horizontal scan line crosses a polygon."""
    intersections = []

    for i, point_a in enumerate(boundary):
        point_b = boundary[(i + 1) % len(boundary)]

        x1, y1 = point_a.x, point_a.y
        x2, y2 = point_b.x, point_b.y

        # Horizontal edges are skipped to avoid duplicate intersections.
        if y1 == y2:
            continue

        # Use a half-open interval so shared vertices are not counted twice.
        if (y1 <= y < y2) or (y2 <= y < y1):
            fraction = (y - y1) / (y2 - y1)
            intersection_x = x1 + fraction * (x2 - x1)
            intersections.append(intersection_x)

    intersections.sort()

    # Remove duplicate intersections.
    unique_intersections = []

    for x in intersections:
        if (
            not unique_intersections
            or abs(x - unique_intersections[-1]) > 1e-9
        ):
            unique_intersections.append(x)

    return unique_intersections


def generate_coverage_waypoints(
    zone: Zone,
    spacing_m: float,
    *,
    z_m: float | None = None,
) -> list[Position]:
    """Generate ordered lawnmower waypoints inside a polygon-shaped Zone.

    Args:
        zone: Search region defined by the shared Zone contract.
        spacing_m: Distance between adjacent scan lines, in metres.
        z_m: Optional altitude in the shared coordinate frame.
             Defaults to the z-coordinate of the first boundary point.

    Returns:
        Ordered Position objects for the drone's coverage search.
    """
    if not isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("Spacing must be a positive finite number")

    boundary = zone.boundary

    if len(boundary) < 3:
        raise ValueError("A coverage zone must have at least 3 boundary points")

    min_x = min(point.x for point in boundary)
    max_x = max(point.x for point in boundary)
    min_y = min(point.y for point in boundary)
    max_y = max(point.y for point in boundary)

    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Zone must have positive width and height")

    flight_z = boundary[0].z if z_m is None else z_m

    if not isfinite(flight_z):
        raise ValueError("Altitude must be finite")

    height = max_y - min_y
    first_scan_y = min_y + min(spacing_m / 2, height / 2)

    waypoints: list[Position] = []
    row = 0

    while True:
        y = first_scan_y + row * spacing_m

        if y >= max_y:
            break

        intersections = _scanline_intersections(boundary, y)

        # Each pair represents one section of the polygon crossed by the
        # horizontal scan line. Concave polygons may have multiple sections.
        segments = [
            (intersections[i], intersections[i + 1])
            for i in range(0, len(intersections) - 1, 2)
            if intersections[i + 1] - intersections[i] > 1e-9
        ]

        # Reverse segment order on alternate rows.
        if row % 2 == 1:
            segments.reverse()

        for left_x, right_x in segments:
            # Keep endpoints slightly inside the polygon boundary.
            inset = min(1e-6, (right_x - left_x) / 4)

            if row % 2 == 0:
                x_points = (left_x + inset, right_x - inset)
            else:
                x_points = (right_x - inset, left_x + inset)

            for x in x_points:
                waypoints.append(Position(x=x, y=y, z=flight_z))

        row += 1

    return waypoints