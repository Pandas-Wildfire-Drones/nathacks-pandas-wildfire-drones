"""Deterministic ranking heuristics for hotspot urgency (not fire-behaviour predictions)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from contracts import Hotspot, HotspotStatus

__all__ = ["prioritize_hotspots", "score_hotspot"]

_INACTIVE_STATUSES = {
    HotspotStatus.SUPPRESSED,
    HotspotStatus.VERIFIED_OUT,
    HotspotStatus.FALSE_POSITIVE,
}


def score_hotspot(
    hotspot: Hotspot,
    now: float,
    *,
    freshness_half_life_s: float = 300.0,
) -> float:
    """Return a [0, 1] urgency heuristic; all time values use the caller's Clock."""
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    if not math.isfinite(freshness_half_life_s) or freshness_half_life_s <= 0:
        raise ValueError("freshness_half_life_s must be finite and > 0")
    if hotspot.status in _INACTIVE_STATUSES:
        return 0.0

    temperature = min(1.0, max(0.0, (hotspot.peak_temp_c - 100.0) / 500.0))
    area = hotspot.est_area_m2 / (hotspot.est_area_m2 + 10.0)
    growth_rate = hotspot.growth_rate_m2_per_min or 0.0
    if not math.isfinite(growth_rate):
        raise ValueError("hotspot growth rate must be finite")
    growth_rate = max(0.0, growth_rate)
    growth = growth_rate / (growth_rate + 1.0)
    age_s = max(0.0, now - hotspot.last_seen)
    freshness = math.exp(-math.log(2.0) * age_s / freshness_half_life_s)

    score = (
        0.25 * hotspot.confidence
        + 0.25 * temperature
        + 0.15 * area
        + 0.20 * growth
        + 0.15 * freshness
    )
    if hotspot.status is HotspotStatus.DETECTED:
        score *= 0.8
    elif hotspot.status is HotspotStatus.ASSIGNED:
        score *= 0.7
    return min(1.0, max(0.0, score))


def prioritize_hotspots(
    hotspots: Iterable[Hotspot],
    now: float,
    *,
    freshness_half_life_s: float = 300.0,
) -> list[Hotspot]:
    """Return scored copies, highest urgency first; the input entities are not mutated."""
    ranked: list[Hotspot] = []
    for hotspot in hotspots:
        copy = hotspot.model_copy(deep=True)
        copy.priority_score = score_hotspot(
            hotspot,
            now,
            freshness_half_life_s=freshness_half_life_s,
        )
        ranked.append(copy)
    return sorted(ranked, key=lambda hotspot: (-hotspot.priority_score, hotspot.hotspot_id))