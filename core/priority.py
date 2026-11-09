"""Fire priority ("intensity") from what the swarm can actually observe.

A hotspot report gives us: est_area_m2 (A), growth_rate_m2_per_min (G, tracker slope, often None after one pass),
peak_temp_c (T) and confidence (c). We rank fires by the burning area we expect to face when a drone gets there,
weighted by how hot (= how intense) they burn:

    G_eff      = G if measured, else  2 * sqrt(pi * A) * ROS_prior(T)      (circle of area A growing at ROS)
    A_proj     = A + max(G_eff, 0) * horizon_min                            (area at the response horizon)
    w_T        = 0.5 + 0.5 * clip((T - T_lo) / (T_hi - T_lo), 0, 1)         (0.5 smoulder .. 1.0 flaming)
    raw        = c * A_proj * w_T                                           (Hotspot.priority_score)
    priority   = 1 - exp(-raw / scale)                                      (Task.priority, 0..1)

Why this shape:
  * Size and growth decide how much water a fire will need and how fast that number rises; projecting to a
    horizon turns both into one number with units (m2), so "small but fast" and "big but static" compare fairly.
  * Temperature is our only observable proxy for intensity / rate of spread (Byram intensity ~ fuel x ROS): a
    flaming spot reads near camera saturation, a smoulder reads 150-300 C. ROS_prior(T) is a [ASSUMPTION]
    log-linear ramp between the night ROS of a smoulder (~0.03 m/min) and an active spot (~0.4 m/min).
  * Confidence discounts unconfirmed or decoy-like detections.
Measured growth (re-observation by a second seeker pass or an extinguisher's VERIFY) replaces the prior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from contracts import Hotspot

__all__ = ["DEFAULT_PRIORITY", "PriorityParams", "fire_priority", "normalized_priority", "projected_area_m2", "ros_prior_m_min"]


@dataclass(frozen=True)
class PriorityParams:
    horizon_min: float = 5.0          # look-ahead: roughly one response cycle in real minutes [ASSUMPTION]
    t_lo_c: float = 150.0             # at/below: smoulder-class intensity
    t_hi_c: float = 500.0             # at/above: flaming-class intensity (seeker low gain saturates ~550 C)
    ros_lo_m_min: float = 0.03        # ROS prior at t_lo (night smoulder) [ASSUMPTION]
    ros_hi_m_min: float = 0.40        # ROS prior at t_hi (night active spot) [ASSUMPTION]
    min_area_m2: float = 0.25
    scale: float = 25.0               # raw score that maps to priority ~0.63


DEFAULT_PRIORITY = PriorityParams()


def _temp_frac(t_c: float, p: PriorityParams) -> float:
    return min(max((t_c - p.t_lo_c) / (p.t_hi_c - p.t_lo_c), 0.0), 1.0)


def ros_prior_m_min(peak_temp_c: float, p: PriorityParams = DEFAULT_PRIORITY) -> float:
    """Rate-of-spread prior from temperature: log-linear between the smoulder and active-spot values."""
    f = _temp_frac(peak_temp_c, p)
    return math.exp(math.log(p.ros_lo_m_min) * (1 - f) + math.log(p.ros_hi_m_min) * f)


def projected_area_m2(area_m2: float, peak_temp_c: float, growth_m2_min: float | None,
                      p: PriorityParams = DEFAULT_PRIORITY) -> float:
    a = max(area_m2, p.min_area_m2)
    g = growth_m2_min if growth_m2_min is not None else 2.0 * math.sqrt(math.pi * a) * ros_prior_m_min(peak_temp_c, p)
    return a + max(g, 0.0) * p.horizon_min


def fire_priority(h: Hotspot, p: PriorityParams = DEFAULT_PRIORITY, growth_m2_min: float | None = None) -> float:
    """Raw threat score (>= 0, m2-like). `growth_m2_min` overrides the hotspot's own growth estimate."""
    g = growth_m2_min if growth_m2_min is not None else h.growth_rate_m2_per_min
    a_proj = projected_area_m2(h.est_area_m2, h.peak_temp_c, g, p)
    w_t = 0.5 + 0.5 * _temp_frac(h.peak_temp_c, p)
    return h.confidence * a_proj * w_t


def normalized_priority(raw: float, p: PriorityParams = DEFAULT_PRIORITY) -> float:
    return 1.0 - math.exp(-max(raw, 0.0) / p.scale)
