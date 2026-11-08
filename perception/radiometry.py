"""LWIR radiometry helpers: temperature <-> band radiance (Planck at 10 um, LWIR band centre).

Why it matters: a pixel's reading is the AVERAGE RADIANCE of everything in its footprint, not the average
temperature. Mixing or resampling temperatures linearly under-states small hot spots. Any code that changes
resolution (synthetic rendering, scale-normalized patches) must mix in radiance and convert back.
Single-wavelength approximation [ASSUMPTION]; a real 8-14 um band integral changes numbers by a few percent.
"""
from __future__ import annotations

import numpy as np

__all__ = ["planck_radiance", "radiance_to_temp"]

_C1, _C2 = 1.191042e-16, 1.4387769e-2  # 2hc^2 [W m^2 / sr], hc/k [m K]
_LAMBDA = 10e-6


def planck_radiance(temp_c, lam: float = _LAMBDA):
    t = np.asarray(temp_c, dtype=np.float64) + 273.15
    return _C1 / (lam ** 5 * (np.exp(_C2 / (lam * t)) - 1.0))


def radiance_to_temp(radiance, lam: float = _LAMBDA):
    return _C2 / (lam * np.log1p(_C1 / (lam ** 5 * np.maximum(radiance, 1e-30)))) - 273.15
