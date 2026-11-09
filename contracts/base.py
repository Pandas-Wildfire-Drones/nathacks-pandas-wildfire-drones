"""Shared Pydantic base classes for every contract model.

Why two bases?
  * Contract       - mutable (entities that change state: Hotspot, Task, DroneState...).
  * FrozenContract - immutable + hashable (values and anything that travels on the bus).

Both reject unknown fields (extra="forbid"), so a typo like `peak_temp` instead of
`peak_temp_c` fails loudly at the boundary instead of silently becoming None later.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["CONTRACT_VERSION", "Contract", "FrozenContract", "Seconds"]

# Bump when a field is renamed/removed or semantics change. Add a line to the
# changelog in context.md. Messages carry this in `SwarmMessage.v`.
CONTRACT_VERSION = "0.3.1"

# All timestamps/durations are float SECONDS in the Clock's timebase
# (sim time in the simulator, wall-clock epoch seconds on real hardware).
Seconds = float


class Contract(BaseModel):
    """Mutable contract model. Validates on assignment too."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FrozenContract(BaseModel):
    """Immutable, hashable contract model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
