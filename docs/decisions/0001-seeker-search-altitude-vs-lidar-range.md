# Decision 0001: Seeker search altitude (80 m AGL) vs the 40 m lidar

**Status:** ACCEPTED, Option A (2026-11-07) · **Owners:** C (perception) + B (autonomy) + E (platform)
**Alternatives on file:** C (long-range lidar), D (downward lidar), E (mmWave radar)

> **Judge question:** "Your lidar only reaches 40 m but you search at 80 m. How do you avoid obstacles and see the ground?"
>
> **Our answer:** "Altitude is chosen for the thermal camera's coverage. The lidar guards the flight path within
> our 28 m stopping distance. A long-range altimeter plus the terrain model gives height above ground, and we only
> fly low, and slowly, where the lidar can see."

## Context

The 40 m lidar was never meant to see the ground during search. It is a **flight-level obstacle sensor**: the
Mid-360's vertical field of view is about **−7° to +52°**, looking out and up and barely down [S8]. What it must
cover is the **stopping distance** in the direction of travel: **28 m at 15 m/s** (4 m/s² braking). 40 m covers
that with 12 m (≈ 0.8 s) to spare. Search altitude is set by the **thermal camera** (footprint vs pixel size).

| AGL | Swath | Ground pixel | Search rate | 16 km² with 6 seekers (ideal) |
|---|---|---|---|---|
| 35 m | 20.1 m | 3.1 cm | 1.08 km²/h | 2.5 h |
| 60 m | 34.4 m | 5.4 cm | 1.86 km²/h | 1.4 h |
| **80 m** | **45.9 m** | **7.2 cm** | **2.48 km²/h** | **1.1 h** |
| 120 m | 68.8 m | 10.8 cm | 3.72 km²/h | 0.7 h |

Lidar geometry: mounted upright it sees only ~5 m below its own altitude at 40 m out, so it never sees the ground
from 80 m (fine: collisions happen at flight level). Mounted inverted (−52° to +7°), dark ground is visible only in
a ring (16–35 m out at 20 m AGL, 23–27 m at 30 m, **nothing from 40 m up**).

The question exposed three real gaps:
1. **No height-above-ground (AGL) sensor rated for 80 m**: the seeker rangefinder had no range in `config.py`.
2. **Lidar degrades in smoke** (false returns from particles) `[VERIFY]`.
3. **Low-altitude phases need downward sensing:** INVESTIGATE descent, water survey, landing.

## Decision: Option A

Keep **80 m search**. The lidar is the forward/side **obstacle sensor at flight level**. Add a **long-range downward
altimeter** (laser rated ≥ 100 m, e.g. LightWare SF11/C class ~120 m `[VERIFY]`, or a radar altimeter) for AGL.
Use the DEM + canopy-height map for route planning. Below 40 m AGL, limit speed so stopping distance stays under
0.7 × lidar range (15 m/s gives 28 m).

**Why:** it fixes gap 1 for ~tens of grams and ~1 W `[VERIFY]`, keeps the full search rate, and needs no new
perception model. It is already wired into the code: `SeekerPerception(altimeter_max_m=120)` uses the altimeter
reading when it is within its rated range and falls back to pose height minus the DEM (decision 0002).

## Alternatives considered (keep for the demo Q&A)

| | Option | What it buys | What it costs | Why not now |
|---|---|---|---|---|
| B | Search at ~35 m so the lidar can reach the canopy | lidar sees the canopy top | **2.3× slower coverage** (2.5 h vs 1.1 h for 16 km²); closer to snags, turbulence and smoke; an upright lidar **still** can't look down | Rejected: loses the main metric and doesn't fix the problem |
| **C** | **Longer-range lidar** (Livox Avia class: ~190 m on 10% targets, ~0.5 kg, ~9 W `[VERIFY]`) | Detection range ~5× longer, so faster cruise or low flight in steep terrain becomes safe; earlier warning of towers and snags | +0.23 kg and +2.5 W over the Mid-360 → endurance ~52.5 → ~50.5 min; doesn't give AGL unless tilted down; narrower FOV (forward only) | **Best upgrade if we raise cruise speed** (stopping distance at 20 m/s is 50 m, beyond the Mid-360's 40 m) |
| **D** | **A + downward short-range lidar** (or tilt the Mid-360 for low phases) | Ground and canopy height directly below during INVESTIGATE, water survey ("holes" on calm water, context §8) and landing; local terrain for georeferencing at close range | Extra sensor mass and integration, or losing upward coverage if the Mid-360 is tilted | **Stretch:** do it when water-source validation (SURVEY_WATER) becomes real |
| **E** | **A + mmWave radar** for obstacles | Works through smoke and dust where lidar returns are polluted; heavy ag drones already pair radar with vision [S5] | Coarse angular resolution (misses thin wires `[VERIFY]`), mass, power | **Stretch:** the answer to "what about smoke?" (gap 2) |

**Demo framing:** A is the baseline we built. C is the upgrade path for faster flight, D for water survey and
landing, E for smoke. Each one fixes one named gap, so they can be adopted independently.

## Follow-up

- E: add the altimeter to the seeker sensor table (context §6.3) and give `rangefinder` a range in `config.py`
  (a contract/config change, so it needs the integration owner).
- B: speed limit below 40 m AGL tied to `PlatformSpec.stopping_distance_m()`; smoke-aware lidar filtering.
- C (done): georeferencing uses the altimeter reading when valid, else the DEM (`perception/pipeline.py`).
