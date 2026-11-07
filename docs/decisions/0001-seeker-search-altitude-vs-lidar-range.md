# Decision 0001: Seeker search altitude (80 m AGL) vs the 40 m lidar

**Status:** PROPOSED (team to pick an option) · **Date:** 2026-11-07 · **Owners:** C (perception) + B (autonomy) + E (platform)
**Question a judge will ask:** "Your lidar only reaches 40 m but you search at 80 m. How do you avoid obstacles and see the ground?"

## Short answer
The 40 m lidar was never meant to see the ground during search. It is a **flight-level obstacle sensor**: the
Mid-360's vertical field of view is about **−7° to +52°** (looking out and up, barely down) [S8]. What it must
cover is the **stopping distance** in the direction of travel: **28 m at 15 m/s** (4 m/s² braking). 40 m covers
that with 12 m (≈ 0.8 s) to spare. Search altitude is set by the **thermal camera** (footprint vs pixel size),
not by the lidar. But the question does expose three real gaps, listed below.

## The numbers (seeker: 640×512, 32° HFOV, 15 m/s; `contracts/config.py`, computed with `CameraModel`)

| AGL | Swath | Ground pixel | 32 px patch | Search rate | 16 km² with 6 seekers (ideal) |
|---|---|---|---|---|---|
| 35 m | 20.1 m | 3.1 cm | 1.0 m | 1.08 km²/h | 2.5 h |
| 60 m | 34.4 m | 5.4 cm | 1.7 m | 1.86 km²/h | 1.4 h |
| **80 m** | **45.9 m** | **7.2 cm** | **2.3 m** | **2.48 km²/h** | **1.1 h** |
| 120 m | 68.8 m | 10.8 cm | 3.4 m | 3.72 km²/h | 0.7 h |

Lidar: 40 m at 10% reflectivity, 70 m at 80% (range is shorter toward the top of the FOV) [S8].
- **Mounted upright** (normal): at 40 m out it sees only ~5 m below its own altitude, so it **never sees the
  ground** from 80 m. That's fine for collisions, which happen at flight level.
- **Mounted inverted** (−52° to +7°): ground visible only in a ring around the drone: 16–35 m out at 20 m AGL,
  23–27 m at 30 m AGL, **nothing from 40 m up** on dark (10%) surfaces. **Even inverted, no Mid-360 sees the
  ground under an 80 m seeker.**

## Gaps the question exposes
1. **Height above ground (AGL) at 80 m is not specified.** The seeker's "laser rangefinder" has no range in
   `config.py`. Many small drone rangefinders are ~40 m class `[VERIFY]`. Without AGL we depend entirely on the
   DEM. A 5 m DEM error moves georeferenced hotspots ~1.4 m at the image edge, which is acceptable, but terrain
   following and pixel-size estimates need a real AGL reading or a trusted DEM.
2. **Lidar is degraded in smoke** (returns from particles: false obstacles or lost range) `[VERIFY]`. Thermal
   LWIR sees through thin smoke much better.
3. **Low-altitude phases need downward sensing:** INVESTIGATE descent, water survey (lidar "holes" on calm
   water, context §8), and any landing. The upright Mid-360 can't look down.

## Options

| | Option | What changes | Cost | Verdict |
|---|---|---|---|---|
| **A** | **Keep 80 m; lidar = forward obstacle sensor; add a long-range downward altimeter** | + laser altimeter rated ≥ 100 m (e.g. LightWare SF11/C class, ~120 m `[VERIFY]`) or radar altimeter, for AGL; DEM + canopy-height map for route planning; below 40 m AGL limit speed so stopping distance < 0.7 × lidar range (≤ 15 m/s gives 28 m) | ~tens of grams, ~1 W `[VERIFY]`; no coverage loss | **Recommended**: fixes gap 1, keeps search rate |
| B | Search at ~35 m so lidar can reach canopy/ground | altitude only | **2.3× slower coverage** (2.5 h vs 1.1 h for 16 km²); closer to snags, turbulence and smoke; upright lidar *still* can't see down | Reject |
| C | Longer-range lidar (Livox Avia class: ~190 m at 10%, ~0.5 kg, ~9 W `[VERIFY]`) | swap sensor | +0.23 kg, +2.5 W → ~52.5 → ~50.5 min endurance | Only if we want faster cruise or low flying in steep terrain. Doesn't solve AGL unless tilted down |
| D | A + downward-tilted / second short-range lidar for low phases | + sensor or mount | mass, integration | Stretch, for water survey and landing |
| E | A + mmWave radar for smoke-robust obstacle detection | + radar | mass, power | Stretch. Heavy ag drones already pair radar with vision for obstacles [S5]. Fixes gap 2 |

## Recommendation
**Option A now** (+ D/E as stretch). Demo talking point: *"Altitude is chosen for the thermal camera's
coverage; lidar guards the flight path within our 28 m stopping distance; a long-range altimeter and the DEM
give height above ground; we only fly low, and slowly, where the lidar can see."*

## If adopted: follow-up work
- E: add the altimeter to the seeker sensor table (context §6.3); set `rangefinder` range in `config.py` (a
  contract/config change, so it needs the integration owner).
- B: speed limit below 40 m AGL tied to `PlatformSpec.stopping_distance_m()`; smoke-aware lidar filtering.
- C: georeference with DEM + measured AGL (`TerrainGeoreferencer`), and use lidar ground points when low.
