# natHacks 2026 · Team Pandas · Night Initial-Attack Drone Swarm — CONTEXT

`context_version: 0.1` · `contract_version: 0.1.0` · last updated 2026-11-07

---

## 0. READ THIS FIRST (instructions for any agent working on this project)

1. **Purpose.** This file is the shared brief so a fresh chat can be productive in minutes. Read §1–§5, then your workstream in §2, then §11 (contracts). Read §6–§10 when your task touches sensors, agents, refill logistics, gas sensing or ML data.
2. **Adjust this document when it is wrong or incomplete.** Edit the relevant section *in place* (don't append contradictions). Add one line to §13 (date · chat/author · what · why). Keep it concise: tables over prose.
3. **Changing a contract** (anything in `contracts/`): edit the code, bump `CONTRACT_VERSION` in `contracts/base.py` and the version line above, run `python -m pytest -q`, then run `python tools/sync_contracts.py` to refresh §11. Tell the human/team lead — contracts are the one thing every workstream depends on.
4. **Tags.** `[SOURCED Sx]` = backed by a URL in §14. `[ASSUMPTION]` = our choice, owner should confirm. `[VERIFY]` = plausible but unchecked. `[OPEN]` = needs a human decision. Never promote an assumption to a fact without a source. Never invent specs or citations.
5. **Stay in your lane (§2).** Cross-cutting changes (contracts, topics, coordinate conventions, scenario schema, hard rules in §4) need a human OK first.
6. **Ask the human a short question** only for genuinely human decisions (`[OPEN]` items in §12). Otherwise take the default written here, say which default you took, and move on.
7. **Don't tune physical numbers to make the demo look better.** Physical values live in `contracts/config.py` and are owned by the mech E (workstream E).

---

## 1. Mission and constraints

**How might we design a coordinated system of autonomous drones that can deliver water to active wildfires safely and efficiently at night?** (natHacks 2026 challenge.)

**Provider brief (paraphrased):**
- *Platform:* no fixed platform. Assume a **heavy-lift multirotor**; pick realistic numbers, state them, **show the math**. Payload and endurance are our call.
- *Scenario:* **night initial attack** — small hotspots / spot fires, caught before they grow; fire behaviour drops overnight, that's the window. Think **swarm finding, prioritizing and hitting hotspots**, not fighting a full fire front.
- *Data:* nothing proprietary. Start with **NASA FIRMS** (hotspots), **CWFIS** (Canadian fire weather and behaviour), **Alberta Wildfire** historical data. Thermal/IR imagery datasets are fair game.
- *Generator/endurance:* longer-term hardware goal. **Assume a generator exists giving 5–8 h flight time.** Spend our effort on the swarm: **coordination, detection, tasking, autonomy.**

**Our decisions so far (Dev Reports #1–#2):**
- Criteria and weights: detection of wildfires at night (>90% F1) = 3; navigation at night (obstacle avoidance) = 3; efficient extinguishing = 3; swarm coordination = 2; economical/effective drone design = 1.
- Scored 9 approaches; the **mixed algorithmic + ML** approach won (85/85 for "LWIR + lidar" and "LWIR + NIR/SWIR, DEM, RTK-GNSS"). Radar was dropped for now (see §6.4 for a correction: radar is not limited by darkness; its weakness is coarse resolution).
- Multiple drone types: **Seekers** (find/confirm) and **Extinguishers** (carry water).

**Demo reality:** the final demo is **purely software**, shown in a room — no drones, no real fires. Sensor data is synthetic or comes from static datasets. The **stretch goal** is code that runs on real drones (ROS 2 / PX4 / Jetson) without rewriting the logic.

---

## 2. Team and workstreams (team of 5: 1 mech E, 4 CS)

| ID | Role | Owns | Talks to others via |
|---|---|---|---|
| **A** (CS) | Swarm brain | Leader role, election/failover, task allocation (auction/greedy → CBBA stretch), hotspot prioritization, refill assignment, comms protocol, failure handling | messages in `contracts/messages.py` |
| **B** (CS) | Per-drone autonomy | Coverage search, path planning, obstacle avoidance, deconfliction (`PathClaim`), endurance logic, extinguisher drop logic, refill execution, self-health checks | `Autopilot`, `PayloadSystem`, `Bus` |
| **C** (CS) | Perception | Detectors (threshold, ONNX, fake), georeferencing, log-odds map, tracking, F1 evaluation, model export; water-body detection (stretch) | `Detector`, `Georeferencer`, `ThermalFrame` |
| **D** (CS) | Ground base + frontend | FastAPI backend, DB, health monitor, ops console (map, alerts), the "breathing polygons" cinematic sim view | topics `ground/*`, `drone/*/heartbeat`, WebSocket |
| **E** (Mech E) | Assumptions and physics | Platform/payload math, sensor footprint, fire behaviour, agent and water models, scenario data (FIRMS/CWFIS/Alberta), thermal-frame synthesizer (with C), the "show your math" document | `contracts/config.py` |

Gaps to cover (assign owners): **integration owner** (contracts, merges, end-to-end build — `[ASSUMPTION]` team lead O'Mario), **sim engine owner** (runner, `SimBus`, fire model wiring — `[OPEN]`, A builds the Phase-1 skeleton then hands over), **evaluation harness** (Monte Carlo metrics), **story/demo owner** (dev reports, pitch, backup video).

Rule of thumb: nobody waits on anybody — `FakeDetector`, `SimAutopilot`, `SimBus` and `example_scenario()` let every workstream run from day one.

---

## 3. Product vision and phases

**Final product:** (1) mixed algorithmic + ML imaging/analysis; (2) coordination, comms, navigation and ground-base code with a firefighter-facing frontend (hotspot/heat map, drone health, alerts) plus backend and DB; (3) a demoable abstract simulation with a modern "breathing polygons" art style, realistic conditions embedded.

| Phase | Goal | Done when |
|---|---|---|
| **0** Contracts + skeleton | Contracts (done), `SimBus`/`SimClock`/`SimAutopilot` stubs, `FakeDetector`, example scenario, repo + tests | everyone can import `contracts` and run a no-op swarm |
| **1** Walking skeleton | 1 fire, 3 seekers, 1 extinguisher, fake detector, leader allocates, dots on a dashboard, a "drop" | end-to-end loop runs ugly but complete |
| **2** Depth | Real detectors + georeferencing, log-odds map, auction + election, coverage planning, ground health + alerts, real fire model, refill from station | each workstream replaces its stubs |
| **3** Realism + polish | Chaos buttons, Monte Carlo metrics, decoys, water-basin refill, visual style, backup video | demo script rehearsed |
| **4** Stretch | `Ros2Bus`/`MqttBus`, PX4 SITL adapter, Jetson ONNX build, basin validation, gas sensing | logic runs unchanged on a second fidelity level |

**Fidelity levels** (same code, different implementations of the `contracts/interfaces.py` protocols): **Level 0** abstract Python sim (the demo, any OS, hundreds of drones) · **Level 1** PX4 SITL + Gazebo (real flight firmware; `[SOURCED S14]` Linux/WSL2; multi-vehicle Gazebo is Linux-only) · **Level 2** real drones.

**Repo layout:**
```
contracts/    shared Pydantic models, topics, interfaces, config   (this doc §11)
core/         pure-Python swarm logic: leader, allocation, planning, map   (no ROS/web imports)
perception/   detectors, georeferencing, mapping/tracking
sim/          world, fire model, sensor/comms models, scenario runner, metrics
groundbase/   FastAPI, health monitor, DB
frontend/     React: ops console + cinematic sim view
adapters/     (stretch) ros2, mqtt, px4
scenarios/    YAML/JSON ScenarioConfig files
tests/  tools/  context.md
```

---

## 4. Architecture and hard rules

```
                 ┌───────────────── GROUND BASE ─────────────────┐
                 │ health monitor · DB · ops console · RTK base   │
                 └──────▲────────────────────────────┬───────────┘
        heartbeats,     │                            │ GroundCommand
        alerts          │                            ▼
                 ┌──────┴────────────────────────────────────────┐
                 │   LEADER = a ROLE held by one drone (+standby) │
                 │ zones · hotspot priority · auction · refill    │
                 └──────▲────────────────────────────┬───────────┘
        hotspots, bids  │                            │ TASK_ASSIGN / REFILL_GRANT
        ┌───────────────┴──────┐          ┌──────────▼───────────────┐
        │ SEEKERS              │          │ EXTINGUISHERS            │
        │ thermal + lidar      │          │ tank + agent + thermal   │
        └──────────────────────┘          └──────────────────────────┘
```

**Hard rules (change only with human OK):**
1. **Contract-first.** Drone logic imports only `contracts` + stdlib/numpy. No ROS, PX4, FastAPI or web imports inside `core/` or `perception/`.
2. **Everything crosses `encode()`/`decode()`**, even on the in-process bus, so drones never share Python objects. Receivers validate on decode.
3. **Raw frames and point clouds never travel on the bus.** Only `Detection → Hotspot`, obstacles and map deltas do. (Radio bandwidth is the reason; see `CommsSpec`.)
4. **Safety is local.** Collision avoidance, geofence, low-endurance return-to-base and link-loss behaviour run on each drone and never depend on the leader or the ground link.
5. **Time comes from `Clock`** (float seconds). No `time.time()` / `datetime.now()` in core logic. Sim must be deterministic given `ScenarioConfig.seed`.
6. **Conventions** (see `contracts/geometry.py`): ENU meters, local origin; `z` = height above scenario datum (not AGL); compass headings; grids `[row, col]` with row 0 = south; PX4 NED/FRD conversion only inside the PX4 adapter.
7. **Commands carry a leader `epoch`.** Ignore commands whose epoch is lower than the highest you've seen (stops a stale leader after failover).
8. **Humans declare fires out.** Drones can mark a hotspot `SUPPRESSED`; only `GroundCommand(CONFIRM_OUT)` produces `VERIFIED_OUT` `[SOURCED S2]` (FireSwarm: fires are suppressed by air assets, extinguished by humans).

**Failure behaviour:**
- *Leader lost* (missed heartbeats / `LeaderClaim` silence) → followers finish the current task, then hold; election: highest `(epoch, score, candidate_id)` wins; new leader increments epoch and re-announces.
- *Follower lost* → leader marks it `LOST`, re-queues its task, emits an `ALERT`.
- *Ground link lost* → swarm continues; ground shows drones as `LOST` until heartbeats resume.
- *RTK corrections lost* → `sensors_ok["gnss_rtk"]=False`; drop accuracy degrades (model it in sim).
- *Low endurance / water* → return/refill, never "push on".

**Messaging (QoS):** heartbeats/telemetry and map deltas are best-effort (stale is worthless); hotspot reports, task assignment/ack/complete, refill, commands and alerts are reliable. Defaults are in `DEFAULT_QOS`. Topics: `drone/{id}/…`, `swarm/…`, `ground/…`, `sim/…` (see `ROUTES`).

---

## 5. Tech stack

| Layer | Choice | Status | Notes |
|---|---|---|---|
| Language | Python 3.11+ | core | typed, `pydantic>=2.6` |
| Contracts | Pydantic v2 models, discriminated-union messages | core | validate at boundaries; arrays (NumPy) inside hot loops |
| Bus | in-process `SimBus` (with `LossyBus` wrapper: latency/loss by distance) | core | implements `Bus`; use `topic_matches()` |
| Imaging | NumPy, OpenCV, `scipy.ndimage`, scikit-image | core | keep radiometric °C until after thresholding |
| ML inference | ONNX Runtime (train in PyTorch, export `.onnx`) | core | `Detector` implementations: THRESHOLD, ML, FUSED, FAKE |
| Planning | A*/potential fields on grid; OR-Tools / `scipy.optimize.linear_sum_assignment` for allocation; Voronoi/boustrophedon for zones | core | CBBA = stretch |
| Georeferencing | `pyproj` + `rasterio` (DEM) | core (edges only) | lat/lon never inside `core/` |
| Ground base | FastAPI + WebSockets, SQLite | core | `diagnostics`-style health from heartbeats |
| Frontend | React + TypeScript + Vite; MapLibre GL + deck.gl (ops map); PixiJS/p5.js (breathing-polygons view) | core | one WebSocket state stream feeds both views |
| Debug/replay | Foxglove / Rerun, MCAP logs | optional | |
| Real-world middleware | ROS 2 (Jazzy on Ubuntu 24.04 is the safe pick; Lyrical Luth is the newest LTS `[SOURCED S15]`), PX4 + MAVSDK | stretch | adapters only |
| Tooling | `uv`, `ruff`, `pytest`, pre-commit; Docker/devcontainer for Linux-only parts | core | |

---

## 6. Drone platform and sensor baseline

### 6.1 Reference platforms (real hardware to anchor our numbers)

| Class | Example | Figures | Source |
|---|---|---|---|
| Ag spray, heavy-lift multirotor | DJI Agras T50 | 39.9 kg w/o battery, 52 kg with; max takeoff (spraying) 92 kg; 40 L tank; battery 30,000 mAh at ~51.8 V; a reseller listing quotes ~7 min hover at full spray load | S5 |
| US initial-attack drone fleets | FireSparrow-class (San Bernardino County) | ~80 lb (~10 gal) payload per one report, ~100 lb per another; round trip up to ~10 mi; one technician refills and swaps the battery "within minutes" | S6 |
| Ultra-heavy | ACC ThunderWasp (FireSwarm) | up to 400 kg or 100 gal (~379 L); turbine-powered quadcopter; cruise up to 48 kt (~24.7 m/s); endurance 60–90 min (ASCE) vs up to 2 h (CTIF) — treat as a range | S1, S2, S3 |

**Our default (`[ASSUMPTION]`, mech E to confirm):** both roles use a **T50-class airframe** — Extinguisher: 40 L water (40 kg), Seeker: same airframe, no water. Cruise 15 m/s, max 22 m/s, braking 3 m/s², **endurance 6 h** (middle of the provider's 5–8 h generator range). Generator mass/fuel is *not* modelled yet `[OPEN]`. `[OPEN]` whether to move to an ultra-heavy class (fewer, bigger drones) — it changes swarm size, drops per hotspot and refill logistics.

### 6.2 FireSwarm (Canadian, Squamish BC) — what we learned

FireSwarm does not publish its onboard sensor spec (ASCE and CTIF both omit it) `[SOURCED S2,S3]`, so the baseline below is built from component data plus analogy. What *is* public and useful:

1. **Layered detection, then suppression.** Their platform combines infrared sensing, swarm intelligence and ultra-heavy-lift drones, and is *drone-agnostic*. Detection comes from layers: an under-wing IR pod that detects and geolocates early fires **without gimbaled cameras**, AI optical/thermal imaging with ground climate-sensor mesh, a high-altitude platform, and **low-altitude VTOL overwatch that confirms early readings** `[S1]`. → Our pipeline is *cue → Seeker confirm → Extinguisher suppress → human verify*; FIRMS/ground sensors are the cue; Seekers are the close-range confirm layer. → **Use a fixed nadir camera + pose-based georeferencing; no gimbal.**
2. **Suppression drones are water haulers with light sensing**; the detection layer is separate. Supports the Seeker/Extinguisher split and a lighter Extinguisher sensor load.
3. **Water pickup is part of autonomy.** FireSwarm drones use Bambi buckets, plan their own water pickups and drops, and fly at night `[S2]`; ThunderWasp carries water from natural and human-made sources `[S1]`. → Validates the refill-basin idea (§8).
4. **Humans stay in the loop.** Air assets suppress, humans extinguish and declare fires out `[S2]`.
5. **Decoys.** The XPRIZE autonomous final gives teams 10 minutes to detect and suppress a high-risk fire in a 1,000 km² area **while leaving decoy fires untouched** `[S1]`. → `ScenarioConfig.decoys` exists; measure *water wasted on non-fires*.

### 6.3 Baseline loadout (per drone)

| Item | Seeker | Extinguisher | Mass | Power | Basis |
|---|---|---|---|---|---|
| Radiometric LWIR thermal | 640×512, ~32° HFOV (dual thermal+visible module optional) | 320×256 nadir, ~50° HFOV: aim + verify | ~25 g | ~1–1.8 W | S7: module 7.5 g bare, ~21–23 g with lens, from 0.5 W; 56 g / 1.8 W dual module; OEM modules ≈ $4.2k–$6.4k |
| 3D lidar (Livox Mid-360 class) | ✔ | ✔ | 265 g | 6.5 W | S8: typical range 40 m on a 10% reflectivity target |
| Companion computer (Jetson Orin NX 16 GB class) | ✔ detector + map | ✔ (could be lighter) | module 26 g; ~200 g with carrier + cooling `[ASSUMPTION]` | 10–25 W | S9 |
| Flight controller (Pixhawk class) | ✔ | ✔ | ~75 g `[ASSUMPTION]` | ~3 W `[ASSUMPTION]` | |
| RTK-GNSS receiver + antenna | ✔ | ✔ | ~75 g `[ASSUMPTION]` | ~1.5 W `[ASSUMPTION]` | needs corrections from the ground base |
| Downward rangefinder / radar altimeter | ✔ | ✔ | ~50 g `[ASSUMPTION]` | ~2 W `[ASSUMPTION]` | terrain following, drop height |
| Mesh radio | ✔ | ✔ | ~300 g `[ASSUMPTION]` | ~8 W `[ASSUMPTION]` | |
| Water system: tank, valve/pump, flow or level sensor, optional foam dosing | — | ✔ | payload | small | `[OPEN]` mech E (§7, §8) |
| ADS-B in (manned-aircraft awareness) | recommended | recommended | small | small | `[ASSUMPTION]` |

**Seeker avionics ≈ 1.0 kg, ≈ 48 W.** For scale: 1 kg ≈ 2.5% of a 40 kg water load; implied hover power for a ~90 kg T50-class aircraft ≈ 13 kW (1.55 kWh ÷ 7 min, a derived order-of-magnitude figure from a reseller listing, `[ASSUMPTION]`), so avionics ≈ 0.4%. **Weight is not the sensor constraint; integration complexity and justification are.** Sensors map to `SENSOR_NAMES` in `contracts/config.py`.

### 6.4 Review notes (what changed after the final review)

- **Stopping distance vs lidar range.** At 15 m/s with 3 m/s² braking, stopping distance is 37.5 m — about the full 40 m lidar range on a dark (10% reflectivity) target. At 10 m/s it is 16.7 m; at 5 m/s, 4.2 m. **Rule:** cruise *above* the tallest obstacle (DEM + canopy margin); descend only for drops/intake and slow to ≤5 m/s there. `PlatformSpec.stopping_distance_m()` encodes the math.
- **Thin obstacles (wires)** are often missed by lidar `[VERIFY]` → use mapped power lines + altitude margin.
- **Radar correction:** radar is unaffected by darkness; its limit is coarse angular resolution. Real heavy-lift ag drones pair active phased-array radar with binocular vision for obstacle avoidance `[S5]`. Dropping radar for the demo is defensible; consider a **radar altimeter** for hover over water (laser returns from calm water are weak `[VERIFY]`).
- **NIR/SWIR is optional, not baseline.** LWIR already covers night detection of smouldering hotspots; SWIR sees hot flames/embers but is costly and misses cooler hotspots `[VERIFY]`.
- **Dual thermal+visible** is a nice-to-have for the ops view and dusk/daytime pre-survey of water sources.
- **RTK dependency:** corrections come from the ground base; model `gnss_rtk` degradation.
- **Added:** water level/flow sensing (otherwise estimate from valve time × flow rate), wind estimate from the flight controller (needed for drop drift), ADS-B in, health telemetry for the ground base.
- **Gas sensors:** deferred, see §9.

### 6.5 Detection physics the sim must reproduce

- **Footprint vs altitude** (640×512 camera; ideal search rate = swath × speed at 15 m/s, no overlap or turns):

| HFOV | AGL | Swath | Ground sample distance | Ideal search rate |
|---|---|---|---|---|
| 32° | 80 m | 45.9 m | 7.2 cm/px | 2.5 km²/h |
| 32° | 100 m | 57.3 m | 9.0 cm/px | 3.1 km²/h |
| 95° | 80 m | 174.6 m | 27.3 cm/px | 9.4 km²/h |
| 95° | 100 m | 218.3 m | 34.1 cm/px | 11.8 km²/h |

  Wide lens = ~3.8× search rate but 3–4× coarser pixels. A 1–4 m² hotspot is a handful of pixels at 27 cm. `CameraModel.ground_sample_distance_m()` computes this.
- **Sub-pixel dilution:** a hotspot smaller than a pixel reads cooler than it is (mix in radiance, not °C `[VERIFY]`). The simulator must model fill fraction; detection thresholds must account for it.
- **Fixed nadir mount**, 16-bit radiometric → `float32` °C array in `ThermalFrame`. Don't convert to 8-bit before thresholding. Frame rate default 9 Hz `[ASSUMPTION]` (export-compliant variants).

---

## 7. Extinguishing agents

| Agent | Mix / facts | Claimed benefit | Fit: night spot-fire knockdown | Basin refill | Constraints |
|---|---|---|---|---|---|
| **Plain water** | — | baseline | ✔ default | ✔ no dosing needed | none |
| **Class A foam** | 0.1–1% concentrate `[S10]`; most products work best near 0.3–0.5% `[S10]` | Manufacturer: often 3–5× more effective than water `[S10]`; independent test authors call that claim controversial with limited comparative testing `[S10]` | ✔ strong: USFS says foams rely on their water and are best for direct attack and mop-up `[S10]` | ✔ **only if dosing happens after intake** (onboard proportioner). 40 L × 0.3% = 120 mL concentrate per fill; a 2 L reservoir ≈ 16 fills | US guideline: no aerial foam/retardant within 300 ft (91.44 m) of waterways `[S11]`; Canadian/Alberta rule `[VERIFY]` |
| **Gel / water enhancer** | mix varies; sensitive to water hardness `[S10]` | adhesion, drop accuracy | weak (structure protection more than hotspot knockdown) | ⚠ basin water quality changes consistency | same waterway buffers |
| **Long-term retardant** (ammonium phosphate) | — | stays effective after water evaporates | ✘ fireline building, not spot knockdown | ✘ | toxic to aquatic life `[S11]`; waterway buffers |

**Decisions (`[ASSUMPTION]`):**
- Baseline `ExtinguishingAgent.WATER`; optional `CLASS_A_FOAM` module (needs dosing, sim multiplier 3× as an *assumption*, not a promise); gel and retardant are out of scope (retardant enum deliberately omitted).
- **Basin refilling stays enabled for foam** via onboard dosing: the tank holds plain water until dosing after intake, so foam never contaminates a basin and gel/foam quality isn't hostage to basin water.
- **Waterway rule as an algorithm:** if a hotspot is within `AgentSpec.aquatic_buffer_m` of a water source, fall back to plain water. (Sim hook: `agent_allowed(hotspot, water_sources)`.)
- Sim settings: `ScenarioConfig.agent`, `agent_specs`, `use_water_sources`. `suppression_l_per_m2` is a **PLACEHOLDER (10 L/m²)**: E must derive it (fuel type, burn depth). With a 3× agent, a 3 m² hotspot at 10 L/m² needs 30 L of water or ~10 L of foam solution.

---

## 8. Refill logistics and the water-basin idea

**Idea.** Seekers (and/or prior hydrography data) identify natural water bodies (ponds, lakes, rivers) near the fire. Extinguishers refill there instead of flying back to the primary station, shortening the refill leg and raising water delivered per drone-hour. A primary refill station (possibly mobile) remains the guaranteed option. The swarm treats refill points as a *planning problem*: choose the best refill target per sortie.

**Reality check.** FireSwarm drones plan their own water pickups with buckets and carry water from natural and human-made sources `[S1,S2]`; US initial-attack drone fleets refill at a base station by a technician "within minutes" `[S6]`; amphibious drones are pitched for night firefighting precisely because they can refill from lakes `[S16]`.

**Throughput model** (one extinguisher; cruise 15 m/s, 40 L tank, fill 120 s, drop 8 s; `d` = refill-point-to-fire distance):
`cycle = 2d/v + t_fill + t_drop` · `throughput = W / cycle`

| Refill point | Cycle | Throughput |
|---|---|---|
| Station 15 km away | 35.5 min | 1.13 L/min (68 L/h) |
| Station 5 km away | 13.2 min | 3.02 L/min (181 L/h) |
| Basin 1 km away | 4.4 min | 9.18 L/min (551 L/h) |
| Basin 0.5 km away | 3.2 min | 12.33 L/min (740 L/h) |

A basin 1 km from the fire beats a station 15 km away by **~8×**. A 3 min queue at a 5 km station drops it to ~2.5 L/min. **Depletion is negligible:** 100 sorties × 40 L = 4 m³ ≈ 1% of a 20 m × 20 m × 1 m pond; still track `est_volume_l` for tiny sources.

**Discovery pipeline.** Prior hydrography → candidate `WaterSource(origin=PRIOR_MAP, status=CANDIDATE)` → a Seeker flies `SURVEY_WATER` → `VALIDATED` (or `REJECTED`) → leader includes it in refill options → `RefillRequest`/`RefillGrant`. Prior water maps (provincial/national hydrography, OpenStreetMap) mean seekers mostly *validate*, not discover `[VERIFY availability/licence]`. Candidate night cues, all `[VERIFY]` with real data: **LWIR:** water often looks warmer than surrounding land at night; **lidar:** calm water returns weakly (holes in the cloud); **DEM:** flat depressions. Also check **approach clearance** (trees around the shore, corridor aligned with wind) → `approach_clearance_m`.

**Optimization problems (in order of effort):**
1. Per-sortie refill choice: minimize `travel + queue + fill + risk penalty (unvalidated)` over stations and validated sources.
2. Joint extinguisher↔hotspot↔refill allocation (a vehicle-routing problem with intermediate facilities): greedy first, OR-Tools later.
3. Mobile station placement (p-median style) given FIRMS cues / hotspot clusters.
4. Water-survey scheduling: survey candidates near predicted fire zones first (value of information).
5. Station queue/bay management.

**Risks / constraints:** foam contamination (see §7); gel quality vs water hardness; rotor wash on shallow ponds; frozen/dry/seasonal sources; legality of withdrawing water and ecology `[VERIFY Alberta]`; false "water" (wet fields); night hover over water (lidar altitude unreliable → RTK/baro + radar altimeter); **intake method is `[OPEN]` for E**: dip bucket (Bambi-style, as FireSwarm) vs hover with hose + pump vs other. Humans may need to approve new sources → `GroundCommand(ENABLE_WATER_SOURCE/DISABLE_WATER_SOURCE)`.

**Contract hooks:** `WaterSource`, `RefillStation`, `TaskType.SURVEY_WATER` / `REFILL`, `WaterSourceReport`, `RefillRequest`, `RefillGrant`, `MapSync.water_sources`, `ScenarioConfig.use_water_sources` (feature flag), `AlertKind.WATER_SOURCE_NEW`.

**Metrics:** litres delivered per drone-hour, time to first drop per hotspot, fraction of time in transit vs on target, refill queue wait, water wasted on decoys.

**Build plan:** MVP = station + prior-map sources (flag on/off for the A/B comparison) · Stretch 1 = `SURVEY_WATER` validation with synthetic thermal water blobs · Stretch 2 = joint optimizer.

---

## 9. Byproduct (gas/smoke) sensing — verdict: **deferred stretch, not inefficient**

- **What the evidence says.** Ground networks use gas/particle sensors for ultra-early detection of *smouldering*: Dryad's Silvanet fuses VOC, CO and particle sensors with built-in AI to separate a wildfire signature from other sources `[S12]`; surveys list MOX/electrochemical sensors for TVOC and CO₂ and optical scattering for PM `[S12]`. A research octocopter carried air-quality sensors (CO₂, particulates, a VOC sampler) but its authors stress that no single system suffices — a suite is needed `[S12]`.
- **Where it helps us:** hotspots under canopy that thermal can't see; corroborating or rejecting a thermal detection (a warm rock makes no CO — useful against decoys).
- **Limits on a moving drone `[VERIFY]`:** rotor wash dilutes the sample; electrochemical sensors lag by tens of seconds; plumes advect with wind (needs a wind estimate, which we need for drop drift anyway); a spot-fire plume is small; CO₂ varies against a ~420 ppm background, so CO and PM are better signals.
- **Plan:** Phase-4 stretch. Add an optional `gas` sensor (CO electrochemical + PM2.5, maybe VOC; tens of grams `[ASSUMPTION]`) and a **Gaussian-plume model + wind** in the sim (physics, not ML). Uses: upwind search prior (cast-and-surge) when thermal sees nothing; extra evidence in `Hotspot.confidence`. Not needed for the F1 target.

---

## 10. Data and ML

- **Imagery for the detector:** FLAME 3 provides paired visible + **radiometric thermal TIFF** from prescribed burns; the full 6-burn set is on request and a single-burn CV subset is on Kaggle `[S13]`. FLAME 1/2 are **not radiometric** (palette imagery) `[S13]` — so earlier suggestions to "use FLAME for radiometric" should read **FLAME 3**. The FLAME 3 authors note radiometric UAV wildfire imagery is underused because public datasets are scarce `[S13]`.
- **Split by burn/flight, not by frame** (neighbouring frames leak). Add a held-out source and a **sim-generated** test set. Report detector F1 *and* swarm metrics separately.
- **FIRMS / CWFIS / Alberta historical** are *not* imagery: they parameterize and validate the scenario generator (where fires start, night behaviour, size at detection, cues).
- **What we likely cannot train with real data (`[VERIFY]` — to our knowledge):**
  1. gas/plume time-series near small fires (Dryad trained on its own data);
  2. lidar/radar in smoke and burning terrain;
  3. SWIR flame/ember imagery;
  4. radiometric night imagery of *small spot fires* at altitude (FLAME 3 covers prescribed burns, which the authors say differ from intense wildfire behaviour `[S13]`);
  5. night thermal signatures of water bodies;
  6. decoy discrimination;
  7. swarm-level behaviour (no dataset exists; use the sim).
- **Approach for those:** physics/classical methods; **anomaly detection against background** (background-only data is plentiful, positives are not); synthetic data from the sim with domain randomization; ML only where data exists (thermal blob classification/verification).

---

## 11. Contracts

**Install/test:** `pip install -e .[dev]` · `python -m pytest -q` (49 tests) · `python tools/sync_contracts.py` refreshes the code below.

**How to use:** build messages with `SwarmMessage.create(sender_id, body, now, seq=…, receiver_id=…)`; publish on `topic_for(msg)`; serialize with `encode()`, parse with `decode()`; match subscriptions with `topic_matches()`; implement `Bus`/`Clock`/`Autopilot`/`PayloadSystem`/`Detector`/`Georeferencer` from `interfaces.py`.

**Differences from the class diagram:**
- Timestamps are **float seconds** (Clock timebase), not `datetime`.
- `SwarmMessage.payload: dict` → a **discriminated union** of typed bodies.
- New: `WaterSource`, `RefillStation` (now core), `TaskType.SURVEY_WATER`, `TaskAnnounce`/`TaskAck`, `LeaderClaim`, `RefillRequest`/`RefillGrant`, `GroundCommand`, `AlertNotice`, `SimFault`, `ScenarioConfig` and platform/agent specs.
- `LOW_BATTERY` became a status/health field plus a derived `ALERT`; `PATH_CLAIM` kept.
- `ImagingMap` is **not** a Pydantic model: it is NumPy layers (log-odds, last-seen). Its wire format is sparse `GridCellUpdate` inside `MapSync`.
- `Sensor.read()` is replaced by `ThermalSensor` / `ObstacleSensor` protocols; frames stay on the drone.
- `HotspotStatus.VERIFIED_OUT` is ground-confirmed only.

<!-- CONTRACTS:BEGIN -->

#### `contracts/__init__.py` (16 lines)

```python
"""natHacks 2026 (Pandas) - shared contracts for the night initial-attack drone swarm.

Single source of truth for message shapes, entities, topics, interfaces and scenario config.
Depends only on pydantic. See context.md for conventions and how to change a contract.
"""
from .base import CONTRACT_VERSION, Contract, FrozenContract, Seconds
from .config import *  # noqa: F401,F403
from .entities import *  # noqa: F401,F403
from .enums import *  # noqa: F401,F403
from .geometry import *  # noqa: F401,F403
from .interfaces import *  # noqa: F401,F403
from .messages import *  # noqa: F401,F403
from .sensing import *  # noqa: F401,F403
from .topics import *  # noqa: F401,F403

__version__ = CONTRACT_VERSION
```

#### `contracts/base.py` (34 lines)

```python
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
CONTRACT_VERSION = "0.1.0"

# All timestamps/durations are float SECONDS in the Clock's timebase
# (sim time in the simulator, wall-clock epoch seconds on real hardware).
Seconds = float


class Contract(BaseModel):
    """Mutable contract model. Validates on assignment too."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FrozenContract(BaseModel):
    """Immutable, hashable contract model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
```

#### `contracts/enums.py` (162 lines)

```python
"""Enumerations shared by every workstream.

Values are uppercase strings equal to the member name so JSON stays readable.
"""
from enum import StrEnum

__all__ = [
    "DroneKind", "SwarmRole", "DroneStatus",
    "TaskType", "TaskStatus", "HotspotStatus",
    "WaterSourceKind", "WaterSourceStatus", "WaterSourceOrigin", "RefillTargetKind",
    "ExtinguishingAgent", "DetectorKind",
    "MessageType", "Severity", "AlertKind", "CommandKind", "SimFaultKind",
]


class DroneKind(StrEnum):
    SEEKER = "SEEKER"              # scout: thermal + lidar, finds/confirms hotspots, surveys water
    EXTINGUISHER = "EXTINGUISHER"  # carries water/agent, drops on confirmed hotspots


class SwarmRole(StrEnum):
    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"  # election in progress
    LEADER = "LEADER"        # runs the SwarmCoordinator role; any drone can hold it


class DroneStatus(StrEnum):
    IDLE = "IDLE"
    TRANSIT = "TRANSIT"
    SCANNING = "SCANNING"
    EXTINGUISHING = "EXTINGUISHING"
    RETURNING = "RETURNING"
    REFILLING = "REFILLING"
    FAULT = "FAULT"  # self-reported
    LOST = "LOST"    # assigned by OBSERVERS (leader/ground) after missed heartbeats


class TaskType(StrEnum):
    SCAN_ZONE = "SCAN_ZONE"
    INVESTIGATE = "INVESTIGATE"      # close-range confirm of a candidate hotspot
    EXTINGUISH = "EXTINGUISH"
    VERIFY = "VERIFY"                # re-image after a drop; sets SUPPRESSED if cold
    REFILL = "REFILL"
    SURVEY_WATER = "SURVEY_WATER"    # validate a candidate water source (or search a zone for one)


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class HotspotStatus(StrEnum):
    DETECTED = "DETECTED"            # seen once, unconfirmed
    CONFIRMED = "CONFIRMED"          # confirmed by close-range investigation
    ASSIGNED = "ASSIGNED"            # an extinguisher is tasked
    SUPPRESSED = "SUPPRESSED"        # drone VERIFY pass found it cold
    VERIFIED_OUT = "VERIFIED_OUT"    # GROUND-CONFIRMED out (humans declare fires out; see GroundCommand.CONFIRM_OUT)
    FALSE_POSITIVE = "FALSE_POSITIVE"


class WaterSourceKind(StrEnum):
    LAKE = "LAKE"
    POND = "POND"
    RIVER = "RIVER"
    RESERVOIR = "RESERVOIR"
    TANK = "TANK"
    OTHER = "OTHER"


class WaterSourceStatus(StrEnum):
    CANDIDATE = "CANDIDATE"  # from prior map or detection, not yet validated by a seeker
    VALIDATED = "VALIDATED"  # approach clearance + presence of water confirmed
    REJECTED = "REJECTED"    # dry / blocked / unsafe / disabled by ground


class WaterSourceOrigin(StrEnum):
    PRIOR_MAP = "PRIOR_MAP"  # imported hydrography data
    DETECTED = "DETECTED"    # found by a seeker's sensors
    MANUAL = "MANUAL"        # entered by ground personnel


class RefillTargetKind(StrEnum):
    STATION = "STATION"
    WATER_SOURCE = "WATER_SOURCE"


class ExtinguishingAgent(StrEnum):
    WATER = "WATER"
    CLASS_A_FOAM = "CLASS_A_FOAM"
    GEL = "GEL"  # water enhancer / gel
    # RETARDANT intentionally omitted: long-term retardant is for fireline building, not spot-fire knockdown.


class DetectorKind(StrEnum):
    THRESHOLD = "THRESHOLD"  # classical: temperature threshold + connected components
    ML = "ML"                # ONNX model
    FUSED = "FUSED"          # mixed approach: classical proposals verified by ML (or vice versa)
    FAKE = "FAKE"            # simulator ground truth + tunable misses/false alarms


class MessageType(StrEnum):
    HEARTBEAT = "HEARTBEAT"
    HOTSPOT_REPORT = "HOTSPOT_REPORT"
    WATER_SOURCE_REPORT = "WATER_SOURCE_REPORT"
    MAP_SYNC = "MAP_SYNC"
    TASK_ANNOUNCE = "TASK_ANNOUNCE"
    TASK_BID = "TASK_BID"
    TASK_ASSIGN = "TASK_ASSIGN"
    TASK_ACK = "TASK_ACK"
    TASK_COMPLETE = "TASK_COMPLETE"
    PATH_CLAIM = "PATH_CLAIM"
    LEADER_CLAIM = "LEADER_CLAIM"
    REFILL_REQUEST = "REFILL_REQUEST"
    REFILL_GRANT = "REFILL_GRANT"
    GROUND_COMMAND = "GROUND_COMMAND"
    ALERT = "ALERT"
    SIM_FAULT = "SIM_FAULT"  # simulator-only fault injection ("chaos buttons")


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertKind(StrEnum):
    DRONE_LOST = "DRONE_LOST"
    LEADER_LOST = "LEADER_LOST"
    LEADER_ELECTED = "LEADER_ELECTED"
    LOW_BATTERY = "LOW_BATTERY"
    SENSOR_FAULT = "SENSOR_FAULT"
    COMMS_DEGRADED = "COMMS_DEGRADED"
    HOTSPOT_NEW = "HOTSPOT_NEW"
    HOTSPOT_SUPPRESSED = "HOTSPOT_SUPPRESSED"
    WATER_SOURCE_NEW = "WATER_SOURCE_NEW"
    GENERIC = "GENERIC"


class CommandKind(StrEnum):
    RETURN_TO_BASE = "RETURN_TO_BASE"
    HOLD = "HOLD"
    RESUME = "RESUME"
    ABORT_TASK = "ABORT_TASK"
    SET_PRIORITY = "SET_PRIORITY"                # params: {"priority": 0..1}; target_id = hotspot_id
    CONFIRM_OUT = "CONFIRM_OUT"                  # human declares hotspot out -> VERIFIED_OUT
    MARK_FALSE_POSITIVE = "MARK_FALSE_POSITIVE"  # human rejects a hotspot
    ENABLE_WATER_SOURCE = "ENABLE_WATER_SOURCE"
    DISABLE_WATER_SOURCE = "DISABLE_WATER_SOURCE"


class SimFaultKind(StrEnum):
    KILL_DRONE = "KILL_DRONE"
    KILL_LEADER = "KILL_LEADER"
    DROP_LINK = "DROP_LINK"                # temporarily cut a drone's radio
    DEGRADE_SENSOR = "DEGRADE_SENSOR"      # magnitude 0..1; target sensor in `target_id` as "drone:sensor"
    DRAIN_BATTERY = "DRAIN_BATTERY"
    INJECT_FALSE_POSITIVE = "INJECT_FALSE_POSITIVE"
    CLEAR = "CLEAR"                        # clear all injected faults
```

#### `contracts/geometry.py` (110 lines)

```python
"""Coordinates, poses and grids.

CONVENTIONS - every workstream must follow these:
  * Local frame is ENU in meters: x = east, y = north, z = up.
  * Origin is ScenarioConfig.origin (a GeoOrigin). lat/lon exist only at the edges
    (data import, ground-base map display); never inside the swarm core.
  * z is height above the scenario datum, NOT above ground (AGL).
  * Headings/yaw are compass degrees clockwise from north (0 = N, 90 = E).
  * Grids are indexed [row, col]; row 0 is the SOUTH edge (flip when plotting).
  * PX4 uses NED/FRD axes: convert inside the PX4 adapter only.
"""
from __future__ import annotations

import math
from typing import Annotated

from pydantic import Field

from .base import FrozenContract

__all__ = ["Meters", "Position", "Pose", "GeoOrigin", "GridIndex", "GridSpec", "point_in_polygon"]

Meters = Annotated[float, Field(allow_inf_nan=False)]


class Position(FrozenContract):
    x: Meters  # meters east of scenario origin
    y: Meters  # meters north of scenario origin
    z: Meters = 0.0  # meters above scenario datum (NOT AGL)

    def distance_to(self, other: "Position", horizontal: bool = False) -> float:
        dz = 0.0 if horizontal else other.z - self.z
        return math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2 + dz ** 2)

    def offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Position":
        return Position(x=self.x + dx, y=self.y + dy, z=self.z + dz)


class Pose(FrozenContract):
    position: Position
    yaw_deg: float = 0.0    # compass heading, clockwise from north
    pitch_deg: float = 0.0  # positive = nose up
    roll_deg: float = 0.0   # positive = right side down


class GeoOrigin(FrozenContract):
    """Where local (0, 0) sits on Earth. Used only at the edges (data import, map display)."""

    lat_deg: float = Field(ge=-90, le=90)
    lon_deg: float = Field(ge=-180, le=180)
    utm_epsg: int  # e.g. 32612 = WGS84 / UTM zone 12N
    datum_elevation_m: float = 0.0


class GridIndex(FrozenContract):
    row: int = Field(ge=0)  # 0 = southern edge
    col: int = Field(ge=0)  # 0 = western edge


class GridSpec(FrozenContract):
    """Shared definition of the map grid. Everyone indexes numpy arrays as [row, col]."""

    origin: Position  # south-west corner of cell (0, 0)
    resolution_m: float = Field(gt=0)
    rows: int = Field(gt=0)  # extent along +y (north)
    cols: int = Field(gt=0)  # extent along +x (east)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def width_m(self) -> float:
        return self.cols * self.resolution_m

    @property
    def height_m(self) -> float:
        return self.rows * self.resolution_m

    def world_to_cell(self, pos: Position) -> GridIndex | None:
        """Cell containing `pos`, or None if outside the grid."""
        col = math.floor((pos.x - self.origin.x) / self.resolution_m)
        row = math.floor((pos.y - self.origin.y) / self.resolution_m)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return GridIndex(row=row, col=col)
        return None

    def cell_to_world(self, idx: GridIndex) -> Position:
        """Center of the cell (z = grid origin z)."""
        return Position(
            x=self.origin.x + (idx.col + 0.5) * self.resolution_m,
            y=self.origin.y + (idx.row + 0.5) * self.resolution_m,
            z=self.origin.z,
        )

    def contains(self, pos: Position) -> bool:
        return self.world_to_cell(pos) is not None


def point_in_polygon(x: float, y: float, polygon: list[Position]) -> bool:
    """Ray-casting point-in-polygon test on the x/y plane."""
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside
```

#### `contracts/sensing.py` (81 lines)

```python
"""Perception-facing types: camera model, thermal frames, detections, obstacles.

PRINCIPLE: raw frames and point clouds NEVER travel on the bus. They stay on the drone.
Only small results (Detection -> Hotspot, Obstacle) leave the perception layer.
ThermalFrame is therefore a plain dataclass holding a numpy array, not a Pydantic model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field

from .base import FrozenContract
from .enums import DetectorKind
from .geometry import Pose, Position

if TYPE_CHECKING:  # contracts depend only on pydantic at runtime
    import numpy as np

__all__ = ["CameraModel", "ThermalFrame", "Detection", "Obstacle"]


class CameraModel(FrozenContract):
    """Pinhole camera. Footprint math below assumes a nadir (straight-down) mount."""

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    hfov_deg: float = Field(gt=0, lt=180)
    mount_pitch_deg: float = Field(default=-90.0, ge=-90, le=90)  # -90 = looking straight down

    @property
    def vfov_deg(self) -> float:
        half_h = math.tan(math.radians(self.hfov_deg) / 2)
        return math.degrees(2 * math.atan(half_h * self.height_px / self.width_px))

    def swath_m(self, agl_m: float) -> float:
        """Ground width covered by the horizontal FOV at height `agl_m` (nadir)."""
        return 2 * agl_m * math.tan(math.radians(self.hfov_deg) / 2)

    def ground_sample_distance_m(self, agl_m: float) -> float:
        """Meters of ground per pixel at height `agl_m` (nadir)."""
        return self.swath_m(agl_m) / self.width_px


@dataclass
class ThermalFrame:
    """One radiometric thermal image. NOT serialized onto the bus.

    temp_c: float32 array, shape (height_px, width_px), temperature in degrees Celsius.
            Keep per-pixel temperatures; do not convert to 8-bit before thresholding.
    """

    drone_id: str
    timestamp: float
    temp_c: "np.ndarray"
    pose: Pose
    camera: CameraModel


class Detection(FrozenContract):
    """Output of a Detector, in IMAGE coordinates. A Georeferencer turns it into a Hotspot position."""

    u: float  # centroid column (pixels)
    v: float  # centroid row (pixels)
    area_px: float = Field(ge=0)
    peak_temp_c: float
    mean_temp_c: float | None = None
    confidence: float = Field(ge=0, le=1)
    source: DetectorKind
    bbox: tuple[float, float, float, float] | None = None  # (u0, v0, u1, v1)


class Obstacle(FrozenContract):
    """Obstacle reported by an obstacle sensor (lidar/radar) or the simulator."""

    position: Position
    radius_m: float = Field(gt=0)
    height_m: float | None = Field(default=None, ge=0)
    kind: str = "unknown"  # e.g. "tree", "pole", "terrain"
```

#### `contracts/entities.py` (171 lines)

```python
"""Domain entities: the nouns of the system.

Entities are mutable (they change state over a mission). Anything sent over the bus is
wrapped in a frozen message body (see messages.py) and serialized, so receivers never
share Python objects with senders.

Design notes
  * Hotspot.priority_score is COMPUTED BY THE LEADER (the formula lives in core/, not here).
  * HotspotStatus.VERIFIED_OUT is set only via a ground-confirmed command: humans declare fires out.
  * Map cells use log-odds (see GridCellUpdate); convert with p = 1 / (1 + exp(-log_odds)).
"""
from __future__ import annotations

from pydantic import Field, model_validator

from .base import Contract, FrozenContract
from .enums import (
    AlertKind, DroneKind, DroneStatus, ExtinguishingAgent, HotspotStatus, Severity, SwarmRole,
    TaskStatus, TaskType, WaterSourceKind, WaterSourceOrigin, WaterSourceStatus,
)
from .geometry import Position, point_in_polygon

__all__ = [
    "DroneState", "Hotspot", "Zone", "Task", "WaterSource", "RefillStation",
    "GridCellUpdate", "Alert",
]


class DroneState(Contract):
    """Telemetry snapshot. Carried inside HEARTBEAT; the ground base builds its health view from it."""

    drone_id: str
    kind: DroneKind
    role: SwarmRole = SwarmRole.FOLLOWER
    position: Position
    heading_deg: float = Field(default=0.0, ge=0, lt=360)
    speed_mps: float = Field(default=0.0, ge=0)
    battery_pct: float = Field(ge=0, le=100)
    endurance_remaining_s: float | None = Field(default=None, ge=0)  # generator + battery estimate
    water_remaining_l: float | None = Field(default=None, ge=0)      # EXTINGUISHER only
    agent: ExtinguishingAgent | None = None                          # what the tank currently holds
    status: DroneStatus = DroneStatus.IDLE
    current_task_id: str | None = None
    leader_id: str | None = None   # who this drone currently believes is leader
    epoch: int = Field(default=0, ge=0)  # leader epoch this drone believes in
    # Health flags keyed by sensor/subsystem name, e.g. {"thermal": True, "lidar": True,
    # "gnss_rtk": False, "comms": True, "payload": True}. Names are defined in config.py (SENSOR_NAMES).
    sensors_ok: dict[str, bool] = Field(default_factory=dict)
    link_quality: float | None = Field(default=None, ge=0, le=1)
    timestamp: float


class Hotspot(Contract):
    hotspot_id: str
    position: Position
    peak_temp_c: float = Field(ge=-50, le=1500)
    est_area_m2: float = Field(default=0.0, ge=0)
    confidence: float = Field(ge=0, le=1)
    status: HotspotStatus = HotspotStatus.DETECTED
    priority_score: float = Field(default=0.0, ge=0)  # set by the leader
    observations: int = Field(default=1, ge=1)        # how many detections fused into this hotspot
    growth_rate_m2_per_min: float | None = None       # from tracking; None until estimated
    assigned_to: str | None = None
    reported_by: str | None = None
    first_seen: float
    last_seen: float

    @model_validator(mode="after")
    def _times_ordered(self) -> "Hotspot":
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must be >= first_seen")
        return self


class Zone(Contract):
    """A search region assigned to a seeker."""

    zone_id: str
    boundary: list[Position] = Field(min_length=3)  # polygon in local x/y
    assigned_to: str | None = None
    last_scanned: float | None = None

    def contains(self, pos: Position) -> bool:
        return point_in_polygon(pos.x, pos.y, self.boundary)


class Task(Contract):
    task_id: str
    type: TaskType
    priority: float = Field(default=0.5, ge=0, le=1)  # 1 = most urgent
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str | None = None
    target_hotspot_id: str | None = None
    target_zone_id: str | None = None
    target_water_source_id: str | None = None
    target_station_id: str | None = None
    created_at: float
    deadline: float | None = None
    epoch: int = Field(default=0, ge=0)  # leader epoch that issued/assigned it

    @model_validator(mode="after")
    def _has_required_target(self) -> "Task":
        t = self.type
        if t is TaskType.SCAN_ZONE and not self.target_zone_id:
            raise ValueError("SCAN_ZONE requires target_zone_id")
        if t in (TaskType.INVESTIGATE, TaskType.EXTINGUISH, TaskType.VERIFY) and not self.target_hotspot_id:
            raise ValueError(f"{t} requires target_hotspot_id")
        if t is TaskType.REFILL and not (self.target_station_id or self.target_water_source_id):
            raise ValueError("REFILL requires target_station_id or target_water_source_id")
        if t is TaskType.SURVEY_WATER and not (self.target_water_source_id or self.target_zone_id):
            raise ValueError("SURVEY_WATER requires target_water_source_id or target_zone_id")
        return self


class WaterSource(Contract):
    """A natural or man-made water body usable for refilling (pond, lake, river, tank)."""

    source_id: str
    position: Position  # intake point (where a drone would hover)
    kind: WaterSourceKind
    origin: WaterSourceOrigin
    status: WaterSourceStatus = WaterSourceStatus.CANDIDATE
    est_volume_l: float | None = Field(default=None, ge=0)
    surface_area_m2: float | None = Field(default=None, ge=0)
    approach_clearance_m: float | None = Field(default=None, ge=0)  # free radius around the intake point
    confidence: float = Field(default=0.5, ge=0, le=1)
    reported_by: str | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0


class RefillStation(Contract):
    """Primary (powered, possibly mobile) refill point with bays and an optional dosing system."""

    station_id: str
    position: Position
    water_available_l: float = Field(ge=0)
    fill_rate_lpm: float = Field(gt=0)
    total_bays: int = Field(default=1, ge=1)
    free_bays: int = Field(default=1, ge=0)
    agents_available: list[ExtinguishingAgent] = Field(
        default_factory=lambda: [ExtinguishingAgent.WATER]
    )
    is_mobile: bool = False

    @model_validator(mode="after")
    def _bays_consistent(self) -> "RefillStation":
        if self.free_bays > self.total_bays:
            raise ValueError("free_bays cannot exceed total_bays")
        return self


class GridCellUpdate(FrozenContract):
    """Sparse map update. log_odds = ln(p/(1-p)) of 'fire present'; 0 means unknown (p = 0.5)."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    log_odds: float = Field(ge=-10, le=10)
    last_seen: float


class Alert(Contract):
    """Human-facing event for the ground base UI."""

    alert_id: str
    severity: Severity
    kind: AlertKind
    subject_id: str | None = None  # drone/hotspot/source the alert is about
    message: str
    position: Position | None = None
    timestamp: float
```

#### `contracts/messages.py` (297 lines)

```python
"""Bus messages: one frozen body model per MessageType, joined by a discriminated union.

The `type` field inside each body is the discriminator: Pydantic reads it and picks the
right model, so a malformed HEARTBEAT is rejected at the boundary instead of crashing a
handler later. Use `SwarmMessage.create(...)` to build messages and `encode`/`decode` at
every bus boundary (even in-process: it guarantees drones never share Python objects).

Ordering/identity rules
  * Each sender keeps its own increasing `seq`; msg_id defaults to f"{sender_id}-{seq}" (deterministic).
  * Commands from a leader carry `epoch`. Receivers IGNORE commands whose epoch is lower than the
    highest epoch they have seen (prevents a stale leader from acting after a failover).
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from .base import CONTRACT_VERSION, FrozenContract
from .entities import Alert, DroneState, GridCellUpdate, Hotspot, Task, WaterSource
from .enums import (
    CommandKind, ExtinguishingAgent, HotspotStatus, MessageType, RefillTargetKind, SimFaultKind,
)
from .geometry import GridSpec, Position

__all__ = [
    "Heartbeat", "HotspotReport", "WaterSourceReport", "MapSync", "TaskAnnounce", "TaskBid",
    "TaskAssign", "TaskAck", "TaskComplete", "PathClaim", "LeaderClaim", "RefillRequest",
    "RefillGrant", "GroundCommand", "AlertNotice", "SimFault",
    "Body", "BODY_TYPES", "QoSProfile", "DEFAULT_QOS", "SwarmMessage", "encode", "decode",
]


class _Body(FrozenContract):
    """Base for all message bodies. Subclasses add a Literal `type` discriminator."""


class Heartbeat(_Body):
    type: Literal[MessageType.HEARTBEAT] = MessageType.HEARTBEAT
    state: DroneState


class HotspotReport(_Body):
    type: Literal[MessageType.HOTSPOT_REPORT] = MessageType.HOTSPOT_REPORT
    hotspot: Hotspot


class WaterSourceReport(_Body):
    type: Literal[MessageType.WATER_SOURCE_REPORT] = MessageType.WATER_SOURCE_REPORT
    source: WaterSource


class MapSync(_Body):
    """Sparse map delta (or full sync) plus the entity lists. Frames are never sent; this is the shared picture."""

    type: Literal[MessageType.MAP_SYNC] = MessageType.MAP_SYNC
    grid: GridSpec
    seq: int = Field(ge=0)
    full: bool = False
    cells: list[GridCellUpdate] = Field(default_factory=list)
    hotspots: list[Hotspot] = Field(default_factory=list)
    water_sources: list[WaterSource] = Field(default_factory=list)


class TaskAnnounce(_Body):
    """Leader offers a task for bidding (auction). Bids must arrive before bid_deadline."""

    type: Literal[MessageType.TASK_ANNOUNCE] = MessageType.TASK_ANNOUNCE
    task: Task
    bid_deadline: float


class TaskBid(_Body):
    type: Literal[MessageType.TASK_BID] = MessageType.TASK_BID
    task_id: str
    bidder_id: str
    cost: float = Field(ge=0)  # lower is better; define the formula in core/, keep units consistent
    feasible: bool = True      # False = "I cannot do this" (battery, water, range, agent mismatch)
    eta_s: float | None = Field(default=None, ge=0)
    energy_cost_pct: float | None = Field(default=None, ge=0, le=100)


class TaskAssign(_Body):
    """Leader -> specific drone (set receiver_id). Carries the leader epoch."""

    type: Literal[MessageType.TASK_ASSIGN] = MessageType.TASK_ASSIGN
    task: Task
    epoch: int = Field(ge=0)


class TaskAck(_Body):
    type: Literal[MessageType.TASK_ACK] = MessageType.TASK_ACK
    task_id: str
    drone_id: str
    accepted: bool
    reason: str | None = None
    epoch: int = Field(ge=0)


class TaskComplete(_Body):
    type: Literal[MessageType.TASK_COMPLETE] = MessageType.TASK_COMPLETE
    task_id: str
    drone_id: str
    success: bool
    result_status: HotspotStatus | None = None  # e.g. SUPPRESSED after a VERIFY, CONFIRMED after INVESTIGATE
    water_used_l: float | None = Field(default=None, ge=0)
    notes: str | None = None


class PathClaim(_Body):
    """Reserve a corridor/time window so others deconflict. Altitude band optional."""

    type: Literal[MessageType.PATH_CLAIM] = MessageType.PATH_CLAIM
    drone_id: str
    waypoints: list[Position] = Field(min_length=1)
    valid_from: float
    valid_until: float
    altitude_band_m: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _window_ordered(self) -> "PathClaim":
        if self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be > valid_from")
        return self


class LeaderClaim(_Body):
    """Election: highest (epoch, score, candidate_id) wins. Re-sent as the leader's own heartbeat of authority."""

    type: Literal[MessageType.LEADER_CLAIM] = MessageType.LEADER_CLAIM
    candidate_id: str
    epoch: int = Field(ge=0)
    score: float  # e.g. health + endurance + link quality; tie-break on candidate_id


class RefillRequest(_Body):
    type: Literal[MessageType.REFILL_REQUEST] = MessageType.REFILL_REQUEST
    drone_id: str
    position: Position
    water_remaining_l: float = Field(ge=0)
    agent: ExtinguishingAgent = ExtinguishingAgent.WATER


class RefillGrant(_Body):
    """Leader -> specific drone (set receiver_id): where to refill, which bay, how long the queue is."""

    type: Literal[MessageType.REFILL_GRANT] = MessageType.REFILL_GRANT
    drone_id: str
    target_kind: RefillTargetKind
    target_id: str
    position: Position
    bay: int | None = Field(default=None, ge=0)
    eta_s: float | None = Field(default=None, ge=0)
    queue_wait_s: float = Field(default=0.0, ge=0)
    epoch: int = Field(default=0, ge=0)


class GroundCommand(_Body):
    """Human/ground-base command. `target_id` is a drone_id, hotspot_id or source_id depending on `command`."""

    type: Literal[MessageType.GROUND_COMMAND] = MessageType.GROUND_COMMAND
    command: CommandKind
    target_id: str | None = None
    params: dict[str, float | str | bool] = Field(default_factory=dict)
    issued_by: str = "ground"


class AlertNotice(_Body):
    type: Literal[MessageType.ALERT] = MessageType.ALERT
    alert: Alert


class SimFault(_Body):
    """SIMULATOR ONLY. Fault injection for demos and resilience tests; real drones must ignore it."""

    type: Literal[MessageType.SIM_FAULT] = MessageType.SIM_FAULT
    fault: SimFaultKind
    target_id: str | None = None
    duration_s: float | None = Field(default=None, gt=0)
    magnitude: float | None = None


Body = Annotated[
    Union[
        Heartbeat, HotspotReport, WaterSourceReport, MapSync, TaskAnnounce, TaskBid, TaskAssign,
        TaskAck, TaskComplete, PathClaim, LeaderClaim, RefillRequest, RefillGrant, GroundCommand,
        AlertNotice, SimFault,
    ],
    Field(discriminator="type"),
]

BODY_TYPES: dict[MessageType, type[_Body]] = {
    MessageType.HEARTBEAT: Heartbeat,
    MessageType.HOTSPOT_REPORT: HotspotReport,
    MessageType.WATER_SOURCE_REPORT: WaterSourceReport,
    MessageType.MAP_SYNC: MapSync,
    MessageType.TASK_ANNOUNCE: TaskAnnounce,
    MessageType.TASK_BID: TaskBid,
    MessageType.TASK_ASSIGN: TaskAssign,
    MessageType.TASK_ACK: TaskAck,
    MessageType.TASK_COMPLETE: TaskComplete,
    MessageType.PATH_CLAIM: PathClaim,
    MessageType.LEADER_CLAIM: LeaderClaim,
    MessageType.REFILL_REQUEST: RefillRequest,
    MessageType.REFILL_GRANT: RefillGrant,
    MessageType.GROUND_COMMAND: GroundCommand,
    MessageType.ALERT: AlertNotice,
    MessageType.SIM_FAULT: SimFault,
}


class QoSProfile(FrozenContract):
    """How hard the bus should try to deliver a message type."""

    reliable: bool          # True = retransmit until acked; False = best-effort, newest wins
    ttl_s: float = Field(gt=0)  # drop if older than this on arrival
    rate_hz: float | None = Field(default=None, gt=0)  # nominal publish rate for periodic messages


DEFAULT_QOS: dict[MessageType, QoSProfile] = {
    MessageType.HEARTBEAT: QoSProfile(reliable=False, ttl_s=3, rate_hz=2),
    MessageType.HOTSPOT_REPORT: QoSProfile(reliable=True, ttl_s=60),
    MessageType.WATER_SOURCE_REPORT: QoSProfile(reliable=True, ttl_s=60),
    MessageType.MAP_SYNC: QoSProfile(reliable=False, ttl_s=10, rate_hz=1),
    MessageType.TASK_ANNOUNCE: QoSProfile(reliable=True, ttl_s=10),
    MessageType.TASK_BID: QoSProfile(reliable=True, ttl_s=10),
    MessageType.TASK_ASSIGN: QoSProfile(reliable=True, ttl_s=30),
    MessageType.TASK_ACK: QoSProfile(reliable=True, ttl_s=30),
    MessageType.TASK_COMPLETE: QoSProfile(reliable=True, ttl_s=60),
    MessageType.PATH_CLAIM: QoSProfile(reliable=False, ttl_s=5, rate_hz=2),
    MessageType.LEADER_CLAIM: QoSProfile(reliable=True, ttl_s=5, rate_hz=1),
    MessageType.REFILL_REQUEST: QoSProfile(reliable=True, ttl_s=30),
    MessageType.REFILL_GRANT: QoSProfile(reliable=True, ttl_s=30),
    MessageType.GROUND_COMMAND: QoSProfile(reliable=True, ttl_s=60),
    MessageType.ALERT: QoSProfile(reliable=True, ttl_s=60),
    MessageType.SIM_FAULT: QoSProfile(reliable=True, ttl_s=60),
}


class SwarmMessage(FrozenContract):
    """Envelope for everything on the bus."""

    v: str = CONTRACT_VERSION
    msg_id: str
    seq: int = Field(default=0, ge=0)          # per-sender increasing counter
    sender_id: str
    receiver_id: str | None = None             # None = broadcast to topic subscribers
    timestamp: float                           # Clock seconds at send time
    ttl_s: float = Field(default=5.0, gt=0)
    reliable: bool
    epoch: int | None = Field(default=None, ge=0)  # leader epoch the sender believes in (commands)
    body: Body

    @property
    def msg_type(self) -> MessageType:
        return self.body.type

    def is_expired(self, now: float) -> bool:
        return (now - self.timestamp) > self.ttl_s

    @classmethod
    def create(
        cls,
        sender_id: str,
        body: _Body,
        now: float,
        *,
        seq: int = 0,
        receiver_id: str | None = None,
        ttl_s: float | None = None,
        reliable: bool | None = None,
        epoch: int | None = None,
        msg_id: str | None = None,
    ) -> "SwarmMessage":
        """Build a message with the default QoS for its type. Increment `seq` per sender per message."""
        qos = DEFAULT_QOS[body.type]
        return cls(
            msg_id=msg_id or f"{sender_id}-{seq}",
            seq=seq,
            sender_id=sender_id,
            receiver_id=receiver_id,
            timestamp=now,
            ttl_s=qos.ttl_s if ttl_s is None else ttl_s,
            reliable=qos.reliable if reliable is None else reliable,
            epoch=epoch,
            body=body,
        )


def encode(msg: SwarmMessage) -> bytes:
    """Serialize for the bus. Use at EVERY boundary, including the in-process bus."""
    return msg.model_dump_json().encode("utf-8")


def decode(raw: bytes | str) -> SwarmMessage:
    """Parse + validate. Raises pydantic.ValidationError on malformed input."""
    return SwarmMessage.model_validate_json(raw)
```

#### `contracts/topics.py` (87 lines)

```python
"""Topic names and matching rules.

Convention (same idea works for ROS 2 namespaces, MQTT topics, or the in-process bus):
  drone/{id}/...   per-drone channels            ground/...   ground base channels
  swarm/...        shared swarm channels         sim/...      simulator-only channels

Wildcards in SUBSCRIPTION patterns:
  *  matches exactly one segment      #  matches the rest (zero or more segments), must be last
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .enums import MessageType

if TYPE_CHECKING:
    from .messages import SwarmMessage

__all__ = [
    "ROUTES", "topic_for", "topic_matches", "validate_pattern",
    "SUB_ALL_HEARTBEATS", "SUB_ALL_HOTSPOTS", "SUB_ALL_WATER_SOURCES", "SUB_ALL_SWARM",
    "SUB_GROUND_COMMANDS", "SUB_GROUND_ALERTS", "SUB_SIM_CONTROL", "sub_my_commands",
]

# Publish routes. {sender}/{receiver} are filled from the message envelope.
ROUTES: dict[MessageType, str] = {
    MessageType.HEARTBEAT: "drone/{sender}/heartbeat",
    MessageType.HOTSPOT_REPORT: "drone/{sender}/hotspots",
    MessageType.WATER_SOURCE_REPORT: "drone/{sender}/water_sources",
    MessageType.MAP_SYNC: "swarm/map_sync",
    MessageType.TASK_ANNOUNCE: "swarm/tasks/announce",
    MessageType.TASK_BID: "swarm/tasks/bids",
    MessageType.TASK_ASSIGN: "drone/{receiver}/cmd",
    MessageType.TASK_ACK: "swarm/tasks/acks",
    MessageType.TASK_COMPLETE: "swarm/tasks/complete",
    MessageType.PATH_CLAIM: "swarm/path_claims",
    MessageType.LEADER_CLAIM: "swarm/leader",
    MessageType.REFILL_REQUEST: "swarm/refill/requests",
    MessageType.REFILL_GRANT: "drone/{receiver}/cmd",
    MessageType.GROUND_COMMAND: "ground/commands",
    MessageType.ALERT: "ground/alerts",
    MessageType.SIM_FAULT: "sim/control",
}

# Common subscription patterns
SUB_ALL_HEARTBEATS = "drone/*/heartbeat"          # leader + ground base
SUB_ALL_HOTSPOTS = "drone/*/hotspots"             # leader
SUB_ALL_WATER_SOURCES = "drone/*/water_sources"   # leader
SUB_ALL_SWARM = "swarm/#"                         # every drone
SUB_GROUND_COMMANDS = "ground/commands"           # every drone (filters on target_id)
SUB_GROUND_ALERTS = "ground/alerts"               # ground base UI
SUB_SIM_CONTROL = "sim/control"                   # simulator (and sim-aware drone wrappers)


def sub_my_commands(drone_id: str) -> str:
    """Direct commands to one drone (TASK_ASSIGN, REFILL_GRANT)."""
    return f"drone/{drone_id}/cmd"


def topic_for(msg: "SwarmMessage") -> str:
    """Topic a message should be published on. Directed messages require receiver_id."""
    pattern = ROUTES[msg.msg_type]
    if "{receiver}" in pattern and not msg.receiver_id:
        raise ValueError(f"{msg.msg_type} requires receiver_id")
    return pattern.format(sender=msg.sender_id, receiver=msg.receiver_id)


def validate_pattern(pattern: str) -> None:
    segs = pattern.split("/")
    if any(s == "" for s in segs):
        raise ValueError(f"empty segment in pattern {pattern!r}")
    if "#" in segs and segs.index("#") != len(segs) - 1:
        raise ValueError("'#' must be the last segment")


def topic_matches(pattern: str, topic: str) -> bool:
    """MQTT-like matching. Reuse this in every Bus implementation."""
    p = pattern.split("/")
    t = topic.split("/")
    for i, seg in enumerate(p):
        if seg == "#":
            return i == len(p) - 1
        if i >= len(t):
            return False
        if seg != "*" and seg != t[i]:
            return False
    return len(p) == len(t)
```

#### `contracts/interfaces.py` (107 lines)

```python
"""Interfaces (Protocols) between layers. Drone logic depends ONLY on these, never on ROS/PX4/FastAPI.

Fidelity levels are different implementations of the same interfaces:
  Level 0 (abstract sim):  SimBus / SimClock / SimAutopilot / SimPayload / FakeDetector / SimThermalSensor
  Level 1 (PX4 SITL):      same Bus/Clock; Px4Autopilot (MAVSDK); Gazebo or sim-supplied sensors
  Level 2 (real drone):    Ros2Bus or MqttBus; Px4Autopilot; real camera drivers + OnnxDetector

Execution model (recommended): single-threaded, tick-based and deterministic.
The simulator calls `tick(now, dt)` on every Tickable each step; the SimBus delivers queued
messages (applying latency/loss) at the start of each tick.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from .enums import DetectorKind, ExtinguishingAgent
from .geometry import Pose, Position
from .messages import SwarmMessage
from .sensing import CameraModel, Detection, Obstacle, ThermalFrame

__all__ = [
    "MessageHandler", "Subscription", "Clock", "Bus", "Tickable", "Autopilot", "PayloadSystem",
    "Detector", "Georeferencer", "ThermalSensor", "ObstacleSensor",
]

MessageHandler = Callable[[SwarmMessage], None]


@runtime_checkable
class Subscription(Protocol):
    def cancel(self) -> None: ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Seconds. Sim time in the simulator, wall-clock epoch seconds on hardware."""
        ...


@runtime_checkable
class Bus(Protocol):
    """Pub/sub. Implementations MUST serialize with encode()/decode() and use topic_matches()."""

    def publish(self, topic: str, msg: SwarmMessage) -> None: ...

    def subscribe(self, pattern: str, handler: MessageHandler) -> Subscription: ...


@runtime_checkable
class Tickable(Protocol):
    def tick(self, now: float, dt: float) -> None: ...


@runtime_checkable
class Autopilot(Protocol):
    """Flight control. goto() is a REQUEST: arrival is observed via position()/at_target(), not assumed."""

    def position(self) -> Position: ...
    def heading_deg(self) -> float: ...
    def speed_mps(self) -> float: ...
    def battery_pct(self) -> float: ...
    def endurance_remaining_s(self) -> float: ...
    def goto(self, target: Position, speed_mps: float | None = None) -> None: ...
    def hold(self) -> None: ...
    def return_to_launch(self) -> None: ...
    def at_target(self, tolerance_m: float = 2.0) -> bool: ...


@runtime_checkable
class PayloadSystem(Protocol):
    """Water/agent tank on an EXTINGUISHER."""

    def water_remaining_l(self) -> float: ...
    def agent(self) -> ExtinguishingAgent: ...
    def release(self, volume_l: float, at: Position) -> float:
        """Drop up to volume_l at `at`; returns litres actually released."""
        ...
    def fill(self, volume_l: float, agent: ExtinguishingAgent = ExtinguishingAgent.WATER) -> float:
        """Add up to volume_l (capped by capacity); returns litres actually added."""
        ...


@runtime_checkable
class Detector(Protocol):
    """Image -> detections. Implementations: THRESHOLD (classical), ML (ONNX), FUSED, FAKE."""

    kind: DetectorKind

    def detect(self, frame: ThermalFrame) -> list[Detection]: ...


@runtime_checkable
class Georeferencer(Protocol):
    """Pixel -> world. Returns None if the ray misses the ground (e.g. above horizon)."""

    def pixel_to_world(self, u: float, v: float, pose: Pose, camera: CameraModel) -> Position | None: ...


@runtime_checkable
class ThermalSensor(Protocol):
    def capture(self) -> ThermalFrame: ...


@runtime_checkable
class ObstacleSensor(Protocol):
    def obstacles(self) -> list[Obstacle]: ...
```

#### `contracts/config.py` (254 lines)

```python
"""Platform, sensor, agent and scenario configuration.

EVERY NUMBER HERE IS AN ASSUMPTION unless its comment says SOURCED. Rationale and sources live in
context.md (sections 6-8). Change defaults in ONE place (here), then add a changelog line to context.md.
The mech E owns the physical values; CS agents should not "tune" them to make the demo look better.
"""
from __future__ import annotations

from pydantic import Field, model_validator

from .base import Contract
from .entities import RefillStation, WaterSource
from .enums import (
    DroneKind, ExtinguishingAgent, WaterSourceKind, WaterSourceOrigin, WaterSourceStatus,
)
from .geometry import GeoOrigin, GridSpec, Position
from .sensing import CameraModel

__all__ = [
    "SENSOR_NAMES", "ThermalCameraSpec", "LidarSpec", "CommsSpec", "PlatformSpec", "AgentSpec",
    "FireSpec", "DecoySpec", "WindSpec", "ScenarioConfig", "DEFAULT_AGENT_SPECS",
    "default_seeker_spec", "default_extinguisher_spec", "example_scenario",
]

# Keys used in DroneState.sensors_ok and SimFault(DEGRADE_SENSOR).
SENSOR_NAMES = ("thermal", "lidar", "gnss_rtk", "rangefinder", "comms", "compute", "payload")


class ThermalCameraSpec(Contract):
    camera: CameraModel
    netd_mk: float = Field(default=50.0, gt=0)          # noise-equivalent delta-T, millikelvin
    frame_rate_hz: float = Field(default=9.0, gt=0)     # ASSUMPTION: export-friendly cores are often limited; verify
    saturation_temp_c: float = 550.0                    # ASSUMPTION: low-gain radiometric range; fires can exceed this
    mass_kg: float = Field(default=0.025, gt=0)         # SOURCED: Boson+ ~21-23 g with lens
    power_w: float = Field(default=1.0, gt=0)           # SOURCED: Boson from ~0.5 W; dual thermal+visible module 1.8 W


class LidarSpec(Contract):
    range_m: float = Field(default=40.0, gt=0)          # SOURCED: Livox Mid-360, typical on 10% reflectivity target
    fov_h_deg: float = 360.0                            # SOURCED
    fov_v_deg: float = 59.0                             # SOURCED
    points_per_s: int = 200_000                         # SOURCED
    mass_kg: float = 0.265                              # SOURCED
    power_w: float = 6.5                                # SOURCED (average)


class CommsSpec(Contract):
    """Drone-to-drone and drone-to-ground radio model used by the simulator's lossy bus."""

    range_m: float = Field(default=3000.0, gt=0)        # ASSUMPTION: mesh radio, line of sight
    ground_link_range_m: float = Field(default=5000.0, gt=0)
    latency_s: float = Field(default=0.05, ge=0)
    latency_jitter_s: float = Field(default=0.02, ge=0)
    loss_prob_near: float = Field(default=0.01, ge=0, le=1)
    loss_prob_edge: float = Field(default=0.5, ge=0, le=1)   # loss probability at range_m; interpolate between
    bandwidth_kbps: float = Field(default=500.0, gt=0)       # why raw frames never travel on the bus


class PlatformSpec(Contract):
    kind: DroneKind
    name: str
    empty_mass_kg: float = Field(gt=0)
    max_takeoff_kg: float = Field(gt=0)
    water_capacity_l: float = Field(default=0.0, ge=0)  # 1 L of water ~ 1 kg
    cruise_speed_mps: float = Field(gt=0)
    max_speed_mps: float = Field(gt=0)
    max_accel_mps2: float = Field(default=3.0, gt=0)    # also used as braking decel for stopping distance
    endurance_s: float = Field(gt=0)                    # with generator; provider range is 5-8 h (18000-28800 s)
    search_altitude_agl_m: float = Field(gt=0)
    drop_height_agl_m: float = Field(default=10.0, gt=0)
    drop_release_s: float = Field(default=8.0, gt=0)    # time to empty the tank
    refill_time_s: float = Field(default=120.0, gt=0)   # hook-up + fill (+ dosing)
    avionics_mass_kg: float = Field(gt=0)
    avionics_power_w: float = Field(gt=0)
    thermal: ThermalCameraSpec
    lidar: LidarSpec | None = None
    has_rtk_gnss: bool = True
    sensor_names: list[str] = Field(default_factory=lambda: list(SENSOR_NAMES))

    @model_validator(mode="after")
    def _physics_sane(self) -> "PlatformSpec":
        if self.empty_mass_kg + self.water_capacity_l > self.max_takeoff_kg + 1e-9:
            raise ValueError("empty_mass_kg + water_capacity_l exceeds max_takeoff_kg")
        if self.cruise_speed_mps > self.max_speed_mps:
            raise ValueError("cruise_speed_mps exceeds max_speed_mps")
        return self

    def stopping_distance_m(self, speed_mps: float) -> float:
        """v^2 / (2a): distance needed to stop from `speed_mps` at max_accel_mps2."""
        return speed_mps ** 2 / (2 * self.max_accel_mps2)


class AgentSpec(Contract):
    """Extinguishing agent model. `water_equiv_multiplier`: litres of plain water that 1 L of this agent replaces."""

    agent: ExtinguishingAgent
    water_equiv_multiplier: float = Field(gt=0)
    concentrate_ratio: float | None = Field(default=None, ge=0, le=1)  # fraction of concentrate in solution
    needs_dosing: bool                  # True = needs an onboard/inline proportioner at fill time
    aquatic_buffer_m: float = Field(default=0.0, ge=0)  # no aerial application within this distance of waterways
    note: str = ""


# SOURCED/ASSUMED values, see context.md section 7.
DEFAULT_AGENT_SPECS: dict[ExtinguishingAgent, AgentSpec] = {
    ExtinguishingAgent.WATER: AgentSpec(
        agent=ExtinguishingAgent.WATER, water_equiv_multiplier=1.0, needs_dosing=False,
        note="Baseline. Compatible with any refill source.",
    ),
    ExtinguishingAgent.CLASS_A_FOAM: AgentSpec(
        agent=ExtinguishingAgent.CLASS_A_FOAM, water_equiv_multiplier=3.0, concentrate_ratio=0.003,
        needs_dosing=True, aquatic_buffer_m=91.44,
        note="Mix 0.1-1% (SOURCED). Manufacturer claims 3-5x vs plain water; independent evidence is mixed "
             "(ASSUMPTION 3x). US guideline: no aerial foam/retardant within 300 ft (91.44 m) of waterways "
             "(VERIFY Alberta/Canada equivalent).",
    ),
    ExtinguishingAgent.GEL: AgentSpec(
        agent=ExtinguishingAgent.GEL, water_equiv_multiplier=1.5, needs_dosing=True, aquatic_buffer_m=91.44,
        note="PLACEHOLDER multiplier, no source. Gel consistency is sensitive to water hardness (SOURCED), "
             "so basin water quality matters.",
    ),
}


def default_seeker_spec() -> PlatformSpec:
    return PlatformSpec(
        kind=DroneKind.SEEKER, name="Seeker (heavy-lift airframe, light payload)",
        empty_mass_kg=52.0, max_takeoff_kg=92.0, water_capacity_l=0.0,   # airframe: DJI Agras T50-class (SOURCED)
        cruise_speed_mps=15.0, max_speed_mps=22.0, endurance_s=6 * 3600.0,  # speeds ASSUMED; 6 h = mid of 5-8 h
        search_altitude_agl_m=80.0, avionics_mass_kg=1.0, avionics_power_w=48.0,
        thermal=ThermalCameraSpec(camera=CameraModel(width_px=640, height_px=512, hfov_deg=32.0)),
        lidar=LidarSpec(),
    )


def default_extinguisher_spec() -> PlatformSpec:
    return PlatformSpec(
        kind=DroneKind.EXTINGUISHER, name="Extinguisher (T50-class, 40 L)",
        empty_mass_kg=52.0, max_takeoff_kg=92.0, water_capacity_l=40.0,  # T50: 52 kg w/ battery, 92 kg spray MTOW, 40 L (SOURCED)
        cruise_speed_mps=15.0, max_speed_mps=22.0, endurance_s=6 * 3600.0,
        search_altitude_agl_m=60.0, drop_height_agl_m=10.0, avionics_mass_kg=1.0, avionics_power_w=40.0,
        thermal=ThermalCameraSpec(camera=CameraModel(width_px=320, height_px=256, hfov_deg=50.0)),
        lidar=LidarSpec(),
    )


class FireSpec(Contract):
    fire_id: str
    position: Position
    ignition_time_s: float = Field(default=0.0, ge=0)
    initial_area_m2: float = Field(gt=0)
    peak_temp_c: float = Field(default=450.0, ge=-50, le=1500)
    growth_rate_m2_per_min: float = Field(default=0.1, ge=0)  # night fire behaviour is low; derive from CWFIS


class DecoySpec(Contract):
    """Heat source that is NOT a fire (warm rock, vehicle, permitted campfire). Detectors must not waste water on it."""

    decoy_id: str
    position: Position
    temp_c: float
    area_m2: float = Field(gt=0)
    kind: str = "warm_rock"


class WindSpec(Contract):
    speed_mps: float = Field(default=2.0, ge=0)
    from_deg: float = Field(default=270.0, ge=0, lt=360)  # compass direction wind blows FROM


def _default_origin() -> GeoOrigin:
    # PLACEHOLDER site (central Alberta). Replace with the real scenario location.
    return GeoOrigin(lat_deg=53.5, lon_deg=-113.5, utm_epsg=32612)


def _default_grid() -> GridSpec:
    # 4 km x 4 km at 2 m cells = 2000 x 2000 (4M cells = 16 MB per float32 layer)
    return GridSpec(origin=Position(x=-2000.0, y=-2000.0), resolution_m=2.0, rows=2000, cols=2000)


class ScenarioConfig(Contract):
    name: str = "default"
    seed: int = 0
    duration_s: float = Field(default=3600.0, gt=0)
    time_scale: float = Field(default=1.0, gt=0)  # sim seconds per wall second
    night: bool = True
    ambient_temp_c: float = 8.0
    wind: WindSpec = Field(default_factory=WindSpec)
    origin: GeoOrigin = Field(default_factory=_default_origin)
    grid: GridSpec = Field(default_factory=_default_grid)
    base_position: Position = Field(default_factory=lambda: Position(x=0.0, y=0.0, z=0.0))

    n_seekers: int = Field(default=4, ge=0)
    n_extinguishers: int = Field(default=6, ge=0)
    seeker_spec: PlatformSpec = Field(default_factory=default_seeker_spec)
    extinguisher_spec: PlatformSpec = Field(default_factory=default_extinguisher_spec)

    agent: ExtinguishingAgent = ExtinguishingAgent.WATER
    agent_specs: dict[ExtinguishingAgent, AgentSpec] = Field(default_factory=lambda: dict(DEFAULT_AGENT_SPECS))
    suppression_l_per_m2: float = Field(default=10.0, gt=0)  # PLACEHOLDER: mech E must source/derive (fuel, burn depth)

    use_water_sources: bool = True  # feature flag: allow refilling from natural water sources
    water_sources: list[WaterSource] = Field(default_factory=list)  # prior map (CANDIDATE) and/or known sources
    stations: list[RefillStation] = Field(default_factory=list)

    fires: list[FireSpec] = Field(default_factory=list)
    decoys: list[DecoySpec] = Field(default_factory=list)
    comms: CommsSpec = Field(default_factory=CommsSpec)

    @model_validator(mode="after")
    def _consistent(self) -> "ScenarioConfig":
        if not self.grid.contains(self.base_position):
            raise ValueError("base_position must lie inside the grid")
        for label, ids in (
            ("fire_id", [f.fire_id for f in self.fires]),
            ("decoy_id", [d.decoy_id for d in self.decoys]),
            ("source_id", [w.source_id for w in self.water_sources]),
            ("station_id", [s.station_id for s in self.stations]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} in scenario")
        if self.agent not in self.agent_specs:
            raise ValueError("agent has no entry in agent_specs")
        return self


def example_scenario() -> ScenarioConfig:
    """Small, valid scenario for sim/UI developers to start from. Positions are local meters."""
    return ScenarioConfig(
        name="example_night_initial_attack",
        seed=42,
        fires=[
            FireSpec(fire_id="F1", position=Position(x=800, y=1200), ignition_time_s=0,
                     initial_area_m2=2.0, peak_temp_c=450, growth_rate_m2_per_min=0.10),
            FireSpec(fire_id="F2", position=Position(x=-1500, y=600), ignition_time_s=300,
                     initial_area_m2=1.0, peak_temp_c=300, growth_rate_m2_per_min=0.05),
            FireSpec(fire_id="F3", position=Position(x=300, y=-1700), ignition_time_s=900,
                     initial_area_m2=4.0, peak_temp_c=600, growth_rate_m2_per_min=0.20),
        ],
        decoys=[DecoySpec(decoy_id="D1", position=Position(x=-400, y=900), temp_c=35.0, area_m2=6.0)],
        water_sources=[
            WaterSource(source_id="W1", position=Position(x=1000, y=900), kind=WaterSourceKind.POND,
                        origin=WaterSourceOrigin.PRIOR_MAP, status=WaterSourceStatus.CANDIDATE,
                        est_volume_l=2_000_000, confidence=0.6),
            WaterSource(source_id="W2", position=Position(x=-1800, y=-900), kind=WaterSourceKind.LAKE,
                        origin=WaterSourceOrigin.PRIOR_MAP, status=WaterSourceStatus.CANDIDATE,
                        est_volume_l=50_000_000, confidence=0.8),
        ],
        stations=[
            RefillStation(station_id="S1", position=Position(x=0, y=0), water_available_l=20_000,
                          fill_rate_lpm=60.0, total_bays=2, free_bays=2,
                          agents_available=[ExtinguishingAgent.WATER, ExtinguishingAgent.CLASS_A_FOAM]),
        ],
    )
```

<!-- CONTRACTS:END -->

---

## 12. Open questions (for the human)

1. `[OPEN]` **Payload class:** keep the T50-class 40 L default, or model an ultra-heavy (ThunderWasp-class) airframe? Drives swarm size and refill logistics.
2. `[OPEN]` **Leaders:** one global leader (+standby) or cluster leaders per area?
3. `[OPEN]` **Human gate:** require ground approval before the first drop, or only before new water sources?
4. `[OPEN]` **Scenario site and area:** default is a 4 km × 4 km grid at 2 m cells around a placeholder central-Alberta origin.
5. `[OPEN]` **Sim engine owner** (see §2) and **evaluation harness owner**.
6. `[OPEN]` **Intake method** for water bodies (bucket vs hose + pump) and whether generator mass should be modelled.
7. `[VERIFY]` Alberta/Canadian rules on drawing water from natural sources and on foam/retardant near waterways.
8. `[VERIFY]` Export/availability limits on thermal cameras (frame rate, licensing) for any real hardware.

---

## 13. Changelog

| Date | By | Change | Why |
|---|---|---|---|
| 2026-11-07 | initial chat | Created context v0.1 and contracts 0.1.0 (49 tests passing); sensor baseline reviewed against FireSwarm; added agents, refill/basin, gas-sensing and ML-data sections | project kickoff for parallel work |

---

## 14. Sources

- **S1** FireSwarm XPRIZE finalist release — https://www.aap.com.au/aapreleases/globenewswire9648888/
- **S2** ASCE, FireSwarm automated aerial firefighting — https://www.asce.org/publications-and-news/civil-engineering-source/article/2025/07/24/automated-aerial-firefighting-drones-to-provide-faster-responses
- **S3** CTIF, 400 kg water-bombing drones in Canada — https://ctif.org/news/heavy-duty-water-bombing-drones-able-lift-400-kilos-are-being-tested-canadian-forest-fire
- **S4** Strategic × FireSwarm partnership — https://newswire.ca/news-releases/revolutionary-drone-technology-for-battling-wildfires-takes-major-step-forward-with-new-partnership-840102801.html
- **S5** DJI Agras T50 — https://ag.dji.com/fr/t50/specs · reseller specs/hover times https://druav.com/products/dji-agras-t50-agriculture-drone · battery/radar https://volatusdrones.com/products/dji-agras-t50-drone-ready-to-fly-spray-bundle
- **S6** US initial-attack drone fleets — https://www.yahoo.com/news/articles/san-bernardino-co-fire-dept-211235723.html · https://www.techspot.com/news/113180-us-fire-departments-test-autonomous-drones-fight-wildfires.html
- **S7** FLIR Boson+ — https://www.acalbfi.com/nl/file/16132/Boson-Plus-Datasheet.pdf · https://oemcameras.com/?p=4033 · Hadron 640R https://w-technika.cz/cdn/shop/files/hadron20640r20datasheet.pdf
- **S8** Livox Mid-360 — https://www.livoxtech.com/mid-360/specs · https://www.mybotshop.de/Livox-Mid360S_1
- **S9** Jetson Orin NX datasheet — https://openzeka.com/wp-content/uploads/2022/03/Jetson-Orin-NX-Module-Series-Datasheet.pdf
- **S10** Class A foam & fire chemicals — https://www.fs.usda.gov/eng/pubs/pdf/92511209.pdf · https://www.fs.usda.gov/rm/fire/wfcs/documents/Overview%20of%20Use%20and%20Safety%20of%20Wildland%20Fire%20Chemical%20Products.pdf · https://www.perimeter-solutions.com/?p=181 · https://www.fireengineering.com/fire-safety/the-salem-tests/ · https://www.fireengineering.com/?p=264355
- **S11** Waterway buffers / retardant toxicity — https://www.fs.usda.gov/t-d/programs/wfcs/retardants/current/gen/appguide.htm · https://www.nifc.gov/sites/default/files/redbook-files/Chapter12.pdf · https://www.hcn.org/?p=199704
- **S12** Gas/smoke sensing — https://dryad.net/wildfiresensor · https://arxiv.org/pdf/2312.10919 · https://hellofuture.orange.com/en/a-drone-that-can-sniff-out-forest-fires-before-they-generate-visible-smoke · https://tech.eu/2023/08/11/ultra-early-detection-system/
- **S13** FLAME 3 — https://arxiv.org/abs/2412.02831 · https://arxiv.org/html/2412.02831v1
- **S14** PX4 — https://docs.px4.io/main/en/dev_setup/dev_env · https://docs.px4.io/main/en/sim_gazebo_gz/multi_vehicle_simulation
- **S15** ROS 2 Lyrical Luth — https://discourse.openrobotics.org/t/ros-2-lyrical-luth-released/55021
- **S16** Water pickup / amphibious drones — https://www.popsci.com/technology/firefighting-drones-help-fight-wildfires/
