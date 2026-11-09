# natHacks 2026 · Team Pandas · Night Initial-Attack Drone Swarm — CONTEXT

`context_version: 0.3.7` · `contract_version: 0.3.1` · last updated 2026-11-09

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
- Multiple drone types: **Seekers** (light, electric; search + confirm + survey water) and **Extinguishers** (ultra-heavy, **hybrid** generator + battery; carry water, drop on seeker-confirmed hotspots, do not search). Chosen 2026-11-07, FireSwarm/Thunder-Wasp class.

**Demo reality:** the final demo is **purely software**, shown in a room — no drones, no real fires. Sensor data is synthetic or comes from static datasets. The **stretch goal** is code that runs on real drones (ROS 2 / PX4 / Jetson) without rewriting the logic.

---

## 2. Team and workstreams (team of 5: 1 mech E, 4 CS)

| ID | Role | Owns | Talks to others via |
|---|---|---|---|
| **A** (CS) | Swarm brain | Leader role, election/failover, task allocation (auction/greedy → CBBA stretch), hotspot prioritization, refill assignment, comms protocol, failure handling | messages in `contracts/messages.py` |
| **B** (CS) | Per-drone autonomy | Coverage search, path planning, obstacle avoidance, deconfliction (`PathClaim`), endurance logic, extinguisher drop logic, refill execution, self-health checks | `Autopilot`, `PayloadSystem`, `Bus` |
| **C** (CS) | Perception | Detectors (threshold, FUSED = threshold + **CNN verifier** in ONNX, fake), georeferencing, log-odds map, tracking, F1 evaluation, model export, synthetic thermal frames (with E); GBM verifier and water-body detection (**stretch**) | `Detector`, `Georeferencer`, `ThermalFrame` |
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
core/         pure-Python swarm brain: priority, priority-queue allocation, zones, SwarmLeader (no ROS/web imports)
perception/   detectors, georeferencing, mapping/tracking, evaluation; models/ (shipped .onnx), RESULTS.md
training/     offline training: train_verifier.ipynb (the whole loop, with plots), CNN primary / GBM stretch, composite.py,
              label_tool.py, dataset adapters, FLAME3.md (citation, how labels are made, cameras/GSD)
exclude/      GITIGNORED: raw datasets, Colab scratch, training runs, DATA_SOURCES.md, training_README.md
sim/          world, fire model, sensor/comms models, scenario runner, metrics
groundbase/   FastAPI, health monitor, DB
frontend/     React: ops console + cinematic sim view
adapters/     (stretch) ros2, mqtt, px4
scenarios/    YAML/JSON ScenarioConfig files
tests/  tools/  context.md
.github/workflows/tests.yml   CI: full pytest suite on push to main + PRs into main (Python 3.11, 3.12)
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
        │ electric, ~52 min    │          │ hybrid, 170 L, fuel-ltd  │
        │ 640x512 thermal+lidar│          │ 320x256 thermal + lidar  │
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
8. **Fuel is a hard constraint for extinguishers.** A hybrid drone returns to a *station* (water sources can't refuel it) when `fuel_remaining_kg` ≤ reserve + fuel to get home. Fuel and battery-buffer state ride in every heartbeat (`DroneState.fuel_*`).
9. **Extinguishers don't scout.** They fly corridors already cleared by Seekers/DEM and approach hotspots a Seeker has confirmed (stopping distance at 20 m/s is 100 m, far beyond the 40 m lidar). Humans stay in the loop, see rule 10.
10. **Humans declare fires out.** Drones can mark a hotspot `SUPPRESSED`; only `GroundCommand(CONFIRM_OUT)` produces `VERIFIED_OUT` `[SOURCED S2]` (FireSwarm: fires are suppressed by air assets, extinguished by humans).

**Failure behaviour:**
- *Leader lost* (missed heartbeats / `LeaderClaim` silence) → followers finish the current task, then hold; election: highest `(epoch, score, candidate_id)` wins; new leader increments epoch and re-announces.
- *Follower lost* → leader marks it `LOST`, re-queues its task, emits an `ALERT`.
- *Ground link lost* → swarm continues; ground shows drones as `LOST` until heartbeats resume.
- *RTK corrections lost* → `sensors_ok["gnss_rtk"]=False`; drop accuracy degrades (model it in sim).
- *Low fuel / battery / water* → return/refill, never "push on". `AlertKind.LOW_FUEL` / `LOW_BATTERY`.
- *Powerplant fault* (generator or engine, `sensors_ok["power"]=False`) → the 5 kWh buffer gives about 2 min of hover-equivalent power: land immediately at the nearest cleared spot; emit `AlertKind.POWERPLANT_FAULT`; leader re-queues the task.
- *Seeker battery low* → return for a swap (~52 min sorties); the leader must rotate seekers so coverage doesn't gap.

**Messaging (QoS):** heartbeats/telemetry and map deltas are best-effort (stale is worthless); hotspot reports, task assignment/ack/complete, refill, commands and alerts are reliable. Defaults are in `DEFAULT_QOS`. Topics: `drone/{id}/…`, `swarm/…`, `ground/…`, `sim/…` (see `ROUTES`).

---

## 5. Tech stack

| Layer | Choice | Status | Notes |
|---|---|---|---|
| Language | Python 3.11+ | core | typed, `pydantic>=2.6` |
| Contracts | Pydantic v2 models, discriminated-union messages | core | validate at boundaries; arrays (NumPy) inside hot loops |
| Bus | in-process `SimBus` (with `LossyBus` wrapper: latency/loss by distance) | core | implements `Bus`; use `topic_matches()` |
| Imaging | NumPy, OpenCV, `scipy.ndimage`, scikit-image | core | keep radiometric °C until after thresholding |
| ML inference | ONNX Runtime (train in PyTorch, export `.onnx`) | core | `Detector` implementations: THRESHOLD, FUSED (threshold proposals + small CNN patch verifier, the primary ML), FAKE. GBM-on-features verifier (sklearn → skl2onnx) = **stretch**. `DetectorKind.ML` reserved |
| Planning | A*/potential fields on grid; OR-Tools / `scipy.optimize.linear_sum_assignment` for allocation; Voronoi/boustrophedon for zones | core | CBBA = stretch |
| Georeferencing | `pyproj` + `rasterio` (DEM) | core (edges only) | lat/lon never inside `core/` |
| Ground base | FastAPI + WebSockets, SQLite | core | `diagnostics`-style health from heartbeats |
| Frontend | React + TypeScript + Vite; MapLibre GL + deck.gl (ops map); PixiJS/p5.js (breathing-polygons view) | core | one WebSocket state stream feeds both views |
| Debug/replay | Foxglove / Rerun, MCAP logs | optional | |
| Real-world middleware | ROS 2 (Jazzy on Ubuntu 24.04 is the safe pick; Lyrical Luth is the newest LTS `[SOURCED S15]`), PX4 + MAVSDK | stretch | adapters only |
| Tooling | `uv`, `ruff`, `pytest`, pre-commit; Docker/devcontainer for Linux-only parts | core | |
| CI | GitHub Actions `.github/workflows/tests.yml` | core | push to `main` + PRs into `main`; Python 3.11/3.12; installs `.[dev,perception,train]` (torch CPU wheel) so **every** test runs, none skipped |

---

## 6. Drone platform and sensor baseline

### 6.1 Reference platforms (real hardware that anchors our numbers)

| Class | Example | Figures | Source |
|---|---|---|---|
| **Ultra-heavy hybrid (our extinguisher anchor)** | ACC Thunder Wasp GT, FireSwarm's platform | quadrotor, gas turbine on Jet A-1; MTOW 800 kg, empty 375 kg; up to 248 kg fuel → just under 4 h with 177 kg payload; with 400 kg payload ~20 min; cruise 70–90 km/h, max 110 km/h; PBS TS100 turbine 61.2 kg with a 1 kW generator; 378.5 L (100 gal) max water; carries a Bambi bucket and lowers *itself* (no winch) | S17, S18 |
| Light electric (seeker anchor) | commodity thermal/lidar quadcopters | ~10 kg class, ~45–55 min; we derive it in §6.3 | derived |
| Electric heavy ag drone (reference only) | DJI Agras T50 | 52 kg w/ battery, 92 kg MTOW, 40 L; ~7 min hover at full load per a reseller | S5 |
| US initial-attack fleets | FireSparrow-class | ~10 gal payload, ~10 mi round trip, technician refills and swaps battery "within minutes" | S6 |

FireSwarm's published endurance is all over the place: 30–40 min (evtol.news, GT hybrid-electric prototype), 60–90 min (ASCE), up to 2 h (CTIF), vs ACC's own **~4 h at 177 kg payload / ~20 min at 400 kg**. Payload and fuel trade 1:1 inside the same 800 kg MTOW, which explains it `[S1,S2,S3,S17,S18]`. The provider's "generator gives 5–8 h" is therefore *our assumption to defend*, so §6.2 shows what it costs.

### 6.2 Extinguisher sizing (hybrid, ultra-heavy) — the math

Script: `python tools/sizing.py` (all numbers below come from it).

**Step 1 — calibrate fuel burn to ACC.** Burn scales with hover power, `burn(m) = b₀·(m/M)^1.5` (same disk loading). Fitting b₀ = **80 kg/h at 800 kg** reproduces both sourced points: 248 kg fuel + 177 kg payload → **4.08 h** (source: just under 4 h); 25 kg fuel + 400 kg payload → **19 min** (source: ~20 min). Cross-check: momentum theory for 4 × 2.6 m rotors, FM 0.7 gives ≈ 138 kW hover shaft power at 800 kg, which implies a small-turbine SFC of ~0.6 kg/kWh, plausible `[ASSUMPTION]`.

**Step 2 — hybrid factor.** A heavy-fuel piston/rotary genset with electric drive burns less fuel per kWh than a small turbine. We take **×0.65 → b₀ = 52 kg/h at MTOW** `[ASSUMPTION/VERIFY]` — *this is the number that makes the provider's 5 h possible*; with the turbine's 80 kg/h the same airframe gets ~3.3 h.

**Step 3 — choose tank vs fuel inside 800 kg MTOW** (empty 420 kg = ACC's 375 + 20 genset + 25 for a 5 kWh battery buffer, `[ASSUMPTION]`):

| Water | Fuel | Endurance, loaded the whole flight (to dry) | With 15% fuel reserve |
|---|---|---|---|
| 120 L | 260 kg | 6.7 h | 5.4 h |
| 150 L | 230 kg | 5.7 h | 4.6 h |
| **170 L (chosen)** | **210 kg** | **5.06 h** | **4.14 h** |
| 200 L | 180 kg | 4.2 h | 3.5 h |
| 250 L | 130 kg | 2.9 h | 2.4 h |
| 300 L | 80 kg | 1.7 h | 1.4 h |

170 L ≈ ACC's 177 kg payload point. It is a **conservative rating** (water aboard the whole flight); real cycling drops water and burns less (see §8 for what one fuel load actually lasts). Planned usable flight ≈ 4.1 h `[ASSUMPTION]`; `endurance_s` in the spec is the 5 h rating.

**Step 4 — speeds and stopping.** Cruise **20 m/s** (inside ACC's 70–90 km/h = 19–25 m/s), max 25 m/s, braking 2 m/s² (heavy). Stopping distance: 6 m at 5 m/s, 25 m at 10, **100 m at 20, 156 m at 25.** The 40 m lidar cannot protect cruise → corridors cleared by seekers/DEM, ≥120 m AGL transit, ≤5 m/s below 30 m AGL.

**Step 5 — emergency buffer.** 5 kWh at 138 kW ≈ **2.2 min** of hover power: enough to land, not to continue. `[ASSUMPTION]`

**Step 6 — drops.** 170 L at 10 L/m² (placeholder) = **17 m²** of hotspot per sortie with water, ~51 m² with 3× foam. A 2–4 m² spot fire needs 20–40 L, so **one sortie serves ~4 hotspots** → `Task.volume_l` allows partial drops; `PayloadSystem.release()` returns litres actually released. Drop height 15 m AGL (rotor wash on a 800 kg machine, `[ASSUMPTION]`), release 8 s.

`[OPEN]` Ultra-heavy drones are few: default **2 extinguishers**, 6 seekers.

### 6.3 Seeker sizing (light, electric) and sensor loadouts

**Seeker, electric multirotor `[ASSUMPTION]`:** MTOW 10 kg = 4.0 kg battery (1000 Wh at 250 Wh/kg, 80% usable → 800 Wh) + 1.0 kg avionics + 3.2 kg frame/motors/props + 1.8 kg margin. 4 × 26" props (A = 1.37 m²): hover ≈ **1.02 kW** (momentum theory, FM 0.65, η 0.8) + 48 W avionics; cruise at 15 m/s ≈ 0.85 × hover + 48 W = **0.92 kW** → **~52 min cruise, ~45 min hover.** Ideal coverage 2.5 km²/h → **~2 km² per sortie** (4 × 4 km default grid = 16 km² = 8 sortie-equivalents; 6 seekers cover it in a couple of rotations). Speed 15 m/s (max 20), braking 4 m/s² → stops in 28 m from 15 m/s, 50 m from 20.
`[OPEN]` Option: electric **VTOL fixed-wing** seeker (12 kg, L/D ~10 → ~0.38 kW at 20 m/s → ~2.6 h on 1 kWh) like FireSwarm's VTOL overwatch layer; costs hover-to-confirm ability.

| Item | Seeker (electric, searches) | Extinguisher (hybrid, confirms) | Why / basis |
|---|---|---|---|
| Radiometric LWIR thermal | **640×512**, 32° HFOV (dual thermal+visible optional) | **320×256**, 50° HFOV, nadir | seeker detects; extinguisher only aims/verifies a known hotspot. Boson-class: ~25 g, 1–1.8 W `[S7]` |
| 3D lidar (Livox Mid-360 class) | ✔ obstacle map + water surveys | ✔ **last-mile only** (descent, drop, intake) | 40 m range `[S8]` |
| Flight controller | ✔ | ✔ | Pixhawk class |
| RTK-GNSS | ✔ | ✔ | needs ground-base corrections |
| Range/altitude sensor | **long-range laser altimeter, rated ≥ 100 m** `[VERIFY]` (decision 0001) | **radar altimeter** (calm water returns weakly to lasers `[VERIFY]`) | terrain following, drop height |
| Companion computer | Jetson Orin NX class: detector + map `[S9]` | **smaller** (Orin Nano class): verify, navigation, bus | extinguisher doesn't run search |
| Mesh radio | ✔ | ✔ | |
| ADS-B in | recommended | **✔ required** (big and in the same airspace as manned water bombers) | `[ASSUMPTION]` |
| Power telemetry | battery pack voltage/current/temp | **fuel level/flow, generator output, engine temp, battery buffer** | new in contracts 0.2.0 |
| Tank, valve/dosing, flow/level sensing | — | ✔ | `[OPEN]` mech E |

**What the extinguisher drops vs the seeker:** the 640×512 thermal, the heavy compute, the search detector/mapping role and the dual visible camera. What it keeps: everything it needs to *not die* and to hit a confirmed spot. Avionics: seeker ≈ 1.0 kg / 48 W; extinguisher ≈ 1.2 kg / 35 W (0.15% of MTOW). **Weight is not why we trim the extinguisher's sensors; integration effort, power-fault surface area and "who owns the perception" are.** Sensors map to `SENSOR_NAMES` in `contracts/config.py` (`power` added in 0.2.0).

**About "gas levels":** I read this as **fuel level** (hybrid telemetry, now in `DroneState.fuel_remaining_kg` / `fuel_pct`, plus `generator_output_kw`, `AlertKind.LOW_FUEL`). If you also meant combustion gases (CO etc.), that's the deferred byproduct-sensing idea in §9.

### 6.4 Review notes (what changed after the final review)

- **Stopping distance vs lidar range.** Seeker at 15 m/s, 4 m/s²: 28 m (the 40 m lidar barely covers it on a dark 10% reflectivity target). Extinguisher at 20 m/s, 2 m/s²: **100 m**, lidar can't help. **Rule:** cruise *above* the tallest obstacle (DEM + canopy margin); extinguishers follow seeker-cleared corridors; descend only for drops/intake and slow to ≤5 m/s. `PlatformSpec.stopping_distance_m()` encodes the math.
- **Thin obstacles (wires)** are often missed by lidar `[VERIFY]` → use mapped power lines + altitude margin.
- **Radar correction:** radar is unaffected by darkness; its limit is coarse angular resolution. Real heavy-lift ag drones pair active phased-array radar with binocular vision for obstacle avoidance `[S5]`. Dropping radar for the demo is defensible; consider a **radar altimeter** for hover over water (laser returns from calm water are weak `[VERIFY]`).
- **NIR/SWIR is optional, not baseline.** LWIR already covers night detection of smouldering hotspots; SWIR sees hot flames/embers but is costly and misses cooler hotspots `[VERIFY]`.
- **Dual thermal+visible** is a nice-to-have for the ops view and dusk/daytime pre-survey of water sources.
- **RTK dependency:** corrections come from the ground base; model `gnss_rtk` degradation.
- **Added:** water level/flow sensing (otherwise estimate from valve time × flow rate), wind estimate from the flight controller (needed for drop drift), ADS-B in, health telemetry for the ground base.
- **Hot/ash environment:** hovering a turbine or engine in smoke, ash and updrafts is unproven even for FireSwarm `[S19]`; approach from upwind, keep a standoff `[ASSUMPTION]`, model derating and `POWERPLANT_FAULT` as sim faults.
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
| **Class A foam** | 0.1–1% concentrate `[S10]`; most products work best near 0.3–0.5% `[S10]` | Manufacturer: often 3–5× more effective than water `[S10]`; independent test authors call that claim controversial with limited comparative testing `[S10]` | ✔ strong: USFS says foams rely on their water and are best for direct attack and mop-up `[S10]` | ✔ **only if dosing happens after intake** (onboard proportioner). 170 L × 0.3% = 0.51 L concentrate per fill; a 10 L reservoir ≈ 19 fills. **Caveat:** ACC/FireSwarm use a dip *bucket*, where dosing after intake is hard; we assume an onboard tank `[OPEN]` | US guideline: no aerial foam/retardant within 300 ft (91.44 m) of waterways `[S11]`; Canadian/Alberta rule `[VERIFY]` |
| **Gel / water enhancer** | mix varies; sensitive to water hardness `[S10]` | adhesion, drop accuracy | weak (structure protection more than hotspot knockdown) | ⚠ basin water quality changes consistency | same waterway buffers |
| **Long-term retardant** (ammonium phosphate) | — | stays effective after water evaporates | ✘ fireline building, not spot knockdown | ✘ | toxic to aquatic life `[S11]`; waterway buffers |

**Decisions (`[ASSUMPTION]`):**
- Baseline `ExtinguishingAgent.WATER`; optional `CLASS_A_FOAM` module (needs dosing, sim multiplier 3× as an *assumption*, not a promise); gel and retardant are out of scope (retardant enum deliberately omitted).
- **Basin refilling stays enabled for foam** via onboard dosing: the tank holds plain water until dosing after intake, so foam never contaminates a basin and gel/foam quality isn't hostage to basin water.
- **Waterway rule as an algorithm:** if a hotspot is within `AgentSpec.aquatic_buffer_m` of a water source, fall back to plain water. (Sim hook: `agent_allowed(hotspot, water_sources)`.)
- Sim settings: `ScenarioConfig.agent`, `agent_specs`, `use_water_sources`. `suppression_l_per_m2` is a **PLACEHOLDER (10 L/m²)**: E must derive it (fuel type, burn depth). With a 3× agent, a 3 m² hotspot at 10 L/m² needs 30 L of water or ~10 L of foam solution; one 170 L sortie = 17 m² of hotspot with water, ~51 m² with foam.

---

## 8. Refill logistics and the water-basin idea

**Idea.** Seekers (and/or prior hydrography data) identify natural water bodies (ponds, lakes, rivers) near the fire. Extinguishers refill there instead of flying back to the primary station, shortening the refill leg and raising water delivered per drone-hour. A primary refill station (possibly mobile) remains the guaranteed option. The swarm treats refill points as a *planning problem*: choose the best refill target per sortie.

**Reality check.** FireSwarm drones plan their own water pickups with buckets and carry water from natural and human-made sources `[S1,S2]`; US initial-attack drone fleets refill at a base station by a technician "within minutes" `[S6]`; amphibious drones are pitched for night firefighting precisely because they can refill from lakes `[S16]`.

**Throughput model** (one extinguisher; 170 L tank, cruise 20 m/s, fill 120 s, drop 8 s; `d` = refill-point-to-fire distance; engine idles 8 kg/h on the pad `[ASSUMPTION]`): `cycle = 2d/v + t_fill + t_drop`, `throughput = W / cycle`.

| Refill point | Cycle | Throughput | Fuel per cycle | Litres per kg fuel |
|---|---|---|---|---|
| Station 15 km | 27.1 min | 6.3 L/min (376 L/h) | 14.9 kg | 11 |
| Station 5 km | 10.5 min | 16.2 L/min (975 L/h) | 5.3 kg | 32 |
| Station 5 km + 3 min queue | 13.5 min | 12.6 L/min | 5.3 kg | 32 |
| Basin 1 km | 3.8 min | 44.7 L/min (2,684 L/h) | 1.5 kg | 116 |
| Basin 0.5 km | 3.0 min | 57.3 L/min | 1.0 kg | 172 |

A basin 1 km from the fire beats a station 15 km away by **~7×** in delivery rate.

**Fuel is now the binding constraint, and basins help it too.** Simulating one full fuel load (210 kg, 15% reserve, mass falling as fuel burns, `tools/sizing.py`):

| Refill point | Cycles | Flight hours | Water delivered | L per kg usable fuel |
|---|---|---|---|---|
| Station 15 km | 11 | 5.0 h | 1.9 m³ | 10.5 |
| Station 5 km | 33 | 5.8 h | 5.6 m³ | 31 |
| Basin 1 km | 131 | 8.3 h | 22.3 m³ | 125 |
| Basin 0.5 km | 206 | 10.2 h | 35.0 m³ | 196 |

So a nearby basin delivers about **4×** more water per kg of fuel than a 5 km station. **But basins can't refuel the drone**, so the leader must still schedule station visits for fuel (`RefillRequest.needs_fuel=True` forces a station), and the sim should cap continuous hours with an engine-life/maintenance limit rather than treat 8–10 h as free `[OPEN]`. **Water depletion is negligible:** 100 sorties × 170 L = 17 m³ ≈ 4% of a 20 × 20 × 1 m pond (400 m³); still track `est_volume_l` for tiny sources.

**Discovery pipeline.** Prior hydrography → candidate `WaterSource(origin=PRIOR_MAP, status=CANDIDATE)` → a Seeker flies `SURVEY_WATER` → `VALIDATED` (or `REJECTED`) → leader includes it in refill options → `RefillRequest`/`RefillGrant`. Prior water maps (provincial/national hydrography, OpenStreetMap) mean seekers mostly *validate*, not discover `[VERIFY availability/licence]`. Candidate night cues, all `[VERIFY]` with real data: **LWIR:** water often looks warmer than surrounding land at night; **lidar:** calm water returns weakly (holes in the cloud); **DEM:** flat depressions. Also check **approach clearance** (trees around the shore, corridor aligned with wind) → `approach_clearance_m`.

**Optimization problems (in order of effort):**
1. Per-sortie refill choice: minimize `travel + queue + fill + risk penalty (unvalidated)` over stations and validated sources.
2. Joint extinguisher↔hotspot↔refill allocation (a vehicle-routing problem with intermediate facilities): greedy first, OR-Tools later.
3. Mobile station placement (p-median style) given FIRMS cues / hotspot clusters.
4. Water-survey scheduling: survey candidates near predicted fire zones first (value of information).
5. Station queue/bay management.

**Risks / constraints:** foam contamination (see §7); gel quality vs water hardness; rotor wash on shallow ponds; frozen/dry/seasonal sources; legality of withdrawing water and ecology `[VERIFY Alberta]`; false "water" (wet fields); night hover over water (lidar altitude unreliable → RTK/baro + radar altimeter); **intake method is `[OPEN]` for E**: dip bucket (what Thunder Wasp actually carries: it lowers itself, no winch `[S17]`) vs hover with hose + pump vs other. A bucket changes fill time and makes onboard foam dosing hard. Humans may need to approve new sources → `GroundCommand(ENABLE_WATER_SOURCE/DISABLE_WATER_SOURCE)`.

**Contract hooks:** `WaterSource`, `RefillStation` (now with `fuel_available_kg`, `fuel_rate_kg_per_min`, `battery_swap_s`), `TaskType.SURVEY_WATER` / `REFILL`, `WaterSourceReport`, `RefillRequest` (`fuel_remaining_kg`, `needs_fuel`), `RefillGrant`, `MapSync.water_sources`, `ScenarioConfig.use_water_sources` (feature flag), `AlertKind.WATER_SOURCE_NEW`.

**Metrics:** litres delivered per drone-hour **and per kg fuel**, time to first drop per hotspot, fraction of time in transit vs on target, refill queue wait, water wasted on decoys.

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
- **Perception ML decisions (2026-11-07, workstream C):**
  - **Verifier = small CNN (PRIMARY)** on 32×32 °C patches (2 channels: absolute temperature, local contrast) centred on `ThresholdDetector` proposals, exported to **ONNX** (not pickle). **GBM on hand-made blob features = STRETCH**, same folds, kept only as a comparison. Preprocessing lives once in `perception/patches.py` (versioned).
  - **Two operating points, one model** (decision 0004): SEARCH threshold is **recall-first** (recall ≥ 0.95 on CV, then max precision; a miss costs more than a seeker INVESTIGATE pass); CONFIRM threshold is **precision-first** (precision ≥ 0.98, then max recall; a confirmation can send an extinguisher). F1 is still reported against the >90% criterion; accuracy is shown but not trusted (class imbalance). `[ASSUMPTION]` 0.95 / 0.98 targets.
  - **Scale-normalized patches (PatchSpec v2):** when the GSD is known, each patch is a fixed 2.3 m ground window resampled in radiance. `[VERIFY]` On synthetic data plain crops did as well (decision 0004 A/B), because synthetic decoys separate on temperature alone; re-run on FLAME 3 composites (2026-11-07, issue #4): still a tie (test object F1 0.940 scaled vs 0.934 plain), so v2 stays the default; the synthetic-only model drops to F1 0.70 on real backgrounds while the composite model keeps 0.95 on the night synthetic set (decision 0004 update). Datasets need a GSD per frame (`gsd_m`) or view geometry for oblique frames (`agl_m`, `pitch_deg`, `focal_px`).
  - **Close-range thresholds:** three bands — CONFIRMED above the confirm bar (precision ≥ 0.98, never below search), FALSE_POSITIVE only below a bar keeping recall ≥ 0.99, otherwise inconclusive → escalate to a human (this removed 9/9 wrongly rejected fires on synthetic data).
  - **Close range:** `perception/confirm.py` — `HotspotConfirmer` (INVESTIGATE: multi-frame mean probability vs the confirm threshold) and `SuppressionChecker` (VERIFY after a drop: physics, "anything still ≥ 15 °C above background?", reports SUPPRESSED only).
  - **Hot-blob bypass** (decision 0003): ≥ 3 px at ≥ 0.9 × the current gain mode's saturation, or a blob larger than the patch window and ≥ 150 °C; no bypass near friendly drones or known industrial heat. Never skips confirmation.
  - **Sensor wiring** (decision 0002, no contract change): pose interpolated at exposure time, DEM/altimeter for georef and GSD, wider gates when RTK is lost, frames skipped during FFC or thermal faults (no false "searched, empty" evidence), optional lens distortion.
  - **Evaluation:** group-wise train/test split (`training/make_split.py`); 5-fold StratifiedGroupKFold on train; per-fold thresholds chosen on the other folds; test split used once by `perception/evaluate.py`; patch, object and frame levels reported separately; bootstrap CI over groups. Results go to `perception/RESULTS.md` + `perception/results/*.json` (for the demo).
  - **FLAME 3 labels are threshold-derived** `[SOURCED S13 §III.A.1 p.5, §III.C.1]`:
    - *Image-level Fire / No-Fire* comes from the frame's **max TIFF temperature**: < ~80 °C → no fire, > ~200 °C → fire. Frames in between are "manually reviewed or excluded". The authors call these thresholds "practical heuristics", tuned to their prescribed burns (typically 55–650 °C, §III.B).
    - *Pixel masks:* the paper describes none. It only says TIFFs *enable* masks via direct, Otsu or hysteresis thresholding; S20 reports thresholded masks. Treat any mask as a threshold unless `python -m training.check_masks` says otherwise (it detects global and per-frame cutoffs).
    - **Consequence:** a threshold + CNN pipeline scores optimistically against these labels. The hard 80–200 °C band (warm decoys, smouldering spots) is under-represented. Report F1 **per max-T band** (< 80 / 80–200 / > 200 °C), and don't cite overall FLAME 3 F1 as evidence for the > 90% criterion. Never copy 80/200 °C into `perception/` as constants.
    - **Training data = synthetic fires on real FLAME 3 No-Fire backgrounds** (`python -m training.composite`): exact masks, radiance mixing, real clutter and noise, groups kept. `objects.csv` gives each fire's observed band for per-band F1. Plain synthetic data (`perception/synth.py`) stays the night test set.
    - **Workflow:** `training/train_verifier.ipynb` runs the loop (data → composites → group split → patches → CV with plots → ONNX → test once → hand-labelled band), calling the same modules as the CLI. Laptop is enough (tiny CNN); Colab only needs `patches.npz` (tens of MB). Raw data never goes to GitHub or Drive in bulk.
    - **Hand-labelled hard band (optional test set):** `training/label_tool.py`. A person labels 80–200 °C blobs from the paired RGB photo; Cohen's kappa checks agreement between two labellers.
    - **FLAME 3 cameras** `[SOURCED S13 Table III]`: Autel 640T (Hanna Hammock, Hundred, Wild Bill), M30T (Sycan), M2EA (Shoetank, Willamette), all 640×512. Paper: high gain clips at 150 °C, low gain at 450–550 °C; GSD ≈ 17.6 cm/px at 122 m nadir (M30T).
    - **Sycan CV subset, measured on the download** (`training/FLAME3.md` table; `python -m training.build_manifest_flame3`): 622 Fire + 116 No Fire, captured 15:00–19:23 local (day/dusk). **Every frame is oblique** (gimbal pitch median −15°, range −46…+12°), 55–124 m above takeoff, so GSD changes per row (centre median ≈ 0.32 m/px, 4–14× coarser than our seeker). Geometry comes from the raw RGB's DJI XMP (height, pitch, laser-rangefinder slant range); focal ≈ 696 px. M30T clips measured at **187.4 °C (high gain) / 608.7 °C (low gain)**. **All 84 Fire frames with max T in 80–200 °C are high-gain frames** (67 exactly at the clip), so frame-level "hard band" Sycan data is clipping, not smouldering. File stems are shuffled and restart per class folder: pair by folder + stem, order by EXIF time. Manifests now carry optional `agl_m`, `pitch_deg`, `focal_px`, `saturation_c`, `rgb_path`, `timestamp` (`perception/datasets.py`); patches, composites and evaluation use a per-row GSD on oblique frames.
    - **Approach comparison + triage (2026-11-08):** `training/compare_approaches.ipynb` (sliders) / `python -m training.compare handlabels|manifest` score algorithmic (peak °C), CNN alone, and full pipeline on the same proposals, with FIRE / REVIEW / NOT_FIRE bands (`perception/triage.py`; defaults = sidecar reject/confirm). **Hand-label result (DoqCA, 98 usable of 297, 199 unsure):** no approach separates fire from not-fire in 80–200 °C (AUC CIs include 0.5). At the search threshold the pipeline drops 7/78 labelled fires that the threshold alone keeps, because composites contain almost no hot non-fire examples (test split 80–200 °C: 58 fire vs 3 non-fire). The 20 negatives are suspect (compact 100–180 °C spots, several in smoke). Numbers: `perception/RESULTS.md`. p(fire) is not calibrated (pos_weight); see issue tracker.
    - **FLAME 3 NADIR + mix model (2026-11-08):** NADIR thermal (Hanna Hammock, nadir, 0.13–0.16 m/px, georeferenced stacks; `training/FLAME3.md`) downloaded with `python -m training.download_flame3_nadir` (one Kaggle API call + HTTP Range reads; per-file calls hit 429). **Labels from each pixel's history** (`training/build_nadir.py`): fire = ≥ 120 °C or burned and ≥ 25 °C above its post-peak floor; not fire = unburned, or burnt out at the ~43–45 °C char plateau; rest = ignore (mask value 2, new `perception.datasets.MASK_IGNORE`, honoured by patches, evaluation and comparison). Plot 1 = test, plot 2 = train. **Sycan weak masks** (`training/build_sycan_weak.py`): clipped/≥ 200 °C pixels in Fire frames = fire, No-Fire frames = not fire, rest ignored; hand-label frames never trained on. `perception/models/fire_cnn_mix.onnx` (`mix_v2`) trains on all three with capped mask-centred positives and hard examples from the masks. Pooled test pipeline F1 0.998 (vs 0.952), NADIR CNN recall 0.997 (vs 0.73); hand-band unchanged. Six test views + matrix: `training/eval_sets.py`, `training/compare_approaches.ipynb`, `perception/RESULTS.md` (`matrix_mix_v2`). Caveats: NADIR is daylight; nadir/cvsubset labels only cover clear cases (temperature separates them by construction).
    - HIT-UAV and BIRDSAI [S21] are **8-bit** (HIT-UAV "bit depth is 8"; BIRDSAI real frames are `.jpg`): no °C, so they can't feed the temperature-channel CNN.
    - The Kaggle subset is **one burn**, so groups = day × 250 m ground cell of the image-centre ground point (37 groups; No Fire 14).
    - Citation (requested by the authors) and full notes: `training/FLAME3.md`. Data list: `exclude/DATA_SOURCES.md`; workflow: `exclude/training_README.md` (local, gitignored).

---

## 11. Contracts

**Install/test:** `pip install -e .[dev]` (perception: `.[dev,perception]`, training: `.[dev,train]`) · `python -m pytest -q` (56 contract tests + perception/training tests; the ONNX round-trip test needs `.[train]`, otherwise it is skipped) · `python tools/sync_contracts.py` refreshes the code below. CI (`.github/workflows/tests.yml`) runs the whole suite on every push to `main` and every PR into `main`: keep it green before merging.

**How to use:** build messages with `SwarmMessage.create(sender_id, body, now, seq=…, receiver_id=…)`; publish on `topic_for(msg)`; serialize with `encode()`, parse with `decode()`; match subscriptions with `topic_matches()`; implement `Bus`/`Clock`/`Autopilot`/`PayloadSystem`/`Detector`/`Georeferencer` from `interfaces.py`.

**Differences from the class diagram:**
- Timestamps are **float seconds** (Clock timebase), not `datetime`.
- `SwarmMessage.payload: dict` → a **discriminated union** of typed bodies.
- New: `WaterSource`, `RefillStation` (now core), `TaskType.SURVEY_WATER`, `TaskAnnounce`/`TaskAck`, `LeaderClaim`, `RefillRequest`/`RefillGrant`, `GroundCommand`, `AlertNotice`, `SimFault`, `ScenarioConfig` and platform/agent specs.
- `LOW_BATTERY` became a status/health field plus a derived `ALERT`; `PATH_CLAIM` kept.
- `ImagingMap` is **not** a Pydantic model: it is NumPy layers (log-odds, last-seen). Its wire format is sparse `GridCellUpdate` inside `MapSync`.
- `Sensor.read()` is replaced by `ThermalSensor` / `ObstacleSensor` protocols; frames stay on the drone.
- `HotspotStatus.VERIFIED_OUT` is ground-confirmed only.
- **0.2.0:** `Powertrain` (ELECTRIC/HYBRID); `DroneState` gains `powertrain`, `fuel_remaining_kg`, `fuel_pct`, `generator_output_kw` (HYBRID must report fuel, ELECTRIC must not); `PlatformSpec` gains `search_capable`, electric (`battery_wh`, powers) and hybrid (`fuel_*`, `battery_buffer_wh`) models with `fuel_burn_kg_h()`, `hybrid_endurance_s()`, `electric_endurance_s()`; `PowerSystem` protocol; `Task.volume_l` for partial drops; `RefillStation` fuel/battery-swap services; `RefillRequest.needs_fuel`; `AlertKind.LOW_FUEL`, `POWERPLANT_FAULT`; sensor name `power`; defaults changed to seeker = electric 10 kg, extinguisher = hybrid 800 kg / 170 L; scenario default 6 seekers, 2 extinguishers.

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
```

#### `contracts/enums.py` (171 lines)

```python
"""Enumerations shared by every workstream.

Values are uppercase strings equal to the member name so JSON stays readable.
"""
from enum import StrEnum

__all__ = [
    "DroneKind", "SwarmRole", "DroneStatus",
    "TaskType", "TaskStatus", "HotspotStatus",
    "WaterSourceKind", "WaterSourceStatus", "WaterSourceOrigin", "RefillTargetKind",
    "ExtinguishingAgent", "DetectorKind", "Powertrain",
    "MessageType", "Severity", "AlertKind", "CommandKind", "SimFaultKind",
]


class DroneKind(StrEnum):
    SEEKER = "SEEKER"              # light electric scout: 640x512 thermal + lidar, searches/confirms, surveys water
    EXTINGUISHER = "EXTINGUISHER"  # ultra-heavy hybrid water hauler: drops on seeker-confirmed hotspots, does not search


class Powertrain(StrEnum):
    ELECTRIC = "ELECTRIC"  # battery only (seekers): telemetry is battery_pct
    HYBRID = "HYBRID"      # generator + battery buffer (extinguishers): telemetry adds fuel and generator state


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
    REVIEW_REPORT = "REVIEW_REPORT"
    WATER_SOURCE_REPORT = "WATER_SOURCE_REPORT"
    MAP_SYNC = "MAP_SYNC"
    CLUSTER_UPDATE = "CLUSTER_UPDATE"
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
    LOW_FUEL = "LOW_FUEL"                    # hybrid drones: fuel at/below reserve
    POWERPLANT_FAULT = "POWERPLANT_FAULT"    # generator/engine fault; drone is on battery buffer only
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

#### `contracts/entities.py` (230 lines)

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

from typing import Annotated

from pydantic import Field, model_validator

from .base import Contract, FrozenContract
from .enums import (
    AlertKind, DroneKind, DroneStatus, ExtinguishingAgent, HotspotStatus, Powertrain, Severity, SwarmRole,
    TaskStatus, TaskType, WaterSourceKind, WaterSourceOrigin, WaterSourceStatus,
)
from .geometry import Position, point_in_polygon

__all__ = [
    "ClusterId", "ClusterSnapshot", "DroneState", "Hotspot", "Zone", "Task", "WaterSource", "RefillStation",
    "GridCellUpdate", "Alert",
]

ClusterId = Annotated[str, Field(min_length=1, pattern=r"^[^/#*+\s]+$")]


class DroneState(Contract):
    """Telemetry snapshot. Carried inside HEARTBEAT; the ground base builds its health view from it."""

    drone_id: str
    kind: DroneKind
    role: SwarmRole = SwarmRole.FOLLOWER
    position: Position
    heading_deg: float = Field(default=0.0, ge=0, lt=360)
    speed_mps: float = Field(default=0.0, ge=0)
    powertrain: Powertrain = Powertrain.ELECTRIC
    battery_pct: float = Field(ge=0, le=100)  # ELECTRIC: the main pack. HYBRID: the buffer (emergency landing reserve)
    endurance_remaining_s: float | None = Field(default=None, ge=0)  # drone's own estimate (fuel or battery)
    # HYBRID only (required for HYBRID, forbidden for ELECTRIC): fuel is what limits an extinguisher sortie.
    fuel_remaining_kg: float | None = Field(default=None, ge=0)
    fuel_pct: float | None = Field(default=None, ge=0, le=100)
    generator_output_kw: float | None = Field(default=None, ge=0)
    water_remaining_l: float | None = Field(default=None, ge=0)      # EXTINGUISHER only
    agent: ExtinguishingAgent | None = None                          # what the tank currently holds
    status: DroneStatus = DroneStatus.IDLE
    current_task_id: str | None = None
    leader_id: str | None = None   # who this drone currently believes is leader
    epoch: int = Field(default=0, ge=0)  # leader epoch this drone believes in
    cluster_id: ClusterId | None = None
    cluster_epoch: int = Field(default=0, ge=0)
    # Health flags keyed by sensor/subsystem name, e.g. {"thermal": True, "lidar": True,
    # "gnss_rtk": False, "comms": True, "payload": True}. Names are defined in config.py (SENSOR_NAMES).
    sensors_ok: dict[str, bool] = Field(default_factory=dict)
    link_quality: float | None = Field(default=None, ge=0, le=1)
    timestamp: float

    @model_validator(mode="after")
    def _powertrain_fields(self) -> "DroneState":
        fuel = (self.fuel_remaining_kg, self.fuel_pct)
        if self.powertrain is Powertrain.HYBRID and any(f is None for f in fuel):
            raise ValueError("HYBRID drones must report fuel_remaining_kg and fuel_pct")
        if self.powertrain is Powertrain.ELECTRIC and (any(f is not None for f in fuel)
                                                       or self.generator_output_kw is not None):
            raise ValueError("ELECTRIC drones must not report fuel or generator fields")
        return self


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
    cluster_id: ClusterId | None = None

    def contains(self, pos: Position) -> bool:
        return point_in_polygon(pos.x, pos.y, self.boundary)


class ClusterSnapshot(FrozenContract):
    """Versioned membership and zone ownership view published by a cluster leader."""

    cluster_id: ClusterId
    leader_id: str = Field(min_length=1)
    cluster_epoch: int = Field(ge=0)
    member_ids: list[str] = Field(min_length=1)
    zone_ids: list[str] = Field(default_factory=list)
    updated_at: float = Field(allow_inf_nan=False)
    lease_until: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _membership_and_lease_valid(self) -> "ClusterSnapshot":
        if any(not member_id for member_id in self.member_ids):
            raise ValueError("member_ids must not contain empty IDs")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("member_ids must be unique")
        if self.leader_id not in self.member_ids:
            raise ValueError("leader_id must be included in member_ids")
        if any(not zone_id for zone_id in self.zone_ids):
            raise ValueError("zone_ids must not contain empty IDs")
        if len(set(self.zone_ids)) != len(self.zone_ids):
            raise ValueError("zone_ids must be unique")
        if self.lease_until <= self.updated_at:
            raise ValueError("lease_until must be later than updated_at")
        return self


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
    cluster_id: ClusterId | None = None
    cluster_epoch: int = Field(default=0, ge=0)
    # EXTINGUISH only: litres (of the loaded agent) to release on this hotspot. One sortie can serve several
    # hotspots because the tank is much bigger than a small spot fire needs. None = "all you have".
    volume_l: float | None = Field(default=None, gt=0)

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
    # HYBRID drones can only refuel at a station (never at a water source). 0 = no fuel service here.
    fuel_available_kg: float = Field(default=0.0, ge=0)
    fuel_rate_kg_per_min: float = Field(default=20.0, gt=0)
    battery_swap_s: float | None = Field(default=None, gt=0)  # None = no swap service (seeker packs)

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

#### `contracts/messages.py` (334 lines)

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
from .entities import Alert, ClusterId, ClusterSnapshot, DroneState, GridCellUpdate, Hotspot, Task, WaterSource
from .enums import (
    CommandKind, ExtinguishingAgent, HotspotStatus, MessageType, RefillTargetKind, SimFaultKind,
)
from .geometry import GridSpec, Position

__all__ = [
    "Heartbeat", "HotspotReport", "ReviewReport", "WaterSourceReport", "MapSync", "ClusterUpdate", "TaskAnnounce", "TaskBid",
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


class ReviewReport(_Body):
    """A tracked candidate requiring human review at the ground base."""

    type: Literal[MessageType.REVIEW_REPORT] = MessageType.REVIEW_REPORT
    review_id: str = Field(min_length=1)
    hotspot: Hotspot
    score: float = Field(ge=0, le=1)
    score_source: Literal["hotspot_confidence"] = "hotspot_confidence"
    not_fire_below: float = Field(ge=0, le=1)
    fire_at_or_above: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _review_band(self) -> "ReviewReport":
        if self.not_fire_below > self.fire_at_or_above:
            raise ValueError("not_fire_below must be <= fire_at_or_above")
        if not self.not_fire_below <= self.score < self.fire_at_or_above:
            raise ValueError("score must fall inside the REVIEW band")
        return self


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


class ClusterUpdate(_Body):
    """Cluster membership, leadership term, and zone ownership snapshot."""

    type: Literal[MessageType.CLUSTER_UPDATE] = MessageType.CLUSTER_UPDATE
    cluster: ClusterSnapshot


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
    cluster_id: ClusterId | None = None
    cluster_epoch: int = Field(default=0, ge=0)


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
    cluster_id: ClusterId | None = None
    cluster_epoch: int = Field(default=0, ge=0)


class RefillRequest(_Body):
    type: Literal[MessageType.REFILL_REQUEST] = MessageType.REFILL_REQUEST
    drone_id: str
    position: Position
    water_remaining_l: float = Field(ge=0)
    agent: ExtinguishingAgent = ExtinguishingAgent.WATER
    fuel_remaining_kg: float | None = Field(default=None, ge=0)  # HYBRID: lets the leader decide station vs basin
    needs_fuel: bool = False   # True forces a station (basins cannot refuel)


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
        Heartbeat, HotspotReport, ReviewReport, WaterSourceReport, MapSync, ClusterUpdate, TaskAnnounce, TaskBid, TaskAssign,
        TaskAck, TaskComplete, PathClaim, LeaderClaim, RefillRequest, RefillGrant, GroundCommand,
        AlertNotice, SimFault,
    ],
    Field(discriminator="type"),
]

BODY_TYPES: dict[MessageType, type[_Body]] = {
    MessageType.HEARTBEAT: Heartbeat,
    MessageType.HOTSPOT_REPORT: HotspotReport,
    MessageType.REVIEW_REPORT: ReviewReport,
    MessageType.WATER_SOURCE_REPORT: WaterSourceReport,
    MessageType.MAP_SYNC: MapSync,
    MessageType.CLUSTER_UPDATE: ClusterUpdate,
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
    MessageType.REVIEW_REPORT: QoSProfile(reliable=True, ttl_s=300),
    MessageType.WATER_SOURCE_REPORT: QoSProfile(reliable=True, ttl_s=60),
    MessageType.MAP_SYNC: QoSProfile(reliable=False, ttl_s=10, rate_hz=1),
    MessageType.CLUSTER_UPDATE: QoSProfile(reliable=True, ttl_s=15),
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

#### `contracts/topics.py` (132 lines)

```python
"""Topic names and matching rules.

Convention (same idea works for ROS 2 namespaces, MQTT topics, or the in-process bus):
  drone/{id}/...   per-drone channels            ground/...   ground base channels
  swarm/...        shared swarm channels         sim/...      simulator-only channels

Wildcards in SUBSCRIPTION patterns:
  *  matches exactly one segment      #  matches the rest (zero or more segments), must be last
"""
from __future__ import annotations

from .enums import MessageType
from .messages import ClusterUpdate, LeaderClaim, SwarmMessage, TaskAnnounce, TaskBid

__all__ = [
    "ROUTES", "topic_for", "topic_matches", "validate_pattern",
    "SUB_ALL_HEARTBEATS", "SUB_ALL_HOTSPOTS", "SUB_ALL_WATER_SOURCES", "SUB_ALL_SWARM",
    "SUB_ALL_CLUSTERS", "SUB_GROUND_COMMANDS", "SUB_GROUND_ALERTS", "SUB_GROUND_REVIEWS", "SUB_SIM_CONTROL",
    "sub_cluster", "sub_my_commands",
]

# Publish routes. {sender}/{receiver} are filled from the message envelope.
ROUTES: dict[MessageType, str] = {
    MessageType.HEARTBEAT: "drone/{sender}/heartbeat",
    MessageType.HOTSPOT_REPORT: "drone/{sender}/hotspots",
    MessageType.REVIEW_REPORT: "ground/reviews",
    MessageType.WATER_SOURCE_REPORT: "drone/{sender}/water_sources",
    MessageType.MAP_SYNC: "swarm/map_sync",
    MessageType.CLUSTER_UPDATE: "swarm/clusters/{cluster_id}/state",
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
SUB_ALL_CLUSTERS = "swarm/clusters/*/#"
SUB_GROUND_COMMANDS = "ground/commands"           # every drone (filters on target_id)
SUB_GROUND_ALERTS = "ground/alerts"               # ground base UI
SUB_GROUND_REVIEWS = "ground/reviews"             # ground base operator review queue
SUB_SIM_CONTROL = "sim/control"                   # simulator (and sim-aware drone wrappers)


def sub_my_commands(drone_id: str) -> str:
    """Direct commands to one drone (TASK_ASSIGN, REFILL_GRANT)."""
    return f"drone/{drone_id}/cmd"


def sub_cluster(cluster_id: str) -> str:
    """Subscribe to all coordination topics for one cluster."""
    _validate_cluster_id(cluster_id)
    return f"swarm/clusters/{cluster_id}/#"


def _validate_cluster_id(cluster_id: str) -> None:
    invalid_character = any(
        char in cluster_id for char in "/#*+"
    ) or any(char.isspace() for char in cluster_id)
    if not cluster_id or invalid_character:
        raise ValueError(
            "cluster_id must be a non-empty topic segment without wildcards or whitespace"
        )


def _cluster_id_for(msg: SwarmMessage) -> str | None:
    body = msg.body
    if isinstance(body, ClusterUpdate):
        return body.cluster.cluster_id
    if isinstance(body, LeaderClaim):
        return body.cluster_id
    if isinstance(body, TaskAnnounce):
        return body.task.cluster_id
    if isinstance(body, TaskBid):
        return body.cluster_id
    return None


def topic_for(msg: SwarmMessage) -> str:
    """Topic a message should be published on. Directed messages require receiver_id."""
    pattern = ROUTES[msg.msg_type]
    cluster_id = _cluster_id_for(msg)
    if cluster_id is not None:
        _validate_cluster_id(cluster_id)
        if msg.msg_type is MessageType.LEADER_CLAIM:
            pattern = "swarm/clusters/{cluster_id}/leader"
        elif msg.msg_type is MessageType.TASK_ANNOUNCE:
            pattern = "swarm/clusters/{cluster_id}/tasks/announce"
        elif msg.msg_type is MessageType.TASK_BID:
            pattern = "swarm/clusters/{cluster_id}/tasks/bids"
    elif msg.msg_type is MessageType.CLUSTER_UPDATE:
        raise ValueError("CLUSTER_UPDATE requires a cluster_id")
    if "{receiver}" in pattern and not msg.receiver_id:
        raise ValueError(f"{msg.msg_type} requires receiver_id")
    return pattern.format(
        sender=msg.sender_id,
        receiver=msg.receiver_id,
        cluster_id=cluster_id,
    )


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

#### `contracts/interfaces.py` (119 lines)

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
    "PowerSystem", "Detector", "Georeferencer", "ThermalSensor", "ObstacleSensor",
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
    """Water/agent tank on an EXTINGUISHER. `release` supports partial drops (several hotspots per sortie)."""

    def water_remaining_l(self) -> float: ...
    def agent(self) -> ExtinguishingAgent: ...
    def release(self, volume_l: float, at: Position) -> float:
        """Drop up to volume_l at `at`; returns litres actually released."""
        ...
    def fill(self, volume_l: float, agent: ExtinguishingAgent = ExtinguishingAgent.WATER) -> float:
        """Add up to volume_l (capped by capacity); returns litres actually added."""
        ...


@runtime_checkable
class PowerSystem(Protocol):
    """Energy state. ELECTRIC: pack only. HYBRID: fuel + generator + battery buffer (fuel_* are None when ELECTRIC)."""

    def battery_pct(self) -> float: ...
    def fuel_remaining_kg(self) -> float | None: ...
    def generator_output_kw(self) -> float | None: ...
    def endurance_remaining_s(self) -> float:
        """Estimate at the CURRENT mass and flight mode; must account for the fuel/battery reserve."""
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

#### `contracts/config.py` (342 lines)

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
    DroneKind, ExtinguishingAgent, Powertrain, WaterSourceKind, WaterSourceOrigin, WaterSourceStatus,
)
from .geometry import GeoOrigin, GridSpec, Position
from .sensing import CameraModel

__all__ = [
    "SENSOR_NAMES", "ThermalCameraSpec", "LidarSpec", "CommsSpec", "PlatformSpec", "AgentSpec",
    "FireSpec", "DecoySpec", "WindSpec", "ScenarioConfig", "DEFAULT_AGENT_SPECS",
    "default_seeker_spec", "default_extinguisher_spec", "example_scenario",
]

# Keys used in DroneState.sensors_ok and SimFault(DEGRADE_SENSOR).
# "power" = battery pack / fuel system / generator health (both powertrains). "payload" = tank/valve (extinguisher).
SENSOR_NAMES = ("thermal", "lidar", "gnss_rtk", "rangefinder", "comms", "compute", "payload", "power")


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
    """One airframe + powertrain + sensor set. See context.md section 6 for the math behind the defaults.

    ELECTRIC (seekers): energy = battery_wh * battery_usable_frac; power from hover_power_w / cruise_power_w.
    HYBRID (extinguishers): fuel + generator, with a small battery BUFFER for peaks and emergency landing.
        Fuel burn follows burn(m) = fuel_burn_kg_h_at_mtow * (m / max_takeoff_kg)^1.5 (hover-power scaling at
        constant disk loading), calibrated to ACC Thunder Wasp GT figures (context.md section 6.2).
    """

    kind: DroneKind
    name: str
    powertrain: Powertrain = Powertrain.ELECTRIC
    search_capable: bool = True          # False = does not run area search; thermal is for aim/verify only
    empty_mass_kg: float = Field(gt=0)   # includes battery (ELECTRIC) or engine + generator + buffer battery (HYBRID); EXCLUDES fuel
    max_takeoff_kg: float = Field(gt=0)
    water_capacity_l: float = Field(default=0.0, ge=0)  # 1 L of water ~ 1 kg
    cruise_speed_mps: float = Field(gt=0)
    max_speed_mps: float = Field(gt=0)
    max_accel_mps2: float = Field(default=3.0, gt=0)    # also used as braking decel for stopping distance
    endurance_s: float = Field(gt=0)                    # RATED, at full mass, to empty (no reserve); see methods below
    search_altitude_agl_m: float = Field(gt=0)          # extinguishers: transit altitude (they do not search)
    drop_height_agl_m: float = Field(default=10.0, gt=0)
    drop_release_s: float = Field(default=8.0, gt=0)    # time to empty the tank
    refill_time_s: float = Field(default=120.0, gt=0)   # hook-up + fill (+ dosing)
    avionics_mass_kg: float = Field(gt=0)
    avionics_power_w: float = Field(gt=0)
    thermal: ThermalCameraSpec
    lidar: LidarSpec | None = None
    has_rtk_gnss: bool = True
    sensor_names: list[str] = Field(default_factory=lambda: list(SENSOR_NAMES))

    # --- ELECTRIC energy model ---
    battery_wh: float | None = Field(default=None, gt=0)
    battery_usable_frac: float = Field(default=0.8, gt=0, le=1)
    hover_power_w: float | None = Field(default=None, gt=0)   # total electric incl. avionics
    cruise_power_w: float | None = Field(default=None, gt=0)  # total electric incl. avionics

    # --- HYBRID energy model ---
    fuel_capacity_kg: float | None = Field(default=None, gt=0)
    fuel_density_kg_per_l: float = Field(default=0.8, gt=0)   # Jet A-1 / diesel class
    fuel_reserve_frac: float = Field(default=0.15, ge=0, lt=1)  # never plan below this fraction of capacity
    fuel_burn_kg_h_at_mtow: float | None = Field(default=None, gt=0)
    battery_buffer_wh: float | None = Field(default=None, gt=0)  # emergency-landing / peak buffer

    @model_validator(mode="after")
    def _physics_sane(self) -> "PlatformSpec":
        fuel = self.fuel_capacity_kg or 0.0
        if self.empty_mass_kg + self.water_capacity_l + fuel > self.max_takeoff_kg + 1e-9:
            raise ValueError("empty_mass_kg + water_capacity_l + fuel_capacity_kg exceeds max_takeoff_kg")
        if self.cruise_speed_mps > self.max_speed_mps:
            raise ValueError("cruise_speed_mps exceeds max_speed_mps")
        unknown = set(self.sensor_names) - set(SENSOR_NAMES)
        if unknown:
            raise ValueError(f"unknown sensor names: {sorted(unknown)}")
        if self.powertrain is Powertrain.ELECTRIC:
            if None in (self.battery_wh, self.hover_power_w, self.cruise_power_w):
                raise ValueError("ELECTRIC needs battery_wh, hover_power_w, cruise_power_w")
            if self.fuel_capacity_kg is not None or self.fuel_burn_kg_h_at_mtow is not None:
                raise ValueError("ELECTRIC must not define fuel fields")
        else:
            if None in (self.fuel_capacity_kg, self.fuel_burn_kg_h_at_mtow, self.battery_buffer_wh):
                raise ValueError("HYBRID needs fuel_capacity_kg, fuel_burn_kg_h_at_mtow, battery_buffer_wh")
        return self

    def stopping_distance_m(self, speed_mps: float) -> float:
        """v^2 / (2a): distance needed to stop from `speed_mps` at max_accel_mps2."""
        return speed_mps ** 2 / (2 * self.max_accel_mps2)

    # ---- ELECTRIC helpers ----
    def electric_endurance_s(self, hover: bool = False) -> float:
        if self.powertrain is not Powertrain.ELECTRIC:
            raise ValueError("electric_endurance_s is for ELECTRIC platforms")
        power = self.hover_power_w if hover else self.cruise_power_w
        return self.battery_wh * self.battery_usable_frac / power * 3600.0

    # ---- HYBRID helpers ----
    def fuel_burn_kg_h(self, mass_kg: float) -> float:
        """Burn rate at total mass `mass_kg` (hover-power scaling, exponent 1.5)."""
        if self.powertrain is not Powertrain.HYBRID:
            raise ValueError("fuel_burn_kg_h is for HYBRID platforms")
        return self.fuel_burn_kg_h_at_mtow * (mass_kg / self.max_takeoff_kg) ** 1.5

    def hybrid_endurance_s(self, water_l: float | None = None, fuel_kg: float | None = None,
                           reserve: bool = False) -> float:
        """Closed-form flight time while carrying `water_l` the whole time, burning `fuel_kg` (default: full tank).

        With dm/dt = -b0 (m/M)^1.5 the substitution u = m^-0.5 gives u(t) = u0 + 0.5 b0 M^-1.5 t, hence
        t = 2 M^1.5 (m1^-0.5 - m0^-0.5) / b0. `reserve=True` stops at fuel_reserve_frac of capacity.
        Conservative: real sorties drop water, so mass (and burn) falls.
        """
        if self.powertrain is not Powertrain.HYBRID:
            raise ValueError("hybrid_endurance_s is for HYBRID platforms")
        w = self.water_capacity_l if water_l is None else water_l
        f0 = self.fuel_capacity_kg if fuel_kg is None else fuel_kg
        f1 = self.fuel_capacity_kg * self.fuel_reserve_frac if reserve else 0.0
        m0 = self.empty_mass_kg + w + f0
        m1 = self.empty_mass_kg + w + f1
        b0 = self.fuel_burn_kg_h_at_mtow
        hours = 2.0 * self.max_takeoff_kg ** 1.5 * (m1 ** -0.5 - m0 ** -0.5) / b0
        return hours * 3600.0


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
    """Light electric multirotor scout. ASSUMPTIONS (context.md 6.3): 10 kg MTOW, 4 kg pack at 250 Wh/kg,
    4 x 26" props. Hover ~1.07 kW, cruise ~0.92 kW (both incl. 48 W avionics) -> ~52 min. The option of a
    VTOL fixed-wing seeker (~2.5 h) is an [OPEN] question in context.md."""
    return PlatformSpec(
        kind=DroneKind.SEEKER, name="Seeker (electric multirotor, 10 kg)", powertrain=Powertrain.ELECTRIC,
        search_capable=True,
        empty_mass_kg=10.0, max_takeoff_kg=10.0, water_capacity_l=0.0,   # empty includes 4 kg battery + 1 kg avionics
        cruise_speed_mps=15.0, max_speed_mps=20.0, max_accel_mps2=4.0, endurance_s=3150.0,
        search_altitude_agl_m=80.0, avionics_mass_kg=1.0, avionics_power_w=48.0,
        thermal=ThermalCameraSpec(camera=CameraModel(width_px=640, height_px=512, hfov_deg=32.0)),
        lidar=LidarSpec(),
        sensor_names=["thermal", "lidar", "gnss_rtk", "rangefinder", "comms", "compute", "power"],
        battery_wh=1000.0, battery_usable_frac=0.8, hover_power_w=1070.0, cruise_power_w=915.0,
    )


def default_extinguisher_spec() -> PlatformSpec:
    """Ultra-heavy hybrid, Thunder-Wasp-GT class. SOURCED anchors: MTOW 800 kg, empty 375 kg, up to 248 kg fuel,
    400 kg payload, cruise 70-90 km/h (ACC dossier). ASSUMED: +45 kg for hybrid drivetrain and 5 kWh buffer
    (empty 420), tank 170 L, fuel 210 kg, burn 52 kg/h at MTOW (80 kg/h turbine fitted to ACC x 0.65 genset
    factor). Gives ~5.1 h to dry / ~4.1 h with 15% reserve at full mass (hybrid_endurance_s())."""
    return PlatformSpec(
        kind=DroneKind.EXTINGUISHER, name="Extinguisher (hybrid ultra-heavy, 170 L)", powertrain=Powertrain.HYBRID,
        search_capable=False,
        empty_mass_kg=420.0, max_takeoff_kg=800.0, water_capacity_l=170.0,
        cruise_speed_mps=20.0, max_speed_mps=25.0, max_accel_mps2=2.0, endurance_s=18000.0,
        search_altitude_agl_m=120.0,   # transit altitude: above canopy + margin; stopping dist at 20 m/s is 100 m
        drop_height_agl_m=15.0, drop_release_s=8.0, refill_time_s=120.0,
        avionics_mass_kg=1.2, avionics_power_w=35.0,
        thermal=ThermalCameraSpec(camera=CameraModel(width_px=320, height_px=256, hfov_deg=50.0)),
        lidar=LidarSpec(),   # last-mile only: 40 m range cannot protect a 20 m/s heavy drone (100 m to stop)
        sensor_names=["thermal", "lidar", "gnss_rtk", "rangefinder", "comms", "compute", "payload", "power"],
        fuel_capacity_kg=210.0, fuel_density_kg_per_l=0.8, fuel_reserve_frac=0.15,
        fuel_burn_kg_h_at_mtow=52.0, battery_buffer_wh=5000.0,
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

    n_seekers: int = Field(default=6, ge=0)        # ~52 min sorties, so plan for battery swaps
    n_extinguishers: int = Field(default=2, ge=0)  # ultra-heavy: few, big, fuel-limited
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
        if self.seeker_spec.kind is not DroneKind.SEEKER or self.extinguisher_spec.kind is not DroneKind.EXTINGUISHER:
            raise ValueError("seeker_spec/extinguisher_spec have the wrong DroneKind")
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
                          fill_rate_lpm=300.0, total_bays=2, free_bays=2,
                          fuel_available_kg=1500.0, fuel_rate_kg_per_min=30.0, battery_swap_s=120.0,
                          agents_available=[ExtinguishingAgent.WATER, ExtinguishingAgent.CLASS_A_FOAM]),
        ],
    )
```

<!-- CONTRACTS:END -->

---

## 12. Open questions (for the human)

1. ~~Payload class~~ **Resolved 2026-11-07:** ultra-heavy hybrid extinguisher (Thunder-Wasp class), light electric seekers.
1b. `[OPEN]` **Seeker airframe:** electric multirotor (default, can hover to confirm, ~52 min) vs electric VTOL fixed-wing (~2.5 h, no hover).
1c. `[VERIFY]` **Hybrid factor ×0.65 / 5 h rating:** is a heavy-fuel genset really ~35% more efficient than a small turbine? If not, endurance is ~3.3 h with 170 L, or shrink the tank to ~120 L for 5+ h (table in §6.2).
2. `[OPEN]` **Leaders:** one global leader (+standby) or cluster leaders per area?
3. `[OPEN]` **Human gate:** require ground approval before the first drop, or only before new water sources?
4. `[OPEN]` **Scenario site and area:** default is a 4 km × 4 km grid at 2 m cells around a placeholder central-Alberta origin.
5. `[OPEN]` **Sim engine owner** (see §2) and **evaluation harness owner**.
6. `[OPEN]` **Intake method** for water bodies (bucket vs hose + pump vs onboard tank) and whether foam dosing is realistic with a bucket; engine-life limit on continuous hours.
7. `[VERIFY]` Alberta/Canadian rules on drawing water from natural sources and on foam/retardant near waterways.
8. `[VERIFY]` Export/availability limits on thermal cameras (frame rate, licensing) for any real hardware.
10. ~~Seeker search altitude vs 40 m lidar~~ **Resolved 2026-11-07: Option A** (keep 80 m; lidar = flight-level obstacle sensor; add a ≥ 100 m altimeter). Alternatives C/D/E on file: `docs/decisions/0001-…`.
11. `[VERIFY]` (E) **Thermal saturation:** `ThermalCameraSpec.saturation_temp_c = 550` but Boson+ low gain is listed up to ~350 °C (original Boson ~500 °C; high gain ~140–150 °C) [S23]. Smouldering peaks are 450–700 °C [S24], so fire cores may saturate. Pick the camera and gain mode, then set the value. For comparison, FLAME 3's 640×512 drone cameras saturate at 450–550 °C in low gain and 150 °C in high gain [S13 Table III].
9. `[OPEN]` **Perception targets:** confirm recall ≥ 0.95 (search) and precision ≥ 0.98 (confirm) as operating points, and the hot-blob bypass parameters (decision 0003). FLAME 3 labels are threshold-derived (§10). Who hand-labels a small 80–200 °C test set so the hard band has independent labels? Has someone emailed the FLAME 3 authors for the full 6-burn set?

---

## 13. Changelog

| Date | By | Change | Why |
|---|---|---|---|
| 2026-11-09 | sim chat (DEMO mode) | v0.3.7: swarm brain in `core/` (fire priority = confidence × projected area × heat weight; max-heap queue; refill-aware ETA allocation drone → best refill point → fire; VERIFY-hot re-queue; equal-time column × band seeker zones; idle drones pre-position at refill points) replaces `sim/brain_stub` for the swarm; sim: bigger arena (r 220 m, 20 drones), DEMO mode (r 450 m, 8 seekers + 40 field-refill attackers ~15 L foam, 6 lakes as `WaterSource` refill points vs dry basins on the same layout, 2 min, auto camera), `sim/attacker.py`, lakes in `static.py`, pause / mouse / orbit fixes in `frontend/`; `tests/core/` + `tests/sim/test_demo.py`; package `core*`; no contract changes (staging uses `Task(REFILL)` + `target_*_id`). Details: `sim/README.md` | team ask: solve the stubbed swarm brain, refills from lakes, cluster-scale demo |
| 2026-11-08 | sim chat (integration) | v0.3.6: added `sim/` (Python game/sim server: FLAME 3 NADIR thermal maps, fire CA, thermal renderer, SimBus, SimAutopilot/SimPayload, real perception + autonomy in the loop, swarm-brain STUBS in `sim/brain_stub.py`, shadow-run scoring) and `frontend/` (three.js + TS game client, pnpm); `tests/sim/` (6 tests); `pyproject` extra `[sim]`, packages `autonomy*`/`sim*`; no contract changes. Details: `sim/README.md`, `exclude/simulator_task.md` | demo: human + bots vs autonomous swarm on the same seed |
| 2026-11-08 | perception / ground-base integration | v0.3.6 / contracts 0.3.1: add reliable `REVIEW_REPORT` routing to `ground/reviews`, wire tracked-hotspot triage through `PerceptionReporter`, and add a deduplicating ground-base review queue | Ensure REVIEW-band candidates reach an operator queue |
| 2026-11-08 | merge (C) | v0.3.5: merged `main` into `DoqCA/flame3-geometry` (contracts 0.3.0 cluster coordination, per-drone autonomy, CBS deconfliction, mission planning); resolved `.gitignore` (main's layout + `.env*`) and this header/changelog; no contract changes on this side | keep the perception branch current with main |
| 2026-11-08 | perception chat (C) | v0.3.4: FLAME 3 NADIR downloaded + history-labelled (`build_nadir.py`), Sycan weak masks (`build_sycan_weak.py`), ignore masks (`MASK_IGNORE`) through datasets/evaluation/patches/evaluate/compare, hard-example sampling + GT cap in `patch_data.py`/`train_cnn.py`, point-level NADIR view, six test views + matrix CLI (`eval_sets.py`), notebook rebuilt (models × views), `fire_cnn_mix.onnx` (mix_v2) recommended; `imagecodecs` in `[train]`; 120 tests; no contract changes | user asked to train/test on NADIR and see all test combinations |
| 2026-11-08 | perception chat (C) | v0.3.3: `perception/triage.py` (FIRE/REVIEW/NOT_FIRE bands), `training/compare.py` + `compare_plots.py` + `compare_approaches.ipynb` (three approaches, threshold sliders, ROC/PR, sweep heatmaps, reliability, error gallery); hand-label comparison in RESULTS.md; `ipywidgets` in `[train]`; 114 tests; no contract changes | user asked where to compare model / algorithmic / pipeline and try 3-band thresholds |
| 2026-11-08 | cluster coordination | Contracts 0.3.0: added cluster membership/lease snapshots, optional cluster identity and term fields, cluster update messages, and cluster-scoped routing for updates, leader claims, task announcements, and bids | Establish additive contract support for cluster-level swarm coordination |
| 2026-11-07 | FLAME 3 data chat (C) | v0.3.2: `build_manifest_flame3` rewritten for the shipped Sycan layout (oblique geometry from DJI XMP, gain modes 187.4/608.7 °C, day × 250 m groups); per-row GSD in datasets/patches/composite/evaluate; label tool pairs RGB from the manifest, skips clipped blobs; issue #4 A/B re-run on FLAME 3 composites (tie, v2 kept); `perception/models/fire_cnn_flame3comp.onnx`; 108 tests; no contract changes | data now downloaded; old builder was a pre-download guess |
| 2026-11-07 | FLAME 3 / CI chat | v0.3.1: §10 FLAME 3 `[VERIFY]` resolved from the paper (labels = max-T heuristic + manual review of 80–200 °C; no masks described); `training/FLAME3.md` with the requested citation; `training/composite.py` (synthetic fires on real No-Fire backgrounds, 6 tests); `training/label_tool.py` (hand-label hard band, 4 tests); `training/train_verifier.ipynb` (whole training loop); FLAME 3 cameras/gain/GSD from Table III; matplotlib + ipykernel in `[train]`; HIT-UAV/BIRDSAI ruled out (8-bit); CI workflow `.github/workflows/tests.yml` (§3, §5, §11); §12 item 9 reworded; no contract changes | user asked whether FLAME 3 thresholds its labels; dataset authors request citation; CI for main |
| 2026-11-07 | perception chat (C) | decisions 0001 (ACCEPTED A) – 0004 in `docs/decisions/`; PatchSpec v2 scale-normalized patches; confirm/verify stage (`perception/confirm.py`, `evaluate_confirm.py`); precision-first confirm threshold; hot-blob bypass rule replaces 300 °C; sensor wiring (`frame_context.py`, `terrain.py`, `distortion.py`); seeker altimeter in §6.3; no contract changes | user decisions on altitude, bypass, sensors, confirm stage |
| 2026-11-07 | perception chat (C) | added `training/check_masks.py` (mask audit), decision doc 0001 (altitude vs lidar, PROPOSED), §12 items 10–11 (altimeter gap, Boson+ saturation 350 °C vs config 550 °C) | follow-up questions on masks, lidar, hot-bypass |
| 2026-11-07 | perception chat (C) | v0.3: added `perception/` (threshold, FUSED CNN verifier, fake detector, georef, log-odds map, tracker, metrics, synthetic radiometric frames, pipeline evaluator) and `training/` (group CV, recall-first threshold, ONNX export; GBM = stretch), `exclude/` gitignored; §2/§3/§5/§10/§12 updated; no contract changes (77 tests) | perception workstream kickoff |
| 2026-11-07 | swarm workstream | Added bus-backed leader failover/hold coordination, task ACK/completion lifecycle management, and nearest-compatible refill assignment | Close the orchestration gaps in workstream A while keeping transport simulation and flight actions behind their existing interfaces |
| 2026-11-07 | swarm workstream | Implemented scored failover election, weighted hotspot priorities, deadline-bounded SSI rounds with normalized priority bids, bus messaging, and multi-heartbeat failure handling in `swarm/` | Start workstream A against the v0.2 contracts |
| 2026-11-07 | swarm workstream | Added status-based hotspot-to-task creation and a clock-driven bus auction coordinator/bid responder | Complete swarm-owned task creation and bus auction flow against the existing contracts |
| 2026-11-07 | follow-up chat | v0.2 / contracts 0.2.0: ultra-heavy hybrid extinguisher (800 kg MTOW, 170 L, 210 kg fuel), electric 10 kg seeker, trimmed extinguisher sensors, fuel/generator telemetry, refuel in refill, partial drops, sizing script; basin math redone with fuel (56 tests) | team decision: FireSwarm-style extinguisher, electric seekers |
| 2026-11-07 | initial chat | Created context v0.1 and contracts 0.1.0 (49 tests passing); sensor baseline reviewed against FireSwarm; added agents, refill/basin, gas-sensing and ML-data sections | project kickoff for parallel work |

---

## 14. Sources

- **S1** FireSwarm XPRIZE finalist release — https://www.aap.com.au/aapreleases/globenewswire9648888/
- **S2** ASCE, FireSwarm automated aerial firefighting — https://www.asce.org/publications-and-news/civil-engineering-source/article/2025/07/24/automated-aerial-firefighting-drones-to-provide-faster-responses
- **S3** CTIF, 400 kg water-bombing drones in Canada — https://ctif.org/news/heavy-duty-water-bombing-drones-able-lift-400-kilos-are-being-tested-canadian-forest-fire
- **S4** Strategic × FireSwarm partnership — https://newswire.ca/news-releases/revolutionary-drone-technology-for-battling-wildfires-takes-major-step-forward-with-new-partnership-840102801.html
- **S17** ACC Thunder Wasp GT dossier (MTOW, fuel, endurance, turbine, bucket) — https://www.uncrewed-systems.com/?p=13251 · https://ust-media.com/ust-magazine/UST059/38 · https://ust-media.com/ust-magazine/UST059/32 · https://www.uncrewed-systems.com/content/uploads/magazines/Issue%20059/36/
- **S18** FireSwarm Thunder Wasp GT (hybrid-electric, 4 props, 378.5 L) — https://evtol.news/fireswarm-solutions-thunderwasp-gt
- **S19** Unvalidated payload/endurance claims, ash/heat risk — https://robotics.press/companies/fireswarm-solutions/
- **S5** DJI Agras T50 (reference only) — https://ag.dji.com/fr/t50/specs · reseller specs/hover times https://druav.com/products/dji-agras-t50-agriculture-drone · battery/radar https://volatusdrones.com/products/dji-agras-t50-drone-ready-to-fly-spray-bundle
- **S6** US initial-attack drone fleets — https://www.yahoo.com/news/articles/san-bernardino-co-fire-dept-211235723.html · https://www.techspot.com/news/113180-us-fire-departments-test-autonomous-drones-fight-wildfires.html
- **S7** FLIR Boson+ — https://www.acalbfi.com/nl/file/16132/Boson-Plus-Datasheet.pdf · https://oemcameras.com/?p=4033 · Hadron 640R https://w-technika.cz/cdn/shop/files/hadron20640r20datasheet.pdf
- **S8** Livox Mid-360 — https://www.livoxtech.com/mid-360/specs · https://www.mybotshop.de/Livox-Mid360S_1
- **S9** Jetson Orin NX datasheet — https://openzeka.com/wp-content/uploads/2022/03/Jetson-Orin-NX-Module-Series-Datasheet.pdf
- **S10** Class A foam & fire chemicals — https://www.fs.usda.gov/eng/pubs/pdf/92511209.pdf · https://www.fs.usda.gov/rm/fire/wfcs/documents/Overview%20of%20Use%20and%20Safety%20of%20Wildland%20Fire%20Chemical%20Products.pdf · https://www.perimeter-solutions.com/?p=181 · https://www.fireengineering.com/fire-safety/the-salem-tests/ · https://www.fireengineering.com/?p=264355
- **S11** Waterway buffers / retardant toxicity — https://www.fs.usda.gov/t-d/programs/wfcs/retardants/current/gen/appguide.htm · https://www.nifc.gov/sites/default/files/redbook-files/Chapter12.pdf · https://www.hcn.org/?p=199704
- **S12** Gas/smoke sensing — https://dryad.net/wildfiresensor · https://arxiv.org/pdf/2312.10919 · https://hellofuture.orange.com/en/a-drone-that-can-sniff-out-forest-fires-before-they-generate-visible-smoke · https://tech.eu/2023/08/11/ultra-early-detection-system/
- **S13** FLAME 3 (Hopkins et al. 2024, arXiv:2412.02831; read v2, labeling in §III.A.1 p.5 and §III.C.1 p.9; cite as in `training/FLAME3.md`) — https://arxiv.org/abs/2412.02831 · https://arxiv.org/html/2412.02831v1
- **S14** PX4 — https://docs.px4.io/main/en/dev_setup/dev_env · https://docs.px4.io/main/en/sim_gazebo_gz/multi_vehicle_simulation
- **S15** ROS 2 Lyrical Luth — https://discourse.openrobotics.org/t/ros-2-lyrical-luth-released/55021
- **S16** Water pickup / amphibious drones — https://www.popsci.com/technology/firefighting-drones-help-fight-wildfires/
- **S20** FLAME 3 masks made by thresholding Celsius TIFFs (segmentation study) — https://arxiv.org/html/2609.01390v1 · FLAME 3 on IEEE DataPort — https://ieee-dataport.org/node/14367
- **S21** Thermal decoy/negative datasets: HIT-UAV — https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset · BIRDSAI — https://openaccess.thecvf.com/content_WACV_2020/html/Bondi_BIRDSAI_A_Dataset_for_Detection_and_Tracking_in_Aerial_Thermal_WACV_2020_paper.html · FLAME 2 — https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset
- **S23** FLIR Boson / Boson+ scene dynamic range — https://groupgets.com/pages/radiometricboson · https://www.oem.flir.com/support/support-center/knowledge-base/what-is-the-scene-dynamic-range-sdr-of-flirs-oem-modules/
- **S24** Smouldering 450–700 °C vs flaming 1500–1800 °C peak temperatures (Rein et al. review) — https://www.frontiersin.org/articles/10.3389/fmech.2019.00049/full
- **S22** Scenario data: Alberta historical wildfire 2006–2024 — https://open.alberta.ca/opendata/a221e7a0-4f46-4be7-9c5a-e29de9a3447e · NASA FIRMS — https://firms.modaps.eosdis.nasa.gov/ · CWFIS — https://cwfis.cfs.nrcan.gc.ca/
