# sim/ — demo simulator & game

Three modes on the same perception / autonomy / swarm-brain code:

* **Play** — you + 18 bots (20 drones with your seeker) vs. our autonomous swarm on the same map and seed, ≤ 4 min.
* **Watch swarm** — the autonomous swarm alone: N seekers + 19 heavy (170 L) extinguishers.
* **DEMO** — a whole cluster on a large map: 8 seekers and 40 small field-refill attackers flying **at the same
  time** for 2 minutes, with lakes as refill points (or the same map with dry basins, for comparison).

Full design notes, decisions and open questions: `exclude/simulator_task.md` (local, gitignored).

## Run it (localhost only)

```bash
pip install -e ".[dev,perception,sim]"          # or: uv pip install fastapi "uvicorn[standard]" into .venv
python -m sim.maps.build                         # once: FLAME 3 NADIR -> exclude/sim_maps/{p1,p2}.npz (needs the dataset)
python -m sim.server                             # http://127.0.0.1:8765 (WebSocket /ws, /api/maps)
pnpm --dir frontend install && pnpm --dir frontend dev   # http://127.0.0.1:5173
```

Without the FLAME 3 data the `synth` tiles are used (DEMO maps fall back to them too). Useful URLs:
`http://127.0.0.1:5173/?auto=watch&map=p1&seed=1` · `?auto=demo&map=demo-lakes-1` · add `&fast=1` for short phases.
Second checkout side by side: `SIM_PORT=8766 python -m sim.server` and `SIM_PORT=8766 pnpm --dir frontend exec vite --port 5174`.

Headless (no browser), as fast as the CPU allows:

```bash
python -m sim.headless --map p1 --seed 1 --seekers 1                                  # watch mode
python -m sim.headless --demo --map demo-lakes-1 --seekers 8 --attackers 40           # DEMO (also demo-dry-1, -2)
```

## Phases and controls

Play/watch: Briefing 8 s → **Seek 90 s** → **Plan 30 s** → **Extinguish 90 s** → **Results** (% fire left vs. a
no-drone baseline). DEMO: one **2-minute** phase where seekers search and attackers fight together → **Results**
(% fire fought = 100 − % fire left).

| Key | |
|---|---|
| `W A S D` · `R F` · `Shift` · `Space` | fly · altitude · boost · drop (play) |
| `Q E` or **mouse** | turn. Click the view to capture the mouse; if the browser refuses (e.g. right after Esc), drag with the left button |
| `Esc` / `P` / ❚❚ | pause menu (resume, restart, new seed, main menu) |
| drag / wheel | orbit / zoom (watch, DEMO) |
| `Tab` | next drone (watch, DEMO) · DEMO: `A` auto camera on/off, `F` free camera |
| `T` · `M` | thermal picture-in-picture · sound |

DEMO's camera hops to another drone every 10 s (a seeker every third hop, otherwise a busy attacker), highlights it on
the map and the minimap, and the card at the bottom says what it is doing ("Flying to lake L3 → then seeker-5-H2 ·
priority 0.81", "Verifying …", "Lawnmower over zone Z4 …"). Orbiting or zooming takes the view back from the auto camera.

## The swarm brain (`core/`, workstream A)

Pure logic (contracts + stdlib only, hard rule 1), on the bus through `core.leader.SwarmLeader`:

| Piece | File | What it does |
|---|---|---|
| Fire priority | `core/priority.py` | `raw = confidence × projected_area × heat_weight`. Projected area = area + growth × 5 min (measured growth from re-observation if we have it, else a rate-of-spread prior from temperature). Heat weight 0.5 (smoulder) … 1.0 (flaming). Rationale in the module docstring. |
| Priority queue | `core/allocation.FireQueue` | max-heap with lazy re-keying. Fires are pushed on detection / every new report / a failed VERIFY. |
| Task assignment | `core/allocation.allocate` | pop fires highest-priority first; each slot goes to the idle extinguisher with the lowest **refill-aware ETA**: `drone → best refill point → fire` (the refill point is chosen per task, minimising the detour, not just nearest to the drone), or straight there if it is loaded. Big fires get several drones (litres needed / payload, capped). With the base as the only refill point this degrades to plain highest-priority-first. |
| VERIFY loop | `core/leader._on_verify` | after a drop the dropping drone re-images the spot (`perception.confirm.SuppressionChecker`). Still hot → new area / position, need and priority recomputed, back on the heap. Suppressed → closed (a later seeker detection re-opens it). |
| Refill choice | `core/leader._on_refill` | `RefillRequest` → `RefillGrant` to the point minimising drone → point → its task's fire (lakes when enabled, else base). |
| Pre-positioning | `core/leader._stage / _standby` | idle + empty drones top up at the nearest point; idle + loaded drones with nothing queued spread evenly over the refill points (lakes + base), so later fires have water and a loaded drone close by. No-op without lakes. |
| Seeker zones | `core/zones.py` | columns × bands of the circle that every seeker finishes at about the same time (transit + scan lines + turns), solved exactly as a contiguous partition of the actual scan lines. |

**Confirmation** (`core/leader._assign_investigations`, `sim/agents.SeekerAgent._investigate`, `sim/ground.py`):
every detection starts DETECTED. The leader sends the nearest free seeker an INVESTIGATE task; it leaves its
lawnmower, descends to 80 m (7.2 cm/px, the CNN's reference GSD), feeds 5–12 frames at 4 Hz to the real
`perception.confirm.HotspotConfirmer` (search radius sized to the fire, 6–10 m), then resumes its zone.

| Close-range result | What happens |
|---|---|
| mean p ≥ confirm threshold (precision-first) | CONFIRMED: water may be committed |
| mean p < reject threshold, target seen | FALSE_POSITIVE: closed; drones already on the way are recalled |
| in between, or nothing hot found near the point | REVIEW: `ReviewReport` → ground base queue (`groundbase/review_queue.py`) |
| ground answers | `GroundCommand(CONFIRM_FIRE)` / `MARK_FALSE_POSITIVE` (sim operator: 70 % / 30 %, after 6 s) |
| no ground answer in 20 s (or `ground_link=False`) | leader confirms on its own: recall-first, the swarm never depends on the link |

DEMO dispatches attackers on detection (`dispatch_on_detect`) but gates the drop: an attacker reaching an unconfirmed
fire HOLDs at its transit layer until the leader re-sends the picture as confirmed. Human-mode bots keep
`brain_stub.StubLeader` + `bots.AnchorAssigner` (the player's own eyes confirm).

## DEMO mode

* **Maps** (`config.DEMO_MAPS`): `demo-lakes-1/2` and `demo-dry-1/2`, radius 450 m, FLAME 3 tiles at 0.3 m/px, 40
  spot fires of 4 kinds (size 0.5–7 m², rate of spread 0–0.6 m/min, smoulder → flaming temperatures), 12 decoys,
  6 lakes. The dry variant uses the **same layout seed**: identical fires, the basins are just dry, so the two runs
  compare only the water.
* **Attackers** (`sim/attacker.py`, `config.attacker_spec`): ~15 L of Class A foam per drop (team assumption: 10–20 kg
  suppressant payload; foam counted as 3× water, contracts `AgentSpec`), mixed on board from water picked up at any
  lake or the base, so they refill after **every** drop. Direct flight at per-drone altitude layers (vertical
  separation standing in for PX4); no CBS (GitHub issue #21). Fill 5 s [GAME-SCALE].
* **Seekers** start airborne, fly equal-time zones and re-fly them (reversed) when done, so fires get re-observed.

Reference runs (`python -m sim.headless --demo`, seed 1, 120 s, 48 drones; % fire fought vs. no drones):

| Seekers + attackers | Lakes | Dry basins |
|---|---|---|
| 3 + 45 | 16.3 | 7.8 |
| 5 + 43 | 19.0 | 8.8 |
| **8 + 40 (default)** | **21.6** | **11.4** |
| 12 + 36 | 21.4 | 11.8 |
| 16 + 32 | 18.6 | 11.7 |

(Sweep run before close-range confirmation existed. With confirmation on, 8 + 40: lakes 22.1 %, dry 12.2 %; seeker
coverage drops from 99.6 % to 86.6 % because seekers stop to investigate.) Too few seekers miss fires (73 % coverage with 3); too many starve the attackers. Lakes roughly double the drops
(104 vs 58) and the fire fought. One seed and placeholder physics: tuning is tracked in GitHub issue #22. Performance:
the 48-drone DEMO runs ~3× faster than real time headless (perception in 4 worker threads, ~0.9 GB RSS).

## What is real, what is placeholder

| Piece | File | Status |
|---|---|---|
| Thermal world | `maps/build.py`, `static.py`, `thermal.py` | real FLAME 3 NADIR backgrounds/char/flame pixels; frames rendered per seeker pose (Planck radiance mixing); lakes read warmer than land at night [VERIFY] |
| Perception | `agents.py` → `perception.pipeline.SeekerPerception` + `FusedDetector(fire_cnn_mix.onnx)` | **real** (workstream C). Swap the model in `world.MODEL_PATH` |
| Lawnmower search | `agents.SeekerAgent.assign_zone` → `autonomy.coverage_search` | **real** (B) over `core.zones` |
| Heavy extinguisher execution (play/watch) | `agents.ExtinguisherAgent` → `autonomy.swarm_adapter.SwarmAutonomyAdapter` | **real** (B): planning, CBS, path claims, drop gate, refill |
| Attacker execution (DEMO) | `attacker.AttackerAgent` | sim executor on the same bus protocol (TaskAssign / RefillRequest / RefillGrant / TaskComplete / VERIFY report) |
| VERIFY after a drop | `agents.VerifyPass` → `perception.confirm.SuppressionChecker` | **real** (C), extinguisher's 320×256 camera |
| Swarm brain | `core/` | **real** (A), see above |
| Fire physics, water, kinematics | `fire.py`, `vehicles.py`, `config.py` (`[GAME-SCALE]` tags) | simple, ours — MechE (E) to replace numbers |
| Human-mode bots | `bots.py` | game logic (fly to anchor, nearest known hotspot) |

## Conventions

ENU metres, origin = arena centre; rasters `[row, col]`, row 0 = south; everything on the bus goes through
`encode()/decode()` (`bus.SimBus`). The sim is deterministic per seed for the swarm/baseline worlds (perception
results are consumed after a fixed latency; the fire model's random field is seeded per step).
