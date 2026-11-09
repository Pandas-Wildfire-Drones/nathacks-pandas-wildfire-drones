# sim/ — demo simulator & game

Human pilot + 9 bots vs. our autonomous swarm, on the same map and seed, in ≤ 4 minutes.
Full design notes, decisions and open questions: `exclude/simulator_task.md` (local, gitignored).

## Run it (localhost only)

```bash
pip install -e ".[dev,perception,sim]"          # or: uv pip install fastapi "uvicorn[standard]" into .venv
python -m sim.maps.build                         # once: FLAME 3 NADIR -> exclude/sim_maps/{p1,p2}.npz (needs the dataset)
python -m sim.server                             # http://127.0.0.1:8765 (WebSocket /ws, /api/maps)
pnpm --dir frontend install && pnpm --dir frontend dev   # http://127.0.0.1:5173
```

Without the FLAME 3 data the `synth` map still works. Useful URLs:
`http://127.0.0.1:5173/?auto=watch&map=p1&seed=1` (start straight into a mode) · add `&fast=1` for short phases.
Headless (no browser) swarm run for tuning: `python -m sim.headless --map p1 --seed 1 --seekers 1`.

## Phases

Briefing 8 s → **Seek 75 s** (player flies the only seeker; the map fills in from the perception log-odds map) →
**Plan 30 s** (revealed map, black elsewhere; assign 9 bots) → **Extinguish 90 s** (player flies the 10th
extinguisher; bots use the real autonomy adapter) → **Results** (% fire left vs. a no-drone baseline, you vs. swarm).
The swarm runs on an identical copy of the world at the same time (shadow run). Pause: Esc / ❚❚ button.

## What is real, what is stubbed — and where to plug in

| Piece | File | Status |
|---|---|---|
| Thermal world | `maps/build.py`, `static.py`, `thermal.py` | real FLAME 3 NADIR backgrounds/char/flame pixels; frames rendered per seeker pose (Planck radiance mixing) |
| Perception | `agents.py` → `perception.pipeline.SeekerPerception` + `FusedDetector(fire_cnn_mix.onnx)` | **real** (workstream C). Swap the model in `world.MODEL_PATH` |
| Lawnmower search | `agents.SeekerAgent.assign_zone` → `autonomy.coverage_search` | **real** (B) |
| Extinguisher execution | `agents.ExtinguisherAgent` → `autonomy.swarm_adapter.SwarmAutonomyAdapter` | **real** (B): planning, CBS, path claims, drop gate, refill |
| VERIFY after a drop | `agents.ExtinguisherAgent._verify_step` → `perception.confirm.SuppressionChecker` | **real** (C), extinguisher's 320×256 camera |
| Zone allocation | `brain_stub.DummyZoneAllocator` | **STUB** (A) – implement `ZoneAllocator` |
| Task assignment | `brain_stub.DummyTaskAssigner` | **STUB** (A) – implement `TaskAssigner` |
| Refill choice | `brain_stub.DummyRefillAssigner` | **STUB** (A) – implement `RefillAssigner` |
| Leader role / confirmation | `brain_stub.StubLeader` | **STUB** (A) – confirms at the CNN confirm threshold |
| Fire physics, water, kinematics | `fire.py`, `vehicles.py`, `config.py` (`[GAME-SCALE]` tags) | simple, ours — MechE (E) to replace numbers |
| Human-mode bots | `bots.py` | game logic (fly to anchor, nearest known hotspot) |

Swarm-brain drop-in: give `World` your assigner, e.g. in `world.py`
`StubLeader("leader", ..., assigner=YourTaskAssigner(), refill=YourRefillAssigner())` and your
`ZoneAllocator` in `World.begin_seek`. Inputs/outputs are documented on the Protocols in `brain_stub.py`.

## Conventions

ENU metres, origin = arena centre; rasters `[row, col]`, row 0 = south; everything on the bus goes through
`encode()/decode()` (`bus.SimBus`). The sim is deterministic per seed for the swarm/baseline worlds (perception
results are consumed after a fixed latency; the fire model's random field is seeded per step).
