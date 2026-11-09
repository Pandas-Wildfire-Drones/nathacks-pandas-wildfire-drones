# Night Initial Attack — an autonomous wildfire drone swarm

**natHacks 2026 · Team Pandas**

> How might we design a coordinated system of autonomous drones that can deliver water to active wildfires safely
> and efficiently at night?

Fire behaviour drops overnight, which makes night the window to catch small spot fires before they grow. We built a
swarm that **finds, confirms, prioritises and knocks down** those hotspots in the dark, and a simulator that runs the
real code end to end so you can watch it work (or try to beat it yourself).

📽️ **Slides:** [link to the shared deck](https://docs.google.com/presentation/d/18SB2Th1RfMuDRv1g1dRPGCWnFcxzPuKzA0KwHjiuiCQ/edit?usp=sharing)

---

## The idea in one minute

| Role | What it does |
|---|---|
| **Seekers** (light electric, radiometric 640×512 thermal) | Fly equal-time search zones, detect hotspots with a classical + CNN pipeline, then descend to **confirm** each one up close before any water is committed. |
| **Extinguishers** | Heavy hybrid drones (play/watch) or, in DEMO mode, small **field-refill attackers** that mix foam concentrate into water picked up at **any lake or the base**, so they refill after every ~15 L drop instead of flying home. |
| **Leader** (a role, not a special drone) | Keeps a **max-heap of fires by threat** (size × growth × heat × confidence) and hands each one to the drone with the shortest *drone → best refill point → fire* trip. A failed post-drop check puts the fire back on the heap. |
| **Ground base** | Reviews uncertain detections (approve / reject). The swarm uses its answer but never waits for it. |

Headline result (DEMO mode, 48 drones, 2 minutes, same fires on both maps): **22 % of fire fought with lakes as
refill points vs 12 % when the only refill point is the base.** Numbers and caveats: [`sim/README.md`](sim/README.md).

## Try it

Everything binds to `127.0.0.1` only.

```bash
# Python 3.11+ (a .venv is recommended)
pip install -e ".[dev,perception,sim]"
python -m sim.maps.build          # once, only if you have the FLAME 3 NADIR data; otherwise a synthetic map is used

# terminal 1: simulation server       terminal 2: browser client
python -m sim.server                  pnpm --dir frontend install && pnpm --dir frontend dev
```

Open **http://127.0.0.1:5173** and pick a mode:

* **Play**: fly the seeker yourself, plan, then fly an extinguisher next to 18 bots. Your score is compared with the
  autonomous swarm on the same map and seed.
* **Watch swarm**: the autonomous system alone.
* **DEMO**: a whole cluster (8 seekers + 40 attackers) on a large map, lakes vs dry basins. The camera hops between
  drones every 10 s and says what each one is doing.

Shortcuts: `http://127.0.0.1:5173/?auto=demo&map=demo-lakes-1` · `?auto=watch&map=p1` · add `&fast=1` for short runs.
Controls are listed in [`sim/README.md`](sim/README.md#phases-and-controls).

Without a browser:

```bash
python -m sim.headless --demo --map demo-lakes-1      # DEMO, prints % fire fought, drops, refills, confirmations
python -m sim.headless --map p1 --seekers 2           # watch mode
python -m pytest -q                                   # full test suite (CI runs it on every PR into main)
```

## What's in the repo

| Path | What | Read first |
|---|---|---|
| `contracts/` | Shared Pydantic messages, entities, topics, interfaces, platform/scenario config: the one thing every part depends on | [`context.md` §11](context.md) |
| `core/` | Swarm brain: fire priority, priority-queue allocation, refill choice, seeker zones, confirmation flow, leader role | module docstrings |
| `perception/` | Thermal detection: classical proposals + CNN verifier (ONNX), georeferencing, tracking, log-odds map, close-range confirm/verify, triage | [`perception/README.md`](perception/README.md), [`perception/RESULTS.md`](perception/RESULTS.md) |
| `training/` | Datasets, composites, group-split cross-validation, CNN training and ONNX export, approach comparison notebooks | [`training/FLAME3.md`](training/FLAME3.md), `training/train_verifier.ipynb` |
| `autonomy/` | Per-drone autonomy: coverage search, CBS path planning, path claims, drop safety gate, refill execution | [`AUTONOMY_DESIGN.md`](AUTONOMY_DESIGN.md) |
| `groundbase/` | Ground-base review queue (operator decisions go back to the swarm as `GroundCommand`) | [`groundbase/README.md`](groundbase/README.md) |
| `sim/` | Simulator: FLAME 3 thermal world, fire model, lakes, drones, server, headless runner | [`sim/README.md`](sim/README.md) |
| `frontend/` | three.js + TypeScript client (3D view, HUD, minimap, plan screen) | — |
| `docs/decisions/` | Design decisions (search altitude, sensor wiring, hot-blob bypass, close-range confirmation) | [`docs/decisions/README.md`](docs/decisions/README.md) |

The project brief, architecture, hard rules and the reasoning behind every number live in
[**`context.md`**](context.md). Start there if you are joining the project.

## Supporting material

* **Slides:** [PLACEHOLDER — shared deck link](https://example.com/replace-with-slides-link)
* **Perception results** (cross-validation, held-out tests, approach comparison): [`perception/RESULTS.md`](perception/RESULTS.md)
* **Simulator, swarm brain, DEMO results:** [`sim/README.md`](sim/README.md)
* **Autonomy design record:** [`AUTONOMY_DESIGN.md`](AUTONOMY_DESIGN.md)
* **Design decisions:** [`docs/decisions/`](docs/decisions/)
* **Open issues / next steps:** GitHub issues, e.g. #21 (CBS vs PX4) and #22 (seeker/attacker split)

## Data and credits

Thermal imagery comes from **FLAME 3** (radiometric thermal UAV imagery of prescribed burns). Please cite it in
anything that uses it:

> Hopkins, B., ONeill, L., Marinaccio, M., Rowell, E., Parsons, R., Flanary, S., Nazim I, Seielstad C, Afghah, F.
> (2024). FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management.
> arXiv preprint arXiv:2412.02831.

Scenario context draws on NASA FIRMS, CWFIS and Alberta Wildfire historical data. Raw datasets are not in the repo
(`exclude/` is gitignored).

**Honest limits.** Fire physics, drop footprint and several platform numbers are placeholders (tagged
`[ASSUMPTION]` / `[GAME-SCALE]` in the code). The CNN has seen almost no hot man-made objects, so a warm cabin can
fool it. Real radiometric data in the hard 80–200 °C band is scarce. Details in `perception/RESULTS.md` and
`context.md`.
