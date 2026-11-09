"""Run the autonomous swarm (and the no-action baseline) without a browser, as fast as the CPU allows.

    python -m sim.headless --map p1 --seed 1 --seekers 1
    python -m sim.headless --demo --map demo-lakes-1 --seekers 6 --attackers 30
Prints phase timings, coverage, detections, autonomy log lines and the final % fire left. Use it to tune
[GAME-SCALE] numbers or to check a swarm-brain change before trying it in the game.
"""
from __future__ import annotations

import argparse
import json
import time

from sim.game import Session


def run(map_id: str, seed: int, seekers: int, verbose: bool = True, extinguish_s: float | None = None) -> dict:
    from dataclasses import replace

    from sim.config import GameConfig
    cfg = GameConfig()
    if extinguish_s is not None:
        cfg = replace(cfg, extinguish_s=extinguish_s)
    t0 = time.time()
    s = Session(map_id=map_id, seed=seed, mode="watch", swarm_seekers=seekers, cfg=cfg)
    if verbose:
        print(f"session built in {time.time() - t0:.1f}s; map {s.static.map.title}; {len(s.static.fires)} fires")
    last_phase, tp = s.phase, time.time()
    n = 0
    while s.phase != "SCORE":
        s.step()
        n += 1
        if verbose and n % int(s.cfg.tick_hz * 15) == 0:
            w = s.swarm
            ld = w.leader
            print(f"  t={s.t:6.1f} {s.phase:10s} cov={100 * w.coverage_frac():5.1f}% hotspots={len(ld.hotspots):3d} "
                  f"assign={len(ld.assignments)} tasks={ld.n_tasks} fire={w.fire.fire_metric():6.1f} "
                  f"base={s.baseline.fire.fire_metric():6.1f} drop={w.stats['dropped_l']:.0f}L")
        if s.phase != last_phase:
            if verbose:
                print(f"{last_phase} done in {time.time() - tp:.1f}s wall")
            last_phase, tp = s.phase, time.time()
    if verbose:
        print("autonomy log (last lines):")
        for line in list(s.swarm.log)[-12:]:
            print("   ", line)
        print(json.dumps(s.result, indent=2))
    return s.result


def run_demo(map_id: str = "demo-lakes-1", seed: int = 1, verbose: bool = True, **over) -> dict:
    """DEMO mode, headless. `over` overrides demo_config fields, e.g. demo_seekers=8, n_extinguishers=32."""
    from sim.config import DEMO_MAPS, demo_config
    cfg = demo_config(wet=DEMO_MAPS[map_id]["wet"], **over)
    t0 = time.time()
    s = Session(map_id=map_id, seed=seed, mode="demo", cfg=cfg)
    build_s = time.time() - t0
    if verbose:
        print(f"built in {build_s:.1f}s: {s.title}; {len(s.static.fires)} fires, {len(s.static.lakes)} lakes, "
              f"{len(s.swarm.seekers)} seekers, {len(s.swarm.attackers)} attackers")
    t1, n, worst = time.time(), 0, 0.0
    while s.phase != "SCORE":
        ts = time.perf_counter()
        s.step()
        if n % 3 == 0:
            s.snapshot()
        worst = max(worst, time.perf_counter() - ts)
        n += 1
        if verbose and n % int(s.cfg.tick_hz * 15) == 0:
            d = s.demo_stats()
            print(f"  t={s.t:5.1f} cov={100 * s.swarm.coverage_frac():5.1f}% fought={d['fire_fought_pct']:5.1f}% "
                  f"drops={d['drops']} lake={d['fills_lake']} base={d['fills_base']} supp={d['suppressed']} "
                  f"hot={d['verify_hot']} queue={d['queue']} conf={d['confirmed']}")
    wall = time.time() - t1
    res = s.result["demo"]
    res.update({"wall_s": round(wall, 1), "build_s": round(build_s, 1), "realtime_x": round(s.t / wall, 2),
                "worst_step_ms": round(1000 * worst, 1)})
    if verbose:
        print(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="p1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seekers", type=int, default=1)
    ap.add_argument("--demo", action="store_true", help="DEMO mode (use a demo-* map id)")
    ap.add_argument("--attackers", type=int, default=None)
    a = ap.parse_args()
    if a.demo:
        over = {}
        if a.seekers != 1:
            over["demo_seekers"] = a.seekers
        if a.attackers is not None:
            over["n_extinguishers"] = a.attackers
        run_demo(a.map if a.map.startswith("demo-") else "demo-lakes-1", a.seed, **over)
    else:
        run(a.map, a.seed, a.seekers)


if __name__ == "__main__":
    main()
