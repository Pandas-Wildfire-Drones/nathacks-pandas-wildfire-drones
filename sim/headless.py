"""Run the autonomous swarm (and the no-action baseline) without a browser, as fast as the CPU allows.

    python -m sim.headless --map p1 --seed 1 --seekers 1
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="p1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seekers", type=int, default=1)
    a = ap.parse_args()
    run(a.map, a.seed, a.seekers)


if __name__ == "__main__":
    main()
