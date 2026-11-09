"""DEMO mode: lakes in the static world, and a short end-to-end session (synthetic tiles, so CI needs no dataset)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from sim.config import demo_config
from sim.static import build_static

pytest.importorskip("onnxruntime")


def _small(wet: bool, **over):
    # a smaller DEMO so the test is quick; same code paths as the full one
    kw = {"arena_radius_m": 260.0, "base_xy": (0.0, -235.0), "n_trees": 300, "n_lakes": 3, "lake_r_m": (22.0, 35.0),
              "demo_seekers": 3, "n_extinguishers": 10, "demo_s": 40.0, "fire_clusters": 3}
    kw.update(over)
    return demo_config(wet=wet, **kw)


def test_lakes_have_no_fuel_and_wet_dry_variants_share_the_fires():
    wet, dry = build_static(_small(True), "synth", 7), build_static(_small(False), "synth", 7)
    assert len(wet.lakes) == 3 and all(lk.wet for lk in wet.lakes) and not any(lk.wet for lk in dry.lakes)
    assert [(f.x, f.y, f.area_m2) for f in wet.fires] == [(f.x, f.y, f.area_m2) for f in dry.fires]
    cfg = _small(True)
    R, cell = cfg.arena_radius_m, cfg.fire_cell_m
    for lk in wet.lakes:
        r, c = int((lk.y + R) / cell), int((lk.x + R) / cell)
        assert wet.fuel[r, c] == 0.0
        assert not any(lk.contains(f.x, f.y) for f in wet.fires)
    # night: the lake surface reads a little warmer than the land around it
    lk = wet.lakes[0]
    rr, cc = int((lk.y + R) / wet.res), int((lk.x + R) / wet.res)
    assert wet.bg[rr, cc] > float(np.median(wet.bg))


@pytest.mark.parametrize("wet", [True, False])
def test_short_demo_session_drops_and_refills(wet):
    from sim.game import Session
    s = Session(map_id="demo-lakes-1" if wet else "demo-dry-1", seed=2, mode="demo", cfg=_small(wet))
    init = s.init_message()
    json.dumps(init, allow_nan=False)
    assert init["mode"] == "demo" and len(init["lakes"]) == 3 and len(init["zones"]) == 3
    n = 0
    while s.phase != "SCORE":
        s.step()
        if n % 7 == 0:
            json.dumps(s.snapshot(), allow_nan=False)
        n += 1
    snap = s.snapshot()
    json.dumps(snap, allow_nan=False)
    d = s.result["demo"]
    assert d["drops"] > 0 and d["confirmed"] > 0
    if wet:
        assert d["fills_lake"] > 0
    else:
        assert d["fills_lake"] == 0 and d["lakes"] == 0
    assert -50.0 <= d["fire_fought_pct"] <= 100.0
