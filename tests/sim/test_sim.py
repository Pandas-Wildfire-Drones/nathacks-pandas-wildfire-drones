"""Simulator tests. Use the synthetic map so they run in CI without the FLAME 3 dataset."""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from contracts import Heartbeat, Pose, Position, SwarmMessage, ThermalFrame, default_seeker_spec
from sim.bus import SimBus, SimClock
from sim.config import GameConfig
from sim.fire import FireModel
from sim.static import build_static

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def static():
    return build_static(GameConfig(), "synth", seed=3)


def _single_fire(fm: FireModel, ros_m_min: float) -> int:
    fm.heat[:] = 0
    fm.ignited[:] = False
    fm.fire_id[:] = -1
    c = fm.n // 2
    sl = (slice(c - 1, c + 2), slice(c - 1, c + 2))
    fm.heat[sl] = 1.0
    fm.ignited[sl] = True
    fm.fire_id[sl] = 0
    fm.ros[:] = ros_m_min / 60.0
    fm.resid[:] = 1e9
    fm.fuel[:] = 1.0
    fm.windk[:] = 0.0
    return c


def test_spread_rate_matches_ros(static):
    cfg = replace(GameConfig(), wind_speed_mps=0.0)
    fm = FireModel(static, cfg, seed=1)
    _single_fire(fm, ros_m_min=1.0)
    for _ in range(10 * 12):   # 10 fire-minutes
        fm.step(5.0)
    radius = math.sqrt(fm.ignited.sum() * fm.cell ** 2 / math.pi)
    assert 7.5 < radius < 13.5      # ~10 m after 10 min at 1 m/min (+ initial 1.5 m patch)


def test_water_knocks_fire_down(static):
    cfg = GameConfig()
    fm = FireModel(static, cfg, seed=1)
    f = static.fires[0]
    before = fm.per_fire_metric()[0]
    fm.apply_water(f.x, f.y, 170.0, agl_m=2.0)
    assert fm.per_fire_metric()[0] < 0.5 * before
    assert fm.wet.max() > 0


def test_identical_worlds_evolve_identically(static):
    cfg = GameConfig()
    a, b = FireModel(static, cfg, seed=5), FireModel(static, cfg, seed=5)
    for _ in range(20):
        a.step(1.5)
        b.step(1.5)
    assert np.array_equal(a.heat, b.heat)


def test_bus_decodes_per_subscriber():
    clock = SimClock()
    bus = SimBus(clock, latency_s=0.05)
    got = []
    bus.subscribe("drone/*/heartbeat", got.append)
    bus.subscribe("drone/#", got.append)
    from contracts import DroneKind, DroneState
    st = DroneState(drone_id="s1", kind=DroneKind.SEEKER, position=Position(x=0, y=0, z=10), battery_pct=90,
                    timestamp=0.0)
    msg = SwarmMessage.create("s1", Heartbeat(state=st), 0.0)
    bus.publish("drone/s1/heartbeat", msg)
    bus.pump(0.01)
    assert got == []                       # latency not elapsed
    bus.pump(0.1)
    assert len(got) == 2 and got[0] is not got[1] and got[0] == got[1]


def test_rendered_frame_feeds_real_perception(static):
    """A seeker frame rendered over a sim fire is detected by the real FUSED detector and georeferenced onto it."""
    from perception.detectors import FusedDetector, load_classifier
    from perception.frame_context import FrameContext
    from perception.pipeline import SeekerPerception
    from sim import thermal
    from sim.world import MODEL_PATH

    cfg = GameConfig()
    fm = FireModel(static, cfg, seed=3)
    cam = default_seeker_spec().thermal.camera
    big = max(static.fires, key=lambda f: f.area_m2)
    pose = Pose(position=Position(x=big.x + 4, y=big.y - 3, z=90.0), yaw_deg=20.0)
    job = thermal.prepare(static, fm, pose, cam, noise_seed=1)
    temp = thermal.render(static, job, cfg.fire_cell_m)
    assert temp.shape == (cam.height_px, cam.width_px) and temp.max() > 100
    perc = SeekerPerception("s1", FusedDetector(load_classifier(MODEL_PATH)))
    hs = perc.process(ThermalFrame(drone_id="s1", timestamp=1.0, temp_c=temp, pose=pose, camera=cam),
                      FrameContext(agl_m=90.0, saturation_c=550.0))
    assert any(math.hypot(h.position.x - big.x, h.position.y - big.y) < 6.0 for h in hs)


def test_short_session_reaches_score_with_valid_json():
    from sim.game import Session
    cfg = replace(GameConfig(), briefing_s=0.5, seek_s=6.0, plan_s=0.5, extinguish_s=8.0)
    s = Session(map_id="synth", seed=4, mode="play", swarm_seekers=1, cfg=cfg)
    s.set_plan({e.id: {"x": 0.0, "y": 0.0} for e in s.human.exts})
    json.dumps(s.init_message(), allow_nan=False)
    n = 0
    while s.phase != "SCORE":
        s.step()
        if n % 5 == 0:
            json.dumps(s.snapshot(), allow_nan=False)
        n += 1
        assert n < 20 * 30
    snap = s.snapshot()
    json.dumps(snap, allow_nan=False)
    assert snap["phase"] == "SCORE" and "score" in snap
    for side in ("human", "swarm"):
        assert 0.0 <= s.result[side]["fire_left_pct"] <= 150.0
