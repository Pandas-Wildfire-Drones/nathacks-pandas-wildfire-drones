"""Swarm brain (core/): priority, priority queue, refill-aware allocation, zones, and the bus-facing leader."""
from __future__ import annotations

import math

from autonomy.coverage_search import generate_coverage_waypoints
from contracts import (
    DroneKind,
    DroneState,
    Heartbeat,
    Hotspot,
    HotspotReport,
    HotspotStatus,
    Position,
    Powertrain,
    RefillGrant,
    RefillRequest,
    RefillTargetKind,
    SwarmMessage,
    TaskAssign,
    TaskComplete,
    TaskType,
    sub_my_commands,
    topic_for,
)
from core.allocation import (
    DroneView,
    FireQueue,
    FireView,
    RefillPoint,
    allocate,
    best_refill,
)
from core.leader import LeaderConfig, SwarmLeader
from core.priority import fire_priority, normalized_priority, ros_prior_m_min
from core.zones import EqualTimeZoneAllocator, band_edges, zone_cells
from sim.bus import SimBus, SimClock


def _h(hid="H", area=2.0, temp=400.0, conf=0.95, growth=None, x=0.0, y=0.0) -> Hotspot:
    return Hotspot(hotspot_id=hid, position=Position(x=x, y=y), peak_temp_c=temp, est_area_m2=area,
                   confidence=conf, growth_rate_m2_per_min=growth, first_seen=0.0, last_seen=0.0)


# ---- priority ------------------------------------------------------------------------------------------------------
def test_priority_orders_size_heat_growth_and_confidence():
    base = fire_priority(_h())
    assert fire_priority(_h(area=6.0)) > base                      # bigger
    assert fire_priority(_h(temp=550.0)) > base                    # hotter = more intense
    assert fire_priority(_h(conf=0.5)) < base                      # less sure
    assert fire_priority(_h(growth=3.0)) > fire_priority(_h(growth=0.0))   # measured growth overrides the prior
    assert ros_prior_m_min(550.0) > ros_prior_m_min(200.0) > 0
    # a small but fast-spreading flaming spot can outrank a larger, cooler smoulder
    assert fire_priority(_h(area=2.0, temp=550.0)) > fire_priority(_h(area=3.0, temp=160.0))
    for raw in (0.0, 1.0, 50.0, 1e4):
        assert 0.0 <= normalized_priority(raw) <= 1.0


# ---- queue ---------------------------------------------------------------------------------------------------------
def test_fire_queue_is_a_max_heap_with_rekey_and_discard():
    q = FireQueue()
    for fid, p in (("a", 0.2), ("b", 0.9), ("c", 0.5)):
        q.push(fid, p)
    q.push("a", 0.95)                 # re-key: old entry becomes stale
    q.discard("c")
    assert len(q) == 2 and "c" not in q
    assert q.pop() == ("a", 0.95) and q.pop() == ("b", 0.9) and q.pop() is None
    # a stale entry must never come back after its fire was popped and pushed again
    q.push("x", 0.1)
    q.push("x", 0.2)
    assert q.pop() == ("x", 0.2)
    q.push("x", 0.3)
    assert q.pop() == ("x", 0.3) and q.pop() is None


# ---- allocation ----------------------------------------------------------------------------------------------------
def _pt(pid, x, y, kind=RefillTargetKind.WATER_SOURCE, fill=5.0):
    return RefillPoint(pid, kind, Position(x=x, y=y), fill)


def test_refill_point_is_chosen_per_task_not_nearest_to_drone():
    drone, fire = Position(x=-100, y=0), Position(x=100, y=0)
    near_drone = _pt("L1", -60, 60)       # 72 m from the drone, but 171 m on to the fire: 243 m
    near_fire = _pt("L2", 100, 30)        # 202 m from the drone, 30 m on to the fire: 232 m
    best, _ = best_refill(drone, fire, [near_drone, near_fire])
    assert best.point_id == "L2"
    nearest, _ = best_refill(drone, None, [near_drone, near_fire])   # no task: plain nearest
    assert nearest.point_id == "L1"


def test_highest_priority_fire_gets_the_lowest_refill_aware_eta():
    base = _pt("S1", 0, -400, RefillTargetKind.STATION)
    lake = _pt("L1", 0, 80)
    pts = [base, lake]
    fires = {"hot": FireView("hot", Position(x=0, y=100), 0.9, 15.0), "cool": FireView("cool", Position(x=0, y=-380), 0.2, 15.0)}
    q = FireQueue()
    for f in fires.values():
        q.push(f.fire_id, f.priority)
    drones = [DroneView("loaded-far", Position(x=0, y=-400), 15.0, 15.0, 20.0),     # 500 m away, full
              DroneView("empty-near", Position(x=0, y=60), 0.0, 15.0, 20.0)]        # 20 m from the lake, empty
    out = {a.fire_id: a for a in allocate(q, fires, drones, pts)}
    assert out["hot"].drone_id == "empty-near" and out["hot"].refill.point_id == "L1"
    assert out["cool"].drone_id == "loaded-far" and out["cool"].refill is None
    assert "hot" in q and "cool" in q                                   # fires keep their place in the heap


def test_without_lakes_it_degrades_to_highest_priority_first():
    base = _pt("S1", 0, 0, RefillTargetKind.STATION)
    fires = {f"F{i}": FireView(f"F{i}", Position(x=100 * i, y=300), p, 15.0) for i, p in enumerate((0.3, 0.8, 0.5))}
    q = FireQueue()
    for f in fires.values():
        q.push(f.fire_id, f.priority)
    drones = [DroneView(f"d{i}", Position(x=0, y=0), 0.0, 15.0, 20.0) for i in range(2)]   # all at base, empty
    out = allocate(q, fires, drones, [base])
    assert [a.fire_id for a in out] == ["F1", "F2"]                  # by priority; identical drones
    assert all(a.refill.point_id == "S1" for a in out)


def test_big_fire_gets_several_drones_up_to_the_cap():
    fires = {"F": FireView("F", Position(x=0, y=0), 0.9, need_l=100.0, max_drones=3)}
    q = FireQueue()
    q.push("F", 0.9)
    drones = [DroneView(f"d{i}", Position(x=10 * i, y=0), 15.0, 15.0, 20.0) for i in range(5)]
    out = allocate(q, fires, drones, [])
    assert [a.drone_id for a in out] == ["d0", "d1", "d2"]


# ---- zones ---------------------------------------------------------------------------------------------------------
def _poly_area(pts):
    return 0.5 * abs(sum(a.x * b.y - b.x * a.y for a, b in zip(pts, pts[1:] + pts[:1])))


def test_zones_tile_the_arena_and_balance_better_than_equal_bands():
    R, sp, base = 444.0, 81.7, Position(x=0, y=-420)
    for n in (1, 3, 6, 8):
        zones = EqualTimeZoneAllocator(sp, base, margin_m=0.0).allocate(R, [f"s{i}" for i in range(n)])
        assert len(zones) == n
        total = sum(_poly_area(z.boundary) for z in zones.values())
        assert abs(total - math.pi * R * R) / (math.pi * R * R) < 0.01
        for z in zones.values():
            assert generate_coverage_waypoints(z, sp, z_m=150.0)
        _, worst_cells = zone_cells(n, R, sp, base)
        _, worst_bands = band_edges(n, R, sp, base)
        assert worst_cells <= worst_bands + 1e-6


# ---- leader on the bus ---------------------------------------------------------------------------------------------
class _Bed:
    """A leader with fake extinguishers that just record their commands."""

    def __init__(self, drones: dict[str, tuple[float, float, float]], points, **cfg):
        self.clock = SimClock()
        self.bus = SimBus(self.clock, latency_s=0.0)
        self.leader = SwarmLeader("leader", self.bus, self.clock, LeaderConfig(confirm_threshold=0.8, capacity_l=15.0,
                                  speed_mps=20.0, agent_mult=3.0, **cfg), points)
        self.leader.active = True
        self.inbox: dict[str, list] = {d: [] for d in drones}
        self.drones = drones
        for d in drones:
            self.bus.subscribe(sub_my_commands(d), self.inbox[d].append)
        self.seq = 0

    def send(self, sender, body):
        self.seq += 1
        msg = SwarmMessage.create(sender, body, self.clock.now(), seq=self.seq)
        self.bus.publish(topic_for(msg), msg)

    def heartbeats(self, task: dict | None = None):
        task = task or {}
        for d, (x, y, w) in self.drones.items():
            st = DroneState(drone_id=d, kind=DroneKind.EXTINGUISHER, position=Position(x=x, y=y, z=30),
                            powertrain=Powertrain.HYBRID, battery_pct=100, fuel_remaining_kg=3, fuel_pct=100,
                            water_remaining_l=w, current_task_id=task.get(d), timestamp=self.clock.now())
            self.send(d, Heartbeat(state=st))

    def run(self, s=1.0):
        for _ in range(int(s / 0.05)):
            self.clock.t += 0.05
            self.bus.pump(self.clock.t)
            self.leader.tick(self.clock.t)

    def tasks(self, d):
        return [m.body.task for m in self.inbox[d] if isinstance(m.body, TaskAssign)]


def test_leader_assigns_by_priority_and_requeues_after_failed_verify():
    lake = _pt("L1", 0, 90)
    base = _pt("S1", 0, -400, RefillTargetKind.STATION)
    bed = _Bed({"near": (0, 70, 0.0), "far": (0, -400, 15.0)}, [base, lake], max_per_fire=1)
    bed.heartbeats()
    bed.send("seeker-1", HotspotReport(hotspot=_h("big", area=6.0, temp=550.0, x=0, y=100)))
    bed.send("seeker-1", HotspotReport(hotspot=_h("small", area=0.5, temp=180.0, x=0, y=-300)))
    bed.run(1.0)
    big = bed.leader.hotspots["big"]
    assert big.priority > bed.leader.hotspots["small"].priority
    t_near = bed.tasks("near")
    assert t_near and t_near[0].type is TaskType.EXTINGUISH and t_near[0].target_hotspot_id == "big"
    assert bed.tasks("far")[0].target_hotspot_id == "small"
    # the empty drone asks where to refill: the lake next to its fire, not the base
    bed.send("near", RefillRequest(drone_id="near", position=Position(x=0, y=70), water_remaining_l=0.0))
    bed.run(0.2)
    grants = [m.body for m in bed.inbox["near"] if isinstance(m.body, RefillGrant)]
    assert grants and grants[-1].target_id == "L1" and grants[-1].target_kind is RefillTargetKind.WATER_SOURCE
    # drop done, VERIFY says still hot: the fire goes back on the heap with the same formula
    bed.send("near", TaskComplete(task_id=t_near[0].task_id, drone_id="near", success=True, water_used_l=15.0))
    bed.heartbeats({"near": "VERIFY-big"})
    bed.run(0.2)
    assert "big" not in bed.leader.queue
    hot = big.hotspot.model_copy(update={"status": HotspotStatus.CONFIRMED, "est_area_m2": 4.0})
    bed.send("near", HotspotReport(hotspot=hot))
    bed.heartbeats()
    bed.run(1.0)
    assert bed.leader.stats["verify_hot"] == 1
    assert len(bed.tasks("near")) == 2 and bed.tasks("near")[1].target_hotspot_id == "big"
    # second VERIFY: suppressed -> closed
    t2 = bed.tasks("near")[1]
    bed.send("near", TaskComplete(task_id=t2.task_id, drone_id="near", success=True, water_used_l=15.0))
    bed.run(0.2)
    bed.send("near", HotspotReport(hotspot=hot.model_copy(update={"status": HotspotStatus.SUPPRESSED})))
    bed.run(1.0)
    assert big.hotspot.status is HotspotStatus.SUPPRESSED and "big" not in bed.leader.queue


def test_idle_loaded_drones_stage_over_refill_points_and_empty_ones_top_up():
    pts = [_pt("S1", 0, -400, RefillTargetKind.STATION), _pt("L1", -200, 0), _pt("L2", 200, 0)]
    bed = _Bed({f"d{i}": (0, -400, 15.0) for i in range(6)} | {"e": (0, 0, 0.0)}, pts, max_per_fire=3,
               prefill_idle=True, stage_idle=True)
    bed.heartbeats()
    bed.run(1.0)
    targets = [bed.tasks(d)[0] for d in bed.drones]
    assert all(t.type is TaskType.REFILL for t in targets)
    where = [t.target_station_id or t.target_water_source_id for t in targets]
    assert where.count("S1") >= 1 and where.count("L1") >= 2 and where.count("L2") >= 2
