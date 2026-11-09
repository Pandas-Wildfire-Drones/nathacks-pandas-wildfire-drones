"""Close-range confirmation: INVESTIGATE tasks, ground review (decision delivery), autonomous timeout, drop gate."""
from __future__ import annotations

from contracts import (
    SUB_GROUND_COMMANDS,
    CommandKind,
    DroneKind,
    DroneState,
    GroundCommand,
    Heartbeat,
    Hotspot,
    HotspotReport,
    HotspotStatus,
    Position,
    Powertrain,
    RefillTargetKind,
    ReviewReport,
    SwarmMessage,
    TaskAssign,
    TaskComplete,
    TaskType,
    sub_my_commands,
    topic_for,
)
from core.allocation import RefillPoint
from core.leader import LeaderConfig, SwarmLeader
from groundbase.review_queue import GroundReviewQueue, ReviewDisposition
from sim.bus import SimBus, SimClock


class Bed:
    def __init__(self, **cfg):
        self.clock = SimClock()
        self.bus = SimBus(self.clock, latency_s=0.0)
        base = RefillPoint("S1", RefillTargetKind.STATION, Position(x=0, y=-300), 5.0)
        self.leader = SwarmLeader("leader", self.bus, self.clock,
                                  LeaderConfig(confirm_threshold=0.8, capacity_l=15.0, speed_mps=20.0, **cfg), [base])
        self.leader.active = True
        self.inbox = {d: [] for d in ("s-near", "s-far", "ext")}
        for d in self.inbox:
            self.bus.subscribe(sub_my_commands(d), self.inbox[d].append)
        self.ground_cmds = []
        self.bus.subscribe(SUB_GROUND_COMMANDS, self.ground_cmds.append)
        self.seq = 0

    def send(self, sender, body):
        self.seq += 1
        msg = SwarmMessage.create(sender, body, self.clock.now(), seq=self.seq)
        self.bus.publish(topic_for(msg), msg)

    def heartbeats(self, seeker_task=None):
        for d, (x, y) in (("s-near", (0, 90)), ("s-far", (0, -250))):
            self.send(d, Heartbeat(state=DroneState(
                drone_id=d, kind=DroneKind.SEEKER, position=Position(x=x, y=y, z=150), battery_pct=90,
                current_task_id=(seeker_task or {}).get(d), timestamp=self.clock.now())))
        self.send("ext", Heartbeat(state=DroneState(
            drone_id="ext", kind=DroneKind.EXTINGUISHER, position=Position(x=0, y=-300, z=30),
            powertrain=Powertrain.HYBRID, battery_pct=100, fuel_remaining_kg=3, fuel_pct=100, water_remaining_l=15,
            timestamp=self.clock.now())))

    def run(self, s=1.0):
        for _ in range(int(s / 0.05)):
            self.clock.t += 0.05
            self.bus.pump(self.clock.t)
            self.leader.tick(self.clock.t)

    def tasks(self, d, kind=None):
        return [m.body.task for m in self.inbox[d] if isinstance(m.body, TaskAssign)
                and (kind is None or m.body.task.type is kind)]

    def pictures(self, d):
        return [m.body.hotspot for m in self.inbox[d] if isinstance(m.body, HotspotReport)]

    def detect(self, hid="H1", x=0.0, y=100.0):
        self.send("s-near", HotspotReport(hotspot=Hotspot(
            hotspot_id=hid, position=Position(x=x, y=y), peak_temp_c=420, est_area_m2=2.0, confidence=0.97,
            first_seen=self.clock.now(), last_seen=self.clock.now())))

    def verdict(self, status, seeker="s-near", fid="H1"):
        f = self.leader.hotspots[fid]
        h = f.hotspot.model_copy(update={"status": status or HotspotStatus.DETECTED, "reported_by": seeker})
        self.send(seeker, HotspotReport(hotspot=h))
        self.send(seeker, TaskComplete(task_id=f.investigate_task, drone_id=seeker, success=True, result_status=status))


def test_detection_waits_for_a_close_range_look_by_the_nearest_seeker():
    bed = Bed()
    bed.heartbeats()
    bed.detect()
    bed.run(1.0)
    f = bed.leader.hotspots["H1"]
    assert f.hotspot.status is HotspotStatus.DETECTED            # high detector confidence is not enough any more
    inv = bed.tasks("s-near", TaskType.INVESTIGATE)
    assert len(inv) == 1 and inv[0].target_hotspot_id == "H1" and not bed.tasks("s-far")
    assert not bed.tasks("ext", TaskType.EXTINGUISH)              # no water on an unconfirmed fire
    bed.heartbeats({"s-near": inv[0].task_id})
    bed.verdict(HotspotStatus.CONFIRMED)
    bed.run(1.0)
    assert f.confirmed_by == "investigate" and f.hotspot.status in (HotspotStatus.CONFIRMED, HotspotStatus.ASSIGNED)
    assert bed.tasks("ext", TaskType.EXTINGUISH)[0].target_hotspot_id == "H1"


def test_inconclusive_goes_to_ground_review_and_the_operator_decides():
    bed = Bed()
    queue = GroundReviewQueue(bed.bus, bed.clock.now, publish_decisions=True)
    bed.heartbeats()
    bed.detect()
    bed.run(0.5)
    bed.verdict(None)                                              # inconclusive
    f = bed.leader.hotspots["H1"]
    bed.send("s-near", ReviewReport(review_id="s-near:H1", hotspot=f.hotspot, score=0.5, not_fire_below=0.2,
                                    fire_at_or_above=0.9))
    bed.run(0.5)
    assert f.review_since is not None and len(queue.pending()) == 1
    assert not bed.tasks("s-near", TaskType.INVESTIGATE)[1:]      # nobody re-investigates while under review
    queue.resolve("s-near:H1", ReviewDisposition.NOT_FIRE)
    bed.run(0.5)
    cmd = bed.ground_cmds[-1].body
    assert isinstance(cmd, GroundCommand) and cmd.command is CommandKind.MARK_FALSE_POSITIVE and cmd.target_id == "H1"
    assert f.hotspot.status is HotspotStatus.FALSE_POSITIVE and "H1" not in bed.leader.queue


def test_ground_confirm_fire_and_autonomous_timeout_without_a_ground_link():
    bed = Bed(review_timeout_s=5.0)
    bed.heartbeats()
    bed.detect("H1")
    bed.detect("H2", x=200.0, y=100.0)
    bed.run(0.5)
    for fid in ("H1", "H2"):
        f = bed.leader.hotspots[fid]
        h = f.hotspot.model_copy(update={"reported_by": f.investigator})
        bed.send(f.investigator, HotspotReport(hotspot=h))
        bed.send(f.investigator, TaskComplete(task_id=f.investigate_task, drone_id=f.investigator, success=True))
    bed.run(0.5)
    bed.send("ground", GroundCommand(command=CommandKind.CONFIRM_FIRE, target_id="H1"))
    bed.run(0.5)
    assert bed.leader.hotspots["H1"].confirmed_by == "ground"
    assert bed.leader.hotspots["H2"].hotspot.status is HotspotStatus.DETECTED     # still waiting...
    bed.run(5.0)
    assert bed.leader.hotspots["H2"].confirmed_by == "timeout"                    # ...the swarm doesn't wait forever


def test_dispatch_on_detect_holds_the_drop_until_confirmation():
    bed = Bed(dispatch_on_detect=True)
    bed.heartbeats()
    bed.detect()
    bed.run(1.0)
    assert bed.tasks("ext", TaskType.EXTINGUISH)                  # already flying...
    assert bed.pictures("ext")[-1].status is HotspotStatus.DETECTED   # ...but its picture says "not cleared"
    bed.verdict(HotspotStatus.CONFIRMED)
    bed.run(0.5)
    assert bed.pictures("ext")[-1].status is HotspotStatus.ASSIGNED   # cleared to drop


def test_false_positive_recalls_drones_already_on_the_way():
    bed = Bed(dispatch_on_detect=True)
    bed.heartbeats()
    bed.detect()
    bed.run(1.0)
    assert "ext" in bed.leader.assignments
    bed.verdict(HotspotStatus.FALSE_POSITIVE)
    bed.run(0.5)
    assert "ext" not in bed.leader.assignments and bed.leader.hotspots["H1"].hotspot.status is HotspotStatus.FALSE_POSITIVE


def test_review_queue_delivers_decisions_only_when_enabled():
    clock = SimClock()
    bus = SimBus(clock, latency_s=0.0)
    got = []
    bus.subscribe(SUB_GROUND_COMMANDS, got.append)
    for publish, n in ((False, 0), (True, 1)):
        q = GroundReviewQueue(bus, clock.now, publish_decisions=publish)
        h = Hotspot(hotspot_id=f"H{n}", position=Position(x=0, y=0), peak_temp_c=120, est_area_m2=1, confidence=0.5,
                    first_seen=0, last_seen=0)
        msg = SwarmMessage.create("s1", ReviewReport(review_id=f"s1:H{n}", hotspot=h, score=0.5, not_fire_below=0.2,
                                                     fire_at_or_above=0.9), 0.0)
        q.handle_message(msg)
        q.resolve(f"s1:H{n}", ReviewDisposition.FIRE)
        clock.t += 0.1
        bus.pump(clock.t)
        assert len(got) == n
        q.close()
    assert got[0].body.command is CommandKind.CONFIRM_FIRE and got[0].body.target_id == "H1"


def test_sim_ground_operator_approves_about_seventy_percent_deterministically():
    from sim.ground import SimGroundOperator
    outcomes = []
    for _ in range(2):
        clock = SimClock()
        bus = SimBus(clock, latency_s=0.0)
        op = SimGroundOperator(bus, clock, approve_frac=0.7, delay_s=1.0, seed=3)
        for i in range(200):
            h = Hotspot(hotspot_id=f"H{i}", position=Position(x=0, y=0), peak_temp_c=120, est_area_m2=1,
                        confidence=0.5, first_seen=0, last_seen=0)
            op.queue.handle_message(SwarmMessage.create("s1", ReviewReport(
                review_id=f"s1:H{i}", hotspot=h, score=0.5, not_fire_below=0.2, fire_at_or_above=0.9), 0.0))
        clock.t = 2.0
        op.tick(clock.t)
        outcomes.append(dict(op.stats))
    assert outcomes[0] == outcomes[1]                                 # seeded: same answers every run
    assert 0.62 <= outcomes[0]["fire"] / 200 <= 0.78
