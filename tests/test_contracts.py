"""Contract tests. Run from the repo root:  python -m pytest -q"""
import math

import pytest
from pydantic import ValidationError

from contracts import (
    BODY_TYPES, DEFAULT_AGENT_SPECS, DEFAULT_QOS, ROUTES, SENSOR_NAMES,
    Alert, AlertKind, AlertNotice, Autopilot, Bus, CameraModel, Clock, CommandKind, Detection,
    DetectorKind, DroneKind, DroneState, DroneStatus, ExtinguishingAgent, GridCellUpdate,
    GridIndex, GridSpec, GroundCommand, Heartbeat, Hotspot, HotspotReport, HotspotStatus,
    LeaderClaim, MapSync, MessageType, PathClaim, Position, RefillGrant, RefillRequest,
    RefillStation, RefillTargetKind, ScenarioConfig, Severity, SimFault, SimFaultKind,
    SwarmMessage, Task, TaskAck, TaskAnnounce, TaskAssign, TaskBid, TaskComplete, TaskStatus,
    TaskType, WaterSource, WaterSourceKind, WaterSourceOrigin, WaterSourceReport, Zone,
    decode, default_extinguisher_spec, default_seeker_spec, encode, example_scenario,
    sub_my_commands, topic_for, topic_matches, validate_pattern, CONTRACT_VERSION,
)


# ---------- sample builders ----------
def pos(x=1.0, y=2.0, z=60.0):
    return Position(x=x, y=y, z=z)


def state():
    return DroneState(drone_id="d01", kind=DroneKind.SEEKER, position=pos(), battery_pct=88.0,
                      status=DroneStatus.SCANNING, sensors_ok={"thermal": True, "lidar": True},
                      timestamp=12.5)


def hotspot():
    return Hotspot(hotspot_id="h1", position=pos(100, 200, 0), peak_temp_c=412.0, est_area_m2=3.5,
                   confidence=0.93, first_seen=10.0, last_seen=12.0)


def source():
    return WaterSource(source_id="w1", position=pos(5, 5, 0), kind=WaterSourceKind.POND,
                       origin=WaterSourceOrigin.PRIOR_MAP, est_volume_l=1e6)


def task(kind=TaskType.EXTINGUISH):
    kw = {"target_hotspot_id": "h1"}
    if kind is TaskType.SCAN_ZONE:
        kw = {"target_zone_id": "z1"}
    return Task(task_id="t1", type=kind, created_at=5.0, **kw)


def all_bodies():
    grid = GridSpec(origin=pos(0, 0, 0), resolution_m=2.0, rows=10, cols=10)
    return {
        MessageType.HEARTBEAT: Heartbeat(state=state()),
        MessageType.HOTSPOT_REPORT: HotspotReport(hotspot=hotspot()),
        MessageType.WATER_SOURCE_REPORT: WaterSourceReport(source=source()),
        MessageType.MAP_SYNC: MapSync(grid=grid, seq=3, cells=[GridCellUpdate(row=1, col=2, log_odds=2.0, last_seen=9.0)],
                                      hotspots=[hotspot()], water_sources=[source()]),
        MessageType.TASK_ANNOUNCE: TaskAnnounce(task=task(), bid_deadline=8.0),
        MessageType.TASK_BID: TaskBid(task_id="t1", bidder_id="d02", cost=12.3, eta_s=40.0),
        MessageType.TASK_ASSIGN: TaskAssign(task=task(), epoch=2),
        MessageType.TASK_ACK: TaskAck(task_id="t1", drone_id="d02", accepted=True, epoch=2),
        MessageType.TASK_COMPLETE: TaskComplete(task_id="t1", drone_id="d02", success=True,
                                                result_status=HotspotStatus.SUPPRESSED, water_used_l=40.0),
        MessageType.PATH_CLAIM: PathClaim(drone_id="d02", waypoints=[pos(), pos(5, 5)], valid_from=1.0,
                                          valid_until=9.0, altitude_band_m=(50.0, 70.0)),
        MessageType.LEADER_CLAIM: LeaderClaim(candidate_id="d01", epoch=2, score=0.9),
        MessageType.REFILL_REQUEST: RefillRequest(drone_id="d02", position=pos(), water_remaining_l=3.0),
        MessageType.REFILL_GRANT: RefillGrant(drone_id="d02", target_kind=RefillTargetKind.WATER_SOURCE,
                                              target_id="w1", position=pos(5, 5, 0), bay=None, eta_s=60.0),
        MessageType.GROUND_COMMAND: GroundCommand(command=CommandKind.CONFIRM_OUT, target_id="h1",
                                                  params={"note": "checked by crew", "ok": True}),
        MessageType.ALERT: AlertNotice(alert=Alert(alert_id="a1", severity=Severity.WARNING,
                                                   kind=AlertKind.DRONE_LOST, subject_id="d03",
                                                   message="d03 missed 5 heartbeats", timestamp=30.0)),
        MessageType.SIM_FAULT: SimFault(fault=SimFaultKind.KILL_LEADER, duration_s=60.0),
    }


# ---------- completeness: nothing added to one table but forgotten in another ----------
def test_every_message_type_is_fully_wired():
    all_types = set(MessageType)
    assert set(BODY_TYPES) == all_types
    assert set(DEFAULT_QOS) == all_types
    assert set(ROUTES) == all_types
    assert set(all_bodies()) == all_types  # the samples in this file cover every type too


def test_body_type_field_matches_registry():
    for mtype, cls in BODY_TYPES.items():
        assert cls.model_fields["type"].default == mtype


# ---------- round trips ----------
@pytest.mark.parametrize("mtype", list(MessageType))
def test_round_trip_every_message_type(mtype):
    body = all_bodies()[mtype]
    msg = SwarmMessage.create("d01", body, now=1.0, seq=7, receiver_id="d02", epoch=2)
    raw = encode(msg)
    back = decode(raw)
    assert back == msg
    assert back is not msg                       # no shared objects across the boundary
    assert back.msg_type == mtype
    assert type(back.body) is type(body)
    assert back.msg_id == "d01-7"
    topic = topic_for(back)
    assert topic_matches(topic, topic)


def test_decode_accepts_str_and_bytes():
    msg = SwarmMessage.create("d01", Heartbeat(state=state()), now=0.0)
    assert decode(encode(msg)) == decode(encode(msg).decode())


def test_discriminator_rejects_unknown_and_malformed():
    msg = SwarmMessage.create("d01", Heartbeat(state=state()), now=0.0)
    good = encode(msg).decode()
    with pytest.raises(ValidationError):
        decode(good.replace('"type":"HEARTBEAT"', '"type":"NOPE"'))
    with pytest.raises(ValidationError):
        decode(good.replace('"battery_pct":88.0', '"battery_pct":"banana"'))
    with pytest.raises(ValidationError):
        decode(good.replace('"battery_pct":88.0', '"battery_pct":188.0'))


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Hotspot(hotspot_id="h", position=pos(), peak_temp_c=300, confidence=0.5, first_seen=0,
                last_seen=0, peak_temp=300)  # typo'd extra field


def test_messages_are_frozen():
    msg = SwarmMessage.create("d01", Heartbeat(state=state()), now=0.0)
    with pytest.raises(ValidationError):
        msg.sender_id = "other"


def test_expiry_and_default_qos():
    hb = SwarmMessage.create("d01", Heartbeat(state=state()), now=10.0)
    assert hb.reliable is False and hb.ttl_s == DEFAULT_QOS[MessageType.HEARTBEAT].ttl_s
    assert not hb.is_expired(10.0 + hb.ttl_s - 0.1)
    assert hb.is_expired(10.0 + hb.ttl_s + 0.1)
    rep = SwarmMessage.create("d01", HotspotReport(hotspot=hotspot()), now=0.0)
    assert rep.reliable is True
    assert rep.v == CONTRACT_VERSION


# ---------- entity validation ----------
def test_hotspot_validation():
    with pytest.raises(ValidationError):
        Hotspot(hotspot_id="h", position=pos(), peak_temp_c=5000, confidence=0.5, first_seen=0, last_seen=1)
    with pytest.raises(ValidationError):
        Hotspot(hotspot_id="h", position=pos(), peak_temp_c=300, confidence=1.7, first_seen=0, last_seen=1)
    with pytest.raises(ValidationError):
        Hotspot(hotspot_id="h", position=pos(), peak_temp_c=300, confidence=0.5, first_seen=5, last_seen=1)
    h = hotspot()
    with pytest.raises(ValidationError):  # validate_assignment
        h.confidence = 3.0


def test_position_rejects_nan_and_is_hashable():
    with pytest.raises(ValidationError):
        Position(x=float("nan"), y=0)
    assert {pos(): 1}[pos()] == 1
    assert pos(0, 0, 0).distance_to(pos(3, 4, 0)) == pytest.approx(5.0)
    assert pos(0, 0, 0).distance_to(pos(3, 4, 12), horizontal=True) == pytest.approx(5.0)


def test_task_requires_matching_target():
    with pytest.raises(ValidationError):
        Task(task_id="t", type=TaskType.EXTINGUISH, created_at=0)
    with pytest.raises(ValidationError):
        Task(task_id="t", type=TaskType.SCAN_ZONE, created_at=0)
    with pytest.raises(ValidationError):
        Task(task_id="t", type=TaskType.REFILL, created_at=0)
    Task(task_id="t", type=TaskType.REFILL, created_at=0, target_water_source_id="w1")
    Task(task_id="t", type=TaskType.REFILL, created_at=0, target_station_id="s1")
    Task(task_id="t", type=TaskType.SURVEY_WATER, created_at=0, target_water_source_id="w1")
    assert task(TaskType.SCAN_ZONE).status is TaskStatus.PENDING


def test_zone_contains():
    z = Zone(zone_id="z", boundary=[pos(0, 0), pos(10, 0), pos(10, 10), pos(0, 10)])
    assert z.contains(pos(5, 5)) and not z.contains(pos(15, 5))
    with pytest.raises(ValidationError):
        Zone(zone_id="z", boundary=[pos(0, 0), pos(1, 1)])


def test_refill_station_bays():
    with pytest.raises(ValidationError):
        RefillStation(station_id="s", position=pos(), water_available_l=1, fill_rate_lpm=10,
                      total_bays=1, free_bays=2)


def test_path_claim_window():
    with pytest.raises(ValidationError):
        PathClaim(drone_id="d", waypoints=[pos()], valid_from=5, valid_until=5)


# ---------- grid ----------
def test_grid_round_trip_and_bounds():
    g = GridSpec(origin=Position(x=-100, y=-50), resolution_m=2.0, rows=50, cols=100)
    assert g.shape == (50, 100) and g.width_m == 200 and g.height_m == 100
    for r, c in [(0, 0), (49, 99), (10, 20), (25, 3)]:
        idx = GridIndex(row=r, col=c)
        assert g.world_to_cell(g.cell_to_world(idx)) == idx
    assert g.world_to_cell(Position(x=-100.0, y=-50.0)) == GridIndex(row=0, col=0)
    assert g.world_to_cell(Position(x=100.0, y=0.0)) is None   # x upper bound exclusive
    assert g.world_to_cell(Position(x=-101, y=0)) is None
    assert g.world_to_cell(Position(x=0, y=50.0)) is None


def test_row_zero_is_south():
    g = GridSpec(origin=Position(x=0, y=0), resolution_m=1.0, rows=10, cols=10)
    assert g.world_to_cell(Position(x=0.5, y=0.5)).row == 0
    assert g.world_to_cell(Position(x=0.5, y=9.5)).row == 9


# ---------- camera math ----------
def test_camera_footprint_math():
    cam = CameraModel(width_px=640, height_px=512, hfov_deg=32.0)
    assert cam.swath_m(100) == pytest.approx(2 * 100 * math.tan(math.radians(16)))
    assert cam.ground_sample_distance_m(100) == pytest.approx(0.0896, abs=1e-3)
    assert cam.ground_sample_distance_m(200) == pytest.approx(2 * cam.ground_sample_distance_m(100))
    assert 0 < cam.vfov_deg < cam.hfov_deg


def test_detection_ranges():
    Detection(u=1, v=2, area_px=5, peak_temp_c=300, confidence=0.9, source=DetectorKind.THRESHOLD)
    with pytest.raises(ValidationError):
        Detection(u=1, v=2, area_px=5, peak_temp_c=300, confidence=1.5, source=DetectorKind.THRESHOLD)


# ---------- topics ----------
@pytest.mark.parametrize("pattern,topic,expected", [
    ("drone/*/heartbeat", "drone/d01/heartbeat", True),
    ("drone/*/heartbeat", "drone/d01/hotspots", False),
    ("drone/*/heartbeat", "drone/heartbeat", False),
    ("drone/*/heartbeat", "drone/d01/x/heartbeat", False),
    ("swarm/#", "swarm/tasks/bids", True),
    ("swarm/#", "swarm", True),
    ("swarm/#", "ground/alerts", False),
    ("ground/commands", "ground/commands", True),
    ("ground/commands", "ground/commands/extra", False),
    ("#", "anything/at/all", True),
])
def test_topic_matching(pattern, topic, expected):
    assert topic_matches(pattern, topic) is expected


def test_validate_pattern():
    validate_pattern("swarm/#")
    validate_pattern("drone/*/cmd")
    with pytest.raises(ValueError):
        validate_pattern("swarm/#/x")
    with pytest.raises(ValueError):
        validate_pattern("swarm//x")


def test_directed_messages_need_receiver():
    assign = SwarmMessage.create("d01", all_bodies()[MessageType.TASK_ASSIGN], now=0.0)
    with pytest.raises(ValueError):
        topic_for(assign)
    directed = SwarmMessage.create("d01", all_bodies()[MessageType.TASK_ASSIGN], now=0.0, receiver_id="d05")
    assert topic_for(directed) == sub_my_commands("d05")


# ---------- config ----------
def test_example_scenario_is_valid_and_round_trips():
    cfg = example_scenario()
    again = ScenarioConfig.model_validate_json(cfg.model_dump_json())
    assert again == cfg
    assert cfg.grid.contains(cfg.base_position)


def test_default_platforms_are_physically_consistent():
    ext, seek = default_extinguisher_spec(), default_seeker_spec()
    assert ext.empty_mass_kg + ext.water_capacity_l <= ext.max_takeoff_kg
    assert seek.water_capacity_l == 0
    assert 5 * 3600 <= ext.endurance_s <= 8 * 3600     # provider: generator gives 5-8 h
    assert ext.stopping_distance_m(15.0) == pytest.approx(37.5)
    assert ext.stopping_distance_m(5.0) == pytest.approx(25 / 6)
    assert ExtinguishingAgent.WATER in DEFAULT_AGENT_SPECS and len(SENSOR_NAMES) == len(set(SENSOR_NAMES))


def test_scenario_rejects_bad_config():
    cfg = example_scenario().model_dump()
    cfg["fires"].append(dict(cfg["fires"][0]))
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(cfg)
    cfg2 = example_scenario().model_dump()
    cfg2["base_position"] = {"x": 99999.0, "y": 0.0, "z": 0.0}
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(cfg2)


# ---------- protocols ----------
def test_protocols_are_runtime_checkable():
    class C:
        def now(self) -> float:
            return 0.0

    class B:
        def publish(self, topic, msg): ...
        def subscribe(self, pattern, handler): ...

    assert isinstance(C(), Clock) and isinstance(B(), Bus)
    assert not isinstance(object(), Autopilot)
