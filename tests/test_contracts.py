"""Contract tests. Run from the repo root:  python -m pytest -q"""
import math

import pytest
from pydantic import ValidationError

from contracts import (
    BODY_TYPES, DEFAULT_AGENT_SPECS, DEFAULT_QOS, ROUTES, SENSOR_NAMES,
    Alert, AlertKind, AlertNotice, Autopilot, Bus, CameraModel, Clock, ClusterSnapshot, ClusterUpdate,
    CommandKind, Detection,
    DetectorKind, DroneKind, DroneState, DroneStatus, ExtinguishingAgent, GridCellUpdate,
    GridIndex, GridSpec, GroundCommand, Heartbeat, Hotspot, HotspotReport, HotspotStatus,
    LeaderClaim, MapSync, MessageType, PathClaim, Position, RefillGrant, RefillRequest, ReviewReport,
    RefillStation, RefillTargetKind, ScenarioConfig, Severity, SimFault, SimFaultKind,
    SwarmMessage, Task, TaskAck, TaskAnnounce, TaskAssign, TaskBid, TaskComplete, TaskStatus,
    TaskType, WaterSource, WaterSourceKind, WaterSourceOrigin, WaterSourceReport, Zone,
    decode, default_extinguisher_spec, default_seeker_spec, encode, example_scenario,
    sub_my_commands, topic_for, topic_matches, validate_pattern, CONTRACT_VERSION,
    PlatformSpec, PowerSystem, Powertrain,
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
        MessageType.REVIEW_REPORT: ReviewReport(
            review_id="d01:h1",
            hotspot=hotspot().model_copy(update={"confidence": 0.5}),
            score=0.5,
            not_fire_below=0.2,
            fire_at_or_above=0.8,
        ),
        MessageType.WATER_SOURCE_REPORT: WaterSourceReport(source=source()),
        MessageType.MAP_SYNC: MapSync(grid=grid, seq=3, cells=[GridCellUpdate(row=1, col=2, log_odds=2.0, last_seen=9.0)],
                                      hotspots=[hotspot()], water_sources=[source()]),
        MessageType.CLUSTER_UPDATE: ClusterUpdate(
            cluster=ClusterSnapshot(
                cluster_id="north",
                leader_id="d01",
                cluster_epoch=1,
                member_ids=["d01", "d02"],
                zone_ids=["z1"],
                updated_at=1.0,
                lease_until=5.0,
            )
        ),
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


def test_review_report_routes_to_ground_base():
    msg = SwarmMessage.create("d01", all_bodies()[MessageType.REVIEW_REPORT], now=10.0)
    assert topic_for(msg) == "ground/reviews"
    assert msg.reliable and msg.ttl_s == 300


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


def test_cluster_snapshot_validates_membership_and_lease():
    snapshot = ClusterSnapshot(
        cluster_id="north",
        leader_id="d01",
        cluster_epoch=2,
        member_ids=["d01", "d02"],
        zone_ids=["z1"],
        updated_at=10.0,
        lease_until=15.0,
    )
    assert snapshot.leader_id in snapshot.member_ids
    with pytest.raises(ValidationError):
        ClusterSnapshot(
            cluster_id="north",
            leader_id="d99",
            cluster_epoch=2,
            member_ids=["d01"],
            updated_at=10.0,
            lease_until=15.0,
        )
    with pytest.raises(ValidationError):
        ClusterSnapshot(
            cluster_id="north",
            leader_id="d01",
            cluster_epoch=2,
            member_ids=["d01", "d01"],
            updated_at=10.0,
            lease_until=15.0,
        )
    with pytest.raises(ValidationError):
        ClusterSnapshot(
            cluster_id="north",
            leader_id="d01",
            cluster_epoch=2,
            member_ids=["d01"],
            updated_at=15.0,
            lease_until=15.0,
        )


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


def test_cluster_messages_route_to_cluster_scoped_topics():
    snapshot = ClusterSnapshot(
        cluster_id="north",
        leader_id="d01",
        cluster_epoch=2,
        member_ids=["d01"],
        updated_at=1.0,
        lease_until=5.0,
    )
    update = SwarmMessage.create("d01", ClusterUpdate(cluster=snapshot), now=1.0)
    claim = SwarmMessage.create(
        "d01",
        LeaderClaim(candidate_id="d01", epoch=2, score=0.9, cluster_id="north", cluster_epoch=2),
        now=1.0,
    )
    announce = SwarmMessage.create(
        "d01",
        TaskAnnounce(task=task().model_copy(update={"cluster_id": "north"}), bid_deadline=3.0),
        now=1.0,
    )
    bid = SwarmMessage.create(
        "d02",
        TaskBid(task_id="t1", bidder_id="d02", cost=0.2, cluster_id="north", cluster_epoch=2),
        now=1.0,
    )

    assert topic_for(update) == "swarm/clusters/north/state"
    assert topic_for(claim) == "swarm/clusters/north/leader"
    assert topic_for(announce) == "swarm/clusters/north/tasks/announce"
    assert topic_for(bid) == "swarm/clusters/north/tasks/bids"


def test_unscoped_messages_keep_existing_topics_and_cluster_ids_are_safe():
    announce = SwarmMessage.create(
        "d01",
        TaskAnnounce(task=task(), bid_deadline=3.0),
        now=1.0,
    )
    assert topic_for(announce) == "swarm/tasks/announce"
    with pytest.raises(ValidationError):
        ClusterSnapshot(
            cluster_id="north/zone",
            leader_id="d01",
            cluster_epoch=0,
            member_ids=["d01"],
            updated_at=1.0,
            lease_until=2.0,
        )


# ---------- config ----------
def test_example_scenario_is_valid_and_round_trips():
    cfg = example_scenario()
    again = ScenarioConfig.model_validate_json(cfg.model_dump_json())
    assert again == cfg
    assert cfg.grid.contains(cfg.base_position)


def test_default_platforms_are_physically_consistent():
    ext, seek = default_extinguisher_spec(), default_seeker_spec()
    assert ext.empty_mass_kg + ext.water_capacity_l + ext.fuel_capacity_kg <= ext.max_takeoff_kg
    assert seek.water_capacity_l == 0
    assert 5 * 3600 <= ext.endurance_s <= 8 * 3600     # provider: generator gives 5-8 h
    assert ext.stopping_distance_m(20.0) == pytest.approx(100.0)
    assert ext.stopping_distance_m(5.0) == pytest.approx(6.25)
    assert seek.stopping_distance_m(15.0) == pytest.approx(28.125)
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


# ---------- powertrains, fuel telemetry, refuel ----------
def test_platform_roles_and_powertrains():
    ext, seek = default_extinguisher_spec(), default_seeker_spec()
    assert ext.powertrain is Powertrain.HYBRID and seek.powertrain is Powertrain.ELECTRIC
    assert seek.search_capable and not ext.search_capable
    assert ext.thermal.camera.width_px < seek.thermal.camera.width_px      # confirm, not detect
    assert set(ext.sensor_names) <= set(SENSOR_NAMES) and "power" in ext.sensor_names
    assert ext.max_takeoff_kg > 50 * seek.max_takeoff_kg / 10               # ultra-heavy vs light


def test_hybrid_endurance_math_matches_rating():
    ext = default_extinguisher_spec()
    full = ext.hybrid_endurance_s()
    assert full == pytest.approx(ext.endurance_s, rel=0.03)                 # rated 5 h
    assert ext.hybrid_endurance_s(reserve=True) == pytest.approx(0.82 * full, rel=0.03)
    assert ext.hybrid_endurance_s(water_l=0.0) > full                       # lighter -> longer
    assert ext.fuel_burn_kg_h(ext.max_takeoff_kg) == pytest.approx(52.0)
    assert ext.fuel_burn_kg_h(400.0) == pytest.approx(52.0 * 0.5 ** 1.5)


def test_electric_endurance_math_matches_rating():
    seek = default_seeker_spec()
    assert seek.electric_endurance_s() == pytest.approx(seek.endurance_s, rel=0.02)
    assert seek.electric_endurance_s(hover=True) < seek.electric_endurance_s()
    with pytest.raises(ValueError):
        seek.fuel_burn_kg_h(10.0)


def test_platform_validation_for_powertrains():
    base = default_extinguisher_spec().model_dump()
    bad = dict(base, fuel_capacity_kg=300.0)                                 # 420 + 170 + 300 > 800
    with pytest.raises(ValidationError):
        PlatformSpec.model_validate(bad)
    with pytest.raises(ValidationError):
        PlatformSpec.model_validate(dict(base, fuel_burn_kg_h_at_mtow=None))  # HYBRID needs a burn model
    with pytest.raises(ValidationError):
        PlatformSpec.model_validate(dict(base, sensor_names=["thermal", "sonar"]))
    seeker = default_seeker_spec().model_dump()
    with pytest.raises(ValidationError):
        PlatformSpec.model_validate(dict(seeker, battery_wh=None))
    with pytest.raises(ValidationError):
        PlatformSpec.model_validate(dict(seeker, fuel_capacity_kg=1.0))


def test_drone_state_fuel_rules():
    hybrid = DroneState(drone_id="x01", kind=DroneKind.EXTINGUISHER, position=pos(), battery_pct=95.0,
                        powertrain=Powertrain.HYBRID, fuel_remaining_kg=180.0, fuel_pct=85.7,
                        generator_output_kw=96.0, water_remaining_l=170.0,
                        sensors_ok={"power": True, "payload": True}, timestamp=1.0)
    assert DroneState.model_validate_json(hybrid.model_dump_json()) == hybrid
    with pytest.raises(ValidationError):   # HYBRID must report fuel
        DroneState(drone_id="x01", kind=DroneKind.EXTINGUISHER, position=pos(), battery_pct=90.0,
                   powertrain=Powertrain.HYBRID, timestamp=1.0)
    with pytest.raises(ValidationError):   # ELECTRIC must not
        DroneState(drone_id="d01", kind=DroneKind.SEEKER, position=pos(), battery_pct=90.0,
                   fuel_remaining_kg=1.0, fuel_pct=1.0, timestamp=1.0)


def test_refuel_and_partial_drop_fields_roundtrip():
    req = RefillRequest(drone_id="x01", position=pos(), water_remaining_l=0.0,
                        fuel_remaining_kg=40.0, needs_fuel=True)
    msg = SwarmMessage.create(sender_id="x01", body=req, now=5.0, seq=1)
    assert decode(encode(msg)).body == req
    task = Task(task_id="t1", type=TaskType.EXTINGUISH, target_hotspot_id="h1", created_at=0.0, volume_l=30.0)
    assert Task.model_validate_json(task.model_dump_json()).volume_l == 30.0
    with pytest.raises(ValidationError):
        Task(task_id="t2", type=TaskType.EXTINGUISH, target_hotspot_id="h1", created_at=0.0, volume_l=0.0)
    st = example_scenario().stations[0]
    assert st.fuel_available_kg > 0 and st.battery_swap_s is not None
    assert AlertKind.LOW_FUEL in AlertKind and AlertKind.POWERPLANT_FAULT in AlertKind


def test_power_system_protocol():
    class P:
        def battery_pct(self): return 90.0
        def fuel_remaining_kg(self): return None
        def generator_output_kw(self): return None
        def endurance_remaining_s(self): return 100.0
    assert isinstance(P(), PowerSystem) and not isinstance(object(), PowerSystem)
