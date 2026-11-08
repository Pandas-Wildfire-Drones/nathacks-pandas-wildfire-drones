from contracts.entities import Alert, DroneState, Hotspot, Position, Task
from contracts.enums import (
    AlertKind,
    DroneKind,
    DroneStatus,
    ExtinguishingAgent,
    HotspotStatus,
    Powertrain,
    Severity,
    SwarmRole,
    TaskStatus,
    TaskType,
)
from contracts.messages import (
    AlertNotice,
    Heartbeat,
    HotspotReport,
    RefillGrant,
    SwarmMessage,
    TaskAssign,
)
from autonomy.swarm_adapter import SwarmAutonomyAdapter


class FakeBus:
    def __init__(self):
        self.subscriptions = []
        self.published = []

    def subscribe(self, pattern, handler):
        self.subscriptions.append((pattern, handler))

    def publish(self, topic, message):
        self.published.append((topic, message))


class FakeController:
    def __init__(self):
        self.autopilot = None
        self.begin_calls = []

    def begin_extinguish(self, drone, hotspot, task):
        self.begin_calls.append((drone, hotspot, task))
        return None

    def step_extinguish(self, drone, hotspot, task):
        raise AssertionError("step_extinguish should not be called in this test")


def make_drone():
    return DroneState(
        drone_id="drone-1",
        kind=DroneKind.EXTINGUISHER,
        role=SwarmRole.FOLLOWER,
        position=Position(x=0, y=0, z=20),
        heading_deg=0,
        speed_mps=10,
        powertrain=Powertrain.ELECTRIC,
        battery_pct=80,
        endurance_remaining_s=600,
        water_remaining_l=20,
        agent=ExtinguishingAgent.WATER,
        status=DroneStatus.IDLE,
        current_task_id=None,
        leader_id="drone-leader",
        epoch=3,
        sensors_ok={"thermal": True, "lidar": True},
        link_quality=1.0,
        timestamp=100.0,
    )


def make_hotspot():
    return Hotspot(
        hotspot_id="hotspot-1",
        position=Position(x=100, y=100, z=0),
        peak_temp_c=700,
        est_area_m2=100,
        confidence=0.9,
        status=HotspotStatus.DETECTED,
        priority_score=1.0,
        observations=1,
        growth_rate_m2_per_min=1.0,
        assigned_to=None,
        reported_by="drone-2",
        first_seen=90.0,
        last_seen=100.0,
    )


def make_task():
    return Task(
        task_id="task-1",
        type=TaskType.EXTINGUISH,
        priority=1,
        status=TaskStatus.ASSIGNED,
        assigned_to="drone-1",
        target_hotspot_id="hotspot-1",
        created_at=100.0,
        deadline=200.0,
        epoch=3,
        volume_l=5.0,
    )


def test_adapter_subscribes_to_required_swarm_topics():
    bus = FakeBus()

    SwarmAutonomyAdapter(
        make_drone(),
        FakeController(),
        bus,
    )

    topics = {topic for topic, _ in bus.subscriptions}

    assert "swarm/task_assign" in topics
    assert "swarm/heartbeat" in topics
    assert "swarm/hotspot_report" in topics
    assert "swarm/refill_grant" in topics
    assert "swarm/path_claims" in topics


def test_heartbeat_updates_local_state_without_physical_action():
    bus = FakeBus()
    adapter = SwarmAutonomyAdapter(make_drone(), FakeController(), bus)

    heartbeat = Heartbeat(state=make_drone())

    message = SwarmMessage.create(
        sender_id="drone-2",
        body=heartbeat,
        now=101.0,
        epoch=3,
    )

    result = adapter.handle_message(message)

    assert result is None
    assert adapter.state.latest_heartbeat == heartbeat
    assert bus.published == []


def test_hotspot_report_updates_local_target():
    bus = FakeBus()
    adapter = SwarmAutonomyAdapter(make_drone(), FakeController(), bus)

    hotspot = make_hotspot()

    report = HotspotReport(hotspot=hotspot)

    message = SwarmMessage.create(
        sender_id="drone-2",
        body=report,
        now=101.0,
        epoch=3,
    )

    adapter.handle_message(message)

    assert adapter.state.current_hotspot == hotspot


def test_wrong_drone_task_is_ignored():
    bus = FakeBus()
    controller = FakeController()
    adapter = SwarmAutonomyAdapter(make_drone(), controller, bus)

    task = make_task()
    task = task.model_copy(update={"assigned_to": "drone-2"})

    assignment = TaskAssign(
        task=task,
        epoch=3,
    )

    message = SwarmMessage.create(
        sender_id="swarm-leader",
        body=assignment,
        now=101.0,
        epoch=3,
    )

    result = adapter.handle_message(message)

    assert result is None
    assert controller.begin_calls == []
    assert adapter.state.current_task is None
    assert bus.published == []


def test_alerts_use_valid_shared_alert_contract():
    bus = FakeBus()
    adapter = SwarmAutonomyAdapter(make_drone(), FakeController(), bus)

    adapter._publish_alert(None, "test safety rejection")

    assert len(bus.published) == 1

    topic, message = bus.published[0]

    assert isinstance(message.body, AlertNotice)
    assert isinstance(message.body.alert, Alert)

    alert = message.body.alert

    assert alert.severity is Severity.WARNING
    assert alert.kind is AlertKind.GENERIC
    assert alert.subject_id == "drone-1"
    assert alert.message == "test safety rejection"
    assert alert.timestamp == 100.0


def test_other_drone_refill_grant_is_ignored():
    bus = FakeBus()
    adapter = SwarmAutonomyAdapter(make_drone(), FakeController(), bus)

    message = SwarmMessage.create(
        sender_id="swarm-leader",
        body=RefillGrant(
            drone_id="drone-2",
            target_kind="WATER_SOURCE",
            target_id="source-1",
            position=Position(x=0, y=0, z=0),
            bay=1,
            eta_s=10,
            queue_wait_s=0,
            epoch=3,
        ),
        now=101.0,
        epoch=3,
    )

    adapter.handle_message(message)

    # A valid refill grant for another drone must not alter this drone's
    # local refill state or trigger a safety alert.
    assert adapter.state.latest_refill_grant is None
    assert bus.published == []
