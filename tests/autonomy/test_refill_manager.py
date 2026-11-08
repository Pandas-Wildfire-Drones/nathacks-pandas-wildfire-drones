from autonomy.refill_manager import (
    execute_water_refill,
    validate_refill_station,
    validate_water_source,
)
from contracts.entities import (
    DroneState,
    RefillStation,
    WaterSource,
)
from contracts.enums import (
    DroneKind,
    ExtinguishingAgent,
    Powertrain,
    WaterSourceKind,
    WaterSourceOrigin,
    WaterSourceStatus,
)
from contracts.geometry import Position


class FakeAutopilot:
    def __init__(self, at_target=False):
        self._at_target = at_target
        self.goto_calls = []
        self.hold_calls = 0

    def position(self):
        return Position(x=0, y=0, z=10)

    def heading_deg(self):
        return 0.0

    def speed_mps(self):
        return 0.0

    def battery_pct(self):
        return 80.0

    def endurance_remaining_s(self):
        return 1000.0

    def goto(self, target, speed_mps=None):
        self.goto_calls.append((target, speed_mps))

    def hold(self):
        self.hold_calls += 1

    def return_to_launch(self):
        pass

    def at_target(self, tolerance_m=2.0):
        return self._at_target


class FakePayload:
    def __init__(self, returned_volume):
        self.returned_volume = returned_volume
        self.fill_calls = []

    def water_remaining_l(self):
        return 20.0

    def agent(self):
        return ExtinguishingAgent.WATER

    def release(self, volume_l, at):
        return volume_l

    def fill(self, volume_l, agent=ExtinguishingAgent.WATER):
        self.fill_calls.append((volume_l, agent))
        return self.returned_volume


def make_drone():
    return DroneState(
        drone_id="drone_02",
        kind=DroneKind.EXTINGUISHER,
        position=Position(x=0, y=0, z=10),
        powertrain=Powertrain.HYBRID,
        battery_pct=80,
        fuel_remaining_kg=100,
        fuel_pct=50,
        water_remaining_l=5,
        agent=ExtinguishingAgent.WATER,
        timestamp=0,
    )


def make_source(status=WaterSourceStatus.VALIDATED):
    return WaterSource(
        source_id="src_01",
        position=Position(x=50, y=50, z=0),
        kind=WaterSourceKind.POND,
        origin=WaterSourceOrigin.DETECTED,
        status=status,
        est_volume_l=10000,
        confidence=0.95,
    )


def make_station():
    return RefillStation(
        station_id="station_01",
        position=Position(x=100, y=100, z=0),
        water_available_l=1000,
        fill_rate_lpm=100,
        total_bays=2,
        free_bays=1,
        agents_available=[ExtinguishingAgent.WATER],
    )


def test_validated_water_source_is_allowed():
    result = validate_water_source(make_source())
    assert result.allowed


def test_unvalidated_water_source_is_rejected():
    result = validate_water_source(
        make_source(WaterSourceStatus.CANDIDATE)
    )
    assert not result.allowed


def test_station_requires_free_bay():
    station = make_station()
    station.free_bays = 0

    result = validate_refill_station(
        station,
        ExtinguishingAgent.WATER,
    )

    assert not result.allowed


def test_station_requires_water():
    station = make_station()
    station.water_available_l = 0

    result = validate_refill_station(
        station,
        ExtinguishingAgent.WATER,
    )

    assert not result.allowed


def test_refill_navigates_when_not_at_target():
    autopilot = FakeAutopilot(at_target=False)
    payload = FakePayload(returned_volume=20)

    result = execute_water_refill(
        make_drone(),
        autopilot,
        payload,
        make_source().position,
        20,
        source=make_source(),
    )

    assert result.status == "TRANSIT"
    assert not result.success
    assert len(autopilot.goto_calls) == 1
    assert not payload.fill_calls


def test_refill_fills_after_arrival():
    autopilot = FakeAutopilot(at_target=True)
    payload = FakePayload(returned_volume=20)

    result = execute_water_refill(
        make_drone(),
        autopilot,
        payload,
        make_source().position,
        20,
        source=make_source(),
    )

    assert result.status == "DONE"
    assert result.success
    assert result.water_added_l == 20
    assert len(payload.fill_calls) == 1
    assert autopilot.hold_calls == 1


def test_refill_does_not_exceed_requested_volume():
    autopilot = FakeAutopilot(at_target=True)
    payload = FakePayload(returned_volume=21)

    try:
        execute_water_refill(
            make_drone(),
            autopilot,
            payload,
            make_source().position,
            20,
            source=make_source(),
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "more water than requested" in str(exc)


def test_zero_refill_volume_is_rejected():
    result = execute_water_refill(
        make_drone(),
        FakeAutopilot(at_target=True),
        FakePayload(returned_volume=0),
        make_source().position,
        0,
        source=make_source(),
    )

    assert not result.success
    assert result.status == "REJECTED"


def test_refill_station_can_be_used():
    autopilot = FakeAutopilot(at_target=True)
    payload = FakePayload(returned_volume=20)

    result = execute_water_refill(
        make_drone(),
        autopilot,
        payload,
        make_station().position,
        20,
        station=make_station(),
    )

    assert result.success
    assert result.water_added_l == 20


def test_refill_requires_source_or_station():
    result = execute_water_refill(
        make_drone(),
        FakeAutopilot(at_target=True),
        FakePayload(returned_volume=20),
        Position(x=0, y=0, z=0),
        20,
    )

    assert not result.success
    assert result.status == "REJECTED"
