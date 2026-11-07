"""Bus messages: one frozen body model per MessageType, joined by a discriminated union.

The `type` field inside each body is the discriminator: Pydantic reads it and picks the
right model, so a malformed HEARTBEAT is rejected at the boundary instead of crashing a
handler later. Use `SwarmMessage.create(...)` to build messages and `encode`/`decode` at
every bus boundary (even in-process: it guarantees drones never share Python objects).

Ordering/identity rules
  * Each sender keeps its own increasing `seq`; msg_id defaults to f"{sender_id}-{seq}" (deterministic).
  * Commands from a leader carry `epoch`. Receivers IGNORE commands whose epoch is lower than the
    highest epoch they have seen (prevents a stale leader from acting after a failover).
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from .base import CONTRACT_VERSION, FrozenContract
from .entities import Alert, DroneState, GridCellUpdate, Hotspot, Task, WaterSource
from .enums import (
    CommandKind, ExtinguishingAgent, HotspotStatus, MessageType, RefillTargetKind, SimFaultKind,
)
from .geometry import GridSpec, Position

__all__ = [
    "Heartbeat", "HotspotReport", "WaterSourceReport", "MapSync", "TaskAnnounce", "TaskBid",
    "TaskAssign", "TaskAck", "TaskComplete", "PathClaim", "LeaderClaim", "RefillRequest",
    "RefillGrant", "GroundCommand", "AlertNotice", "SimFault",
    "Body", "BODY_TYPES", "QoSProfile", "DEFAULT_QOS", "SwarmMessage", "encode", "decode",
]


class _Body(FrozenContract):
    """Base for all message bodies. Subclasses add a Literal `type` discriminator."""


class Heartbeat(_Body):
    type: Literal[MessageType.HEARTBEAT] = MessageType.HEARTBEAT
    state: DroneState


class HotspotReport(_Body):
    type: Literal[MessageType.HOTSPOT_REPORT] = MessageType.HOTSPOT_REPORT
    hotspot: Hotspot


class WaterSourceReport(_Body):
    type: Literal[MessageType.WATER_SOURCE_REPORT] = MessageType.WATER_SOURCE_REPORT
    source: WaterSource


class MapSync(_Body):
    """Sparse map delta (or full sync) plus the entity lists. Frames are never sent; this is the shared picture."""

    type: Literal[MessageType.MAP_SYNC] = MessageType.MAP_SYNC
    grid: GridSpec
    seq: int = Field(ge=0)
    full: bool = False
    cells: list[GridCellUpdate] = Field(default_factory=list)
    hotspots: list[Hotspot] = Field(default_factory=list)
    water_sources: list[WaterSource] = Field(default_factory=list)


class TaskAnnounce(_Body):
    """Leader offers a task for bidding (auction). Bids must arrive before bid_deadline."""

    type: Literal[MessageType.TASK_ANNOUNCE] = MessageType.TASK_ANNOUNCE
    task: Task
    bid_deadline: float


class TaskBid(_Body):
    type: Literal[MessageType.TASK_BID] = MessageType.TASK_BID
    task_id: str
    bidder_id: str
    cost: float = Field(ge=0)  # lower is better; define the formula in core/, keep units consistent
    feasible: bool = True      # False = "I cannot do this" (battery, water, range, agent mismatch)
    eta_s: float | None = Field(default=None, ge=0)
    energy_cost_pct: float | None = Field(default=None, ge=0, le=100)


class TaskAssign(_Body):
    """Leader -> specific drone (set receiver_id). Carries the leader epoch."""

    type: Literal[MessageType.TASK_ASSIGN] = MessageType.TASK_ASSIGN
    task: Task
    epoch: int = Field(ge=0)


class TaskAck(_Body):
    type: Literal[MessageType.TASK_ACK] = MessageType.TASK_ACK
    task_id: str
    drone_id: str
    accepted: bool
    reason: str | None = None
    epoch: int = Field(ge=0)


class TaskComplete(_Body):
    type: Literal[MessageType.TASK_COMPLETE] = MessageType.TASK_COMPLETE
    task_id: str
    drone_id: str
    success: bool
    result_status: HotspotStatus | None = None  # e.g. SUPPRESSED after a VERIFY, CONFIRMED after INVESTIGATE
    water_used_l: float | None = Field(default=None, ge=0)
    notes: str | None = None


class PathClaim(_Body):
    """Reserve a corridor/time window so others deconflict. Altitude band optional."""

    type: Literal[MessageType.PATH_CLAIM] = MessageType.PATH_CLAIM
    drone_id: str
    waypoints: list[Position] = Field(min_length=1)
    valid_from: float
    valid_until: float
    altitude_band_m: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _window_ordered(self) -> "PathClaim":
        if self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be > valid_from")
        return self


class LeaderClaim(_Body):
    """Election: highest (epoch, score, candidate_id) wins. Re-sent as the leader's own heartbeat of authority."""

    type: Literal[MessageType.LEADER_CLAIM] = MessageType.LEADER_CLAIM
    candidate_id: str
    epoch: int = Field(ge=0)
    score: float  # e.g. health + endurance + link quality; tie-break on candidate_id


class RefillRequest(_Body):
    type: Literal[MessageType.REFILL_REQUEST] = MessageType.REFILL_REQUEST
    drone_id: str
    position: Position
    water_remaining_l: float = Field(ge=0)
    agent: ExtinguishingAgent = ExtinguishingAgent.WATER


class RefillGrant(_Body):
    """Leader -> specific drone (set receiver_id): where to refill, which bay, how long the queue is."""

    type: Literal[MessageType.REFILL_GRANT] = MessageType.REFILL_GRANT
    drone_id: str
    target_kind: RefillTargetKind
    target_id: str
    position: Position
    bay: int | None = Field(default=None, ge=0)
    eta_s: float | None = Field(default=None, ge=0)
    queue_wait_s: float = Field(default=0.0, ge=0)
    epoch: int = Field(default=0, ge=0)


class GroundCommand(_Body):
    """Human/ground-base command. `target_id` is a drone_id, hotspot_id or source_id depending on `command`."""

    type: Literal[MessageType.GROUND_COMMAND] = MessageType.GROUND_COMMAND
    command: CommandKind
    target_id: str | None = None
    params: dict[str, float | str | bool] = Field(default_factory=dict)
    issued_by: str = "ground"


class AlertNotice(_Body):
    type: Literal[MessageType.ALERT] = MessageType.ALERT
    alert: Alert


class SimFault(_Body):
    """SIMULATOR ONLY. Fault injection for demos and resilience tests; real drones must ignore it."""

    type: Literal[MessageType.SIM_FAULT] = MessageType.SIM_FAULT
    fault: SimFaultKind
    target_id: str | None = None
    duration_s: float | None = Field(default=None, gt=0)
    magnitude: float | None = None


Body = Annotated[
    Union[
        Heartbeat, HotspotReport, WaterSourceReport, MapSync, TaskAnnounce, TaskBid, TaskAssign,
        TaskAck, TaskComplete, PathClaim, LeaderClaim, RefillRequest, RefillGrant, GroundCommand,
        AlertNotice, SimFault,
    ],
    Field(discriminator="type"),
]

BODY_TYPES: dict[MessageType, type[_Body]] = {
    MessageType.HEARTBEAT: Heartbeat,
    MessageType.HOTSPOT_REPORT: HotspotReport,
    MessageType.WATER_SOURCE_REPORT: WaterSourceReport,
    MessageType.MAP_SYNC: MapSync,
    MessageType.TASK_ANNOUNCE: TaskAnnounce,
    MessageType.TASK_BID: TaskBid,
    MessageType.TASK_ASSIGN: TaskAssign,
    MessageType.TASK_ACK: TaskAck,
    MessageType.TASK_COMPLETE: TaskComplete,
    MessageType.PATH_CLAIM: PathClaim,
    MessageType.LEADER_CLAIM: LeaderClaim,
    MessageType.REFILL_REQUEST: RefillRequest,
    MessageType.REFILL_GRANT: RefillGrant,
    MessageType.GROUND_COMMAND: GroundCommand,
    MessageType.ALERT: AlertNotice,
    MessageType.SIM_FAULT: SimFault,
}


class QoSProfile(FrozenContract):
    """How hard the bus should try to deliver a message type."""

    reliable: bool          # True = retransmit until acked; False = best-effort, newest wins
    ttl_s: float = Field(gt=0)  # drop if older than this on arrival
    rate_hz: float | None = Field(default=None, gt=0)  # nominal publish rate for periodic messages


DEFAULT_QOS: dict[MessageType, QoSProfile] = {
    MessageType.HEARTBEAT: QoSProfile(reliable=False, ttl_s=3, rate_hz=2),
    MessageType.HOTSPOT_REPORT: QoSProfile(reliable=True, ttl_s=60),
    MessageType.WATER_SOURCE_REPORT: QoSProfile(reliable=True, ttl_s=60),
    MessageType.MAP_SYNC: QoSProfile(reliable=False, ttl_s=10, rate_hz=1),
    MessageType.TASK_ANNOUNCE: QoSProfile(reliable=True, ttl_s=10),
    MessageType.TASK_BID: QoSProfile(reliable=True, ttl_s=10),
    MessageType.TASK_ASSIGN: QoSProfile(reliable=True, ttl_s=30),
    MessageType.TASK_ACK: QoSProfile(reliable=True, ttl_s=30),
    MessageType.TASK_COMPLETE: QoSProfile(reliable=True, ttl_s=60),
    MessageType.PATH_CLAIM: QoSProfile(reliable=False, ttl_s=5, rate_hz=2),
    MessageType.LEADER_CLAIM: QoSProfile(reliable=True, ttl_s=5, rate_hz=1),
    MessageType.REFILL_REQUEST: QoSProfile(reliable=True, ttl_s=30),
    MessageType.REFILL_GRANT: QoSProfile(reliable=True, ttl_s=30),
    MessageType.GROUND_COMMAND: QoSProfile(reliable=True, ttl_s=60),
    MessageType.ALERT: QoSProfile(reliable=True, ttl_s=60),
    MessageType.SIM_FAULT: QoSProfile(reliable=True, ttl_s=60),
}


class SwarmMessage(FrozenContract):
    """Envelope for everything on the bus."""

    v: str = CONTRACT_VERSION
    msg_id: str
    seq: int = Field(default=0, ge=0)          # per-sender increasing counter
    sender_id: str
    receiver_id: str | None = None             # None = broadcast to topic subscribers
    timestamp: float                           # Clock seconds at send time
    ttl_s: float = Field(default=5.0, gt=0)
    reliable: bool
    epoch: int | None = Field(default=None, ge=0)  # leader epoch the sender believes in (commands)
    body: Body

    @property
    def msg_type(self) -> MessageType:
        return self.body.type

    def is_expired(self, now: float) -> bool:
        return (now - self.timestamp) > self.ttl_s

    @classmethod
    def create(
        cls,
        sender_id: str,
        body: _Body,
        now: float,
        *,
        seq: int = 0,
        receiver_id: str | None = None,
        ttl_s: float | None = None,
        reliable: bool | None = None,
        epoch: int | None = None,
        msg_id: str | None = None,
    ) -> "SwarmMessage":
        """Build a message with the default QoS for its type. Increment `seq` per sender per message."""
        qos = DEFAULT_QOS[body.type]
        return cls(
            msg_id=msg_id or f"{sender_id}-{seq}",
            seq=seq,
            sender_id=sender_id,
            receiver_id=receiver_id,
            timestamp=now,
            ttl_s=qos.ttl_s if ttl_s is None else ttl_s,
            reliable=qos.reliable if reliable is None else reliable,
            epoch=epoch,
            body=body,
        )


def encode(msg: SwarmMessage) -> bytes:
    """Serialize for the bus. Use at EVERY boundary, including the in-process bus."""
    return msg.model_dump_json().encode("utf-8")


def decode(raw: bytes | str) -> SwarmMessage:
    """Parse + validate. Raises pydantic.ValidationError on malformed input."""
    return SwarmMessage.model_validate_json(raw)
