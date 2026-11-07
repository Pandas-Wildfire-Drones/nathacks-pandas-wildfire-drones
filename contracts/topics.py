"""Topic names and matching rules.

Convention (same idea works for ROS 2 namespaces, MQTT topics, or the in-process bus):
  drone/{id}/...   per-drone channels            ground/...   ground base channels
  swarm/...        shared swarm channels         sim/...      simulator-only channels

Wildcards in SUBSCRIPTION patterns:
  *  matches exactly one segment      #  matches the rest (zero or more segments), must be last
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .enums import MessageType

if TYPE_CHECKING:
    from .messages import SwarmMessage

__all__ = [
    "ROUTES", "topic_for", "topic_matches", "validate_pattern",
    "SUB_ALL_HEARTBEATS", "SUB_ALL_HOTSPOTS", "SUB_ALL_WATER_SOURCES", "SUB_ALL_SWARM",
    "SUB_GROUND_COMMANDS", "SUB_GROUND_ALERTS", "SUB_SIM_CONTROL", "sub_my_commands",
]

# Publish routes. {sender}/{receiver} are filled from the message envelope.
ROUTES: dict[MessageType, str] = {
    MessageType.HEARTBEAT: "drone/{sender}/heartbeat",
    MessageType.HOTSPOT_REPORT: "drone/{sender}/hotspots",
    MessageType.WATER_SOURCE_REPORT: "drone/{sender}/water_sources",
    MessageType.MAP_SYNC: "swarm/map_sync",
    MessageType.TASK_ANNOUNCE: "swarm/tasks/announce",
    MessageType.TASK_BID: "swarm/tasks/bids",
    MessageType.TASK_ASSIGN: "drone/{receiver}/cmd",
    MessageType.TASK_ACK: "swarm/tasks/acks",
    MessageType.TASK_COMPLETE: "swarm/tasks/complete",
    MessageType.PATH_CLAIM: "swarm/path_claims",
    MessageType.LEADER_CLAIM: "swarm/leader",
    MessageType.REFILL_REQUEST: "swarm/refill/requests",
    MessageType.REFILL_GRANT: "drone/{receiver}/cmd",
    MessageType.GROUND_COMMAND: "ground/commands",
    MessageType.ALERT: "ground/alerts",
    MessageType.SIM_FAULT: "sim/control",
}

# Common subscription patterns
SUB_ALL_HEARTBEATS = "drone/*/heartbeat"          # leader + ground base
SUB_ALL_HOTSPOTS = "drone/*/hotspots"             # leader
SUB_ALL_WATER_SOURCES = "drone/*/water_sources"   # leader
SUB_ALL_SWARM = "swarm/#"                         # every drone
SUB_GROUND_COMMANDS = "ground/commands"           # every drone (filters on target_id)
SUB_GROUND_ALERTS = "ground/alerts"               # ground base UI
SUB_SIM_CONTROL = "sim/control"                   # simulator (and sim-aware drone wrappers)


def sub_my_commands(drone_id: str) -> str:
    """Direct commands to one drone (TASK_ASSIGN, REFILL_GRANT)."""
    return f"drone/{drone_id}/cmd"


def topic_for(msg: "SwarmMessage") -> str:
    """Topic a message should be published on. Directed messages require receiver_id."""
    pattern = ROUTES[msg.msg_type]
    if "{receiver}" in pattern and not msg.receiver_id:
        raise ValueError(f"{msg.msg_type} requires receiver_id")
    return pattern.format(sender=msg.sender_id, receiver=msg.receiver_id)


def validate_pattern(pattern: str) -> None:
    segs = pattern.split("/")
    if any(s == "" for s in segs):
        raise ValueError(f"empty segment in pattern {pattern!r}")
    if "#" in segs and segs.index("#") != len(segs) - 1:
        raise ValueError("'#' must be the last segment")


def topic_matches(pattern: str, topic: str) -> bool:
    """MQTT-like matching. Reuse this in every Bus implementation."""
    p = pattern.split("/")
    t = topic.split("/")
    for i, seg in enumerate(p):
        if seg == "#":
            return i == len(p) - 1
        if i >= len(t):
            return False
        if seg != "*" and seg != t[i]:
            return False
    return len(p) == len(t)
