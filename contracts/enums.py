"""Enumerations shared by every workstream.

Values are uppercase strings equal to the member name so JSON stays readable.
"""
from enum import StrEnum

__all__ = [
    "DroneKind", "SwarmRole", "DroneStatus",
    "TaskType", "TaskStatus", "HotspotStatus",
    "WaterSourceKind", "WaterSourceStatus", "WaterSourceOrigin", "RefillTargetKind",
    "ExtinguishingAgent", "DetectorKind", "Powertrain",
    "MessageType", "Severity", "AlertKind", "CommandKind", "SimFaultKind",
]


class DroneKind(StrEnum):
    SEEKER = "SEEKER"              # light electric scout: 640x512 thermal + lidar, searches/confirms, surveys water
    EXTINGUISHER = "EXTINGUISHER"  # ultra-heavy hybrid water hauler: drops on seeker-confirmed hotspots, does not search


class Powertrain(StrEnum):
    ELECTRIC = "ELECTRIC"  # battery only (seekers): telemetry is battery_pct
    HYBRID = "HYBRID"      # generator + battery buffer (extinguishers): telemetry adds fuel and generator state


class SwarmRole(StrEnum):
    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"  # election in progress
    LEADER = "LEADER"        # runs the SwarmCoordinator role; any drone can hold it


class DroneStatus(StrEnum):
    IDLE = "IDLE"
    TRANSIT = "TRANSIT"
    SCANNING = "SCANNING"
    EXTINGUISHING = "EXTINGUISHING"
    RETURNING = "RETURNING"
    REFILLING = "REFILLING"
    FAULT = "FAULT"  # self-reported
    LOST = "LOST"    # assigned by OBSERVERS (leader/ground) after missed heartbeats


class TaskType(StrEnum):
    SCAN_ZONE = "SCAN_ZONE"
    INVESTIGATE = "INVESTIGATE"      # close-range confirm of a candidate hotspot
    EXTINGUISH = "EXTINGUISH"
    VERIFY = "VERIFY"                # re-image after a drop; sets SUPPRESSED if cold
    REFILL = "REFILL"
    SURVEY_WATER = "SURVEY_WATER"    # validate a candidate water source (or search a zone for one)


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class HotspotStatus(StrEnum):
    DETECTED = "DETECTED"            # seen once, unconfirmed
    CONFIRMED = "CONFIRMED"          # confirmed by close-range investigation
    ASSIGNED = "ASSIGNED"            # an extinguisher is tasked
    SUPPRESSED = "SUPPRESSED"        # drone VERIFY pass found it cold
    VERIFIED_OUT = "VERIFIED_OUT"    # GROUND-CONFIRMED out (humans declare fires out; see GroundCommand.CONFIRM_OUT)
    FALSE_POSITIVE = "FALSE_POSITIVE"


class WaterSourceKind(StrEnum):
    LAKE = "LAKE"
    POND = "POND"
    RIVER = "RIVER"
    RESERVOIR = "RESERVOIR"
    TANK = "TANK"
    OTHER = "OTHER"


class WaterSourceStatus(StrEnum):
    CANDIDATE = "CANDIDATE"  # from prior map or detection, not yet validated by a seeker
    VALIDATED = "VALIDATED"  # approach clearance + presence of water confirmed
    REJECTED = "REJECTED"    # dry / blocked / unsafe / disabled by ground


class WaterSourceOrigin(StrEnum):
    PRIOR_MAP = "PRIOR_MAP"  # imported hydrography data
    DETECTED = "DETECTED"    # found by a seeker's sensors
    MANUAL = "MANUAL"        # entered by ground personnel


class RefillTargetKind(StrEnum):
    STATION = "STATION"
    WATER_SOURCE = "WATER_SOURCE"


class ExtinguishingAgent(StrEnum):
    WATER = "WATER"
    CLASS_A_FOAM = "CLASS_A_FOAM"
    GEL = "GEL"  # water enhancer / gel
    # RETARDANT intentionally omitted: long-term retardant is for fireline building, not spot-fire knockdown.


class DetectorKind(StrEnum):
    THRESHOLD = "THRESHOLD"  # classical: temperature threshold + connected components
    ML = "ML"                # ONNX model
    FUSED = "FUSED"          # mixed approach: classical proposals verified by ML (or vice versa)
    FAKE = "FAKE"            # simulator ground truth + tunable misses/false alarms


class MessageType(StrEnum):
    HEARTBEAT = "HEARTBEAT"
    HOTSPOT_REPORT = "HOTSPOT_REPORT"
    WATER_SOURCE_REPORT = "WATER_SOURCE_REPORT"
    MAP_SYNC = "MAP_SYNC"
    CLUSTER_UPDATE = "CLUSTER_UPDATE"
    TASK_ANNOUNCE = "TASK_ANNOUNCE"
    TASK_BID = "TASK_BID"
    TASK_ASSIGN = "TASK_ASSIGN"
    TASK_ACK = "TASK_ACK"
    TASK_COMPLETE = "TASK_COMPLETE"
    PATH_CLAIM = "PATH_CLAIM"
    LEADER_CLAIM = "LEADER_CLAIM"
    REFILL_REQUEST = "REFILL_REQUEST"
    REFILL_GRANT = "REFILL_GRANT"
    GROUND_COMMAND = "GROUND_COMMAND"
    ALERT = "ALERT"
    SIM_FAULT = "SIM_FAULT"  # simulator-only fault injection ("chaos buttons")


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertKind(StrEnum):
    DRONE_LOST = "DRONE_LOST"
    LEADER_LOST = "LEADER_LOST"
    LEADER_ELECTED = "LEADER_ELECTED"
    LOW_BATTERY = "LOW_BATTERY"
    LOW_FUEL = "LOW_FUEL"                    # hybrid drones: fuel at/below reserve
    POWERPLANT_FAULT = "POWERPLANT_FAULT"    # generator/engine fault; drone is on battery buffer only
    SENSOR_FAULT = "SENSOR_FAULT"
    COMMS_DEGRADED = "COMMS_DEGRADED"
    HOTSPOT_NEW = "HOTSPOT_NEW"
    HOTSPOT_SUPPRESSED = "HOTSPOT_SUPPRESSED"
    WATER_SOURCE_NEW = "WATER_SOURCE_NEW"
    GENERIC = "GENERIC"


class CommandKind(StrEnum):
    RETURN_TO_BASE = "RETURN_TO_BASE"
    HOLD = "HOLD"
    RESUME = "RESUME"
    ABORT_TASK = "ABORT_TASK"
    SET_PRIORITY = "SET_PRIORITY"                # params: {"priority": 0..1}; target_id = hotspot_id
    CONFIRM_OUT = "CONFIRM_OUT"                  # human declares hotspot out -> VERIFIED_OUT
    MARK_FALSE_POSITIVE = "MARK_FALSE_POSITIVE"  # human rejects a hotspot
    ENABLE_WATER_SOURCE = "ENABLE_WATER_SOURCE"
    DISABLE_WATER_SOURCE = "DISABLE_WATER_SOURCE"


class SimFaultKind(StrEnum):
    KILL_DRONE = "KILL_DRONE"
    KILL_LEADER = "KILL_LEADER"
    DROP_LINK = "DROP_LINK"                # temporarily cut a drone's radio
    DEGRADE_SENSOR = "DEGRADE_SENSOR"      # magnitude 0..1; target sensor in `target_id` as "drone:sensor"
    DRAIN_BATTERY = "DRAIN_BATTERY"
    INJECT_FALSE_POSITIVE = "INJECT_FALSE_POSITIVE"
    CLEAR = "CLEAR"                        # clear all injected faults
