"""Connect per-drone autonomy to the shared swarm message bus.

The swarm brain owns allocation and swarm-wide decisions.

This adapter owns the boundary between those decisions and one drone's
local execution safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autonomy.drone_controller import DroneController, MissionExecutionResult
from autonomy.health_manager import evaluate_drone_health
from contracts.entities import Alert, DroneState, Hotspot, Task
from contracts.enums import AlertKind, Severity, TaskStatus, TaskType
from contracts.messages import (
    AlertNotice,
    Heartbeat,
    HotspotReport,
    PathClaim,
    RefillGrant,
    SwarmMessage,
    TaskAssign,
    TaskComplete,
)
from contracts.topics import topic_for


@dataclass
class SwarmAutonomyState:
    """Local state retained by one drone's autonomy adapter."""

    current_task: Task | None = None
    current_hotspot: Hotspot | None = None
    latest_heartbeat: Heartbeat | None = None
    latest_refill_grant: RefillGrant | None = None
    latest_path_claim: PathClaim | None = None


class SwarmAutonomyAdapter:
    """Message-driven integration layer for one autonomous drone.

    Incoming swarm decisions are never treated as unconditional permission
    to perform physical actions. They first pass through local autonomy
    validation and the DroneController safety gates.
    """

    def __init__(
        self,
        drone: DroneState,
        controller: DroneController,
        bus: Any,
    ) -> None:
        self.drone = drone
        self.controller = controller
        self.bus = bus

        # The adapter currently receives time through the DroneState contract.
        # This keeps the integration API compatible with the existing swarm
        # tests while still giving generated alerts/messages a valid timestamp.
        self.now = drone.timestamp

        self.state = SwarmAutonomyState()

        self._subscribe()

    def _subscribe(self) -> None:
        """Subscribe to topics relevant to this drone."""

        topics = (
            "swarm/task_assign",
            "swarm/heartbeat",
            "swarm/hotspot_report",
            "swarm/refill_grant",
            "swarm/path_claim",
        )

        for topic in topics:
            self.bus.subscribe(topic, self.handle_message)

    def handle_message(
        self,
        message: SwarmMessage,
    ) -> MissionExecutionResult | None:
        """Dispatch one swarm message to the appropriate local handler."""

        body = message.body

        if isinstance(body, TaskAssign):
            return self.handle_task_assign(message, body)

        if isinstance(body, Heartbeat):
            self.handle_heartbeat(message, body)
            return None

        if isinstance(body, HotspotReport):
            self.handle_hotspot_report(message, body)
            return None

        if isinstance(body, RefillGrant):
            self.handle_refill_grant(message, body)
            return None

        if isinstance(body, PathClaim):
            self.handle_path_claim(message, body)
            return None

        self._publish_alert(
            message,
            f"Unsupported swarm message body: {type(body).__name__}",
        )
        return None

    def handle_task_assign(
        self,
        message: SwarmMessage,
        assignment: TaskAssign,
    ) -> MissionExecutionResult | None:
        """Accept an assignment only when it belongs to this drone.

        The assignment epoch is checked before any navigation or physical
        action is attempted.
        """

        task = assignment.task

        # A command for another drone is simply ignored.
        if task.assigned_to != self.drone.drone_id:
            return None

        # Never execute stale swarm work.
        if assignment.epoch != self.drone.epoch:
            self._publish_alert(
                message,
                "Rejected stale task assignment before execution",
            )
            return None

        # The task itself also carries an epoch. Both must agree with
        # the drone's currently accepted leader epoch.
        if task.epoch != self.drone.epoch:
            self._publish_alert(
                message,
                "Rejected task whose epoch does not match drone epoch",
            )
            return None

        # Only assigned/in-progress work is executable.
        if task.status not in (
            TaskStatus.ASSIGNED,
            TaskStatus.IN_PROGRESS,
        ):
            self._publish_alert(
                message,
                "Rejected task because it is not executable",
            )
            return None

        self.state.current_task = task

        if task.type is not TaskType.EXTINGUISH:
            return None

        hotspot = self._find_hotspot(task.target_hotspot_id)

        # Never execute toward an unknown target.
        if hotspot is None:
            self._publish_alert(
                message,
                "Extinguish task received before target hotspot was known",
            )
            return None

        self.state.current_hotspot = hotspot

        # The leader may allocate the hotspot, but local autonomy still
        # refuses an explicit assignment to another drone.
        if (
            hotspot.assigned_to is not None
            and hotspot.assigned_to != self.drone.drone_id
        ):
            self._publish_alert(
                message,
                "Hotspot is assigned to another drone",
            )
            return None

        try:
            return self.controller.begin_extinguish(
                self.drone,
                hotspot,
                task,
            )
        except (ValueError, RuntimeError) as exc:
            self._publish_alert(message, str(exc))
            return None

    def step(self) -> MissionExecutionResult | None:
        """Advance the currently assigned local mission by one control step."""

        task = self.state.current_task
        hotspot = self.state.current_hotspot

        if task is None or hotspot is None:
            return None

        health = evaluate_drone_health(self.drone)

        # Health is a hard safety gate. It happens before mission execution.
        if not health.safe:
            if self.controller.autopilot is not None:
                self.controller.autopilot.hold()

            self._publish_alert(
                None,
                f"Autonomy blocked mission: {health.reason}",
            )
            return None

        if task.type is not TaskType.EXTINGUISH:
            return None

        result = self.controller.step_extinguish(
            self.drone,
            hotspot,
            task,
        )

        if result.status in ("DONE", "FAILED", "REJECTED"):
            self._publish_task_complete(task, result)

        return result

    def handle_heartbeat(
        self,
        message: SwarmMessage,
        heartbeat: Heartbeat,
    ) -> None:
        """Record the latest heartbeat without overriding local safety."""

        self.state.latest_heartbeat = heartbeat

    def handle_hotspot_report(
        self,
        message: SwarmMessage,
        report: HotspotReport,
    ) -> None:
        """Cache the reported hotspot."""

        self.state.current_hotspot = report.hotspot

    def handle_refill_grant(
        self,
        message: SwarmMessage,
        grant: RefillGrant,
    ) -> None:
        """Record a refill grant only for this drone and current epoch."""

        if grant.drone_id != self.drone.drone_id:
            return

        if grant.epoch != self.drone.epoch:
            self._publish_alert(
                message,
                "Rejected stale refill grant",
            )
            return

        self.state.latest_refill_grant = grant

    def handle_path_claim(
        self,
        message: SwarmMessage,
        claim: PathClaim,
    ) -> None:
        """Record path information relevant to local deconfliction."""

        if claim.drone_id == self.drone.drone_id:
            self.state.latest_path_claim = claim

    def _find_hotspot(self, hotspot_id: str | None) -> Hotspot | None:
        if hotspot_id is None:
            return None

        current = self.state.current_hotspot

        if current is not None and current.hotspot_id == hotspot_id:
            return current

        return None

    def _publish_task_complete(
        self,
        task: Task,
        result: MissionExecutionResult,
    ) -> None:
        body = TaskComplete(
            task_id=task.task_id,
            drone_id=self.drone.drone_id,
            success=result.success,
            result_status=None,
            water_used_l=result.water_used_l,
            notes=result.reason,
        )

        swarm_message = SwarmMessage.create(
            sender_id=self.drone.drone_id,
            body=body,
            now=self.now,
            epoch=self.drone.epoch,
        )

        self.bus.publish(
            topic_for(swarm_message),
            swarm_message,
        )

    def _publish_alert(
        self,
        message: SwarmMessage | None,
        reason: str,
    ) -> None:
        """Publish a valid human-facing autonomy safety alert."""

        alert = Alert(
            alert_id=(
                f"{self.drone.drone_id}-"
                f"autonomy-{int(self.now * 1000)}"
            ),
            severity=Severity.WARNING,
            kind=AlertKind.GENERIC,
            subject_id=self.drone.drone_id,
            message=reason,
            position=self.drone.position,
            timestamp=self.now,
        )

        body = AlertNotice(alert=alert)

        swarm_message = SwarmMessage.create(
            sender_id=self.drone.drone_id,
            body=body,
            now=self.now,
            epoch=self.drone.epoch,
        )

        self.bus.publish(
            topic_for(swarm_message),
            swarm_message,
        )
