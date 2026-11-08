"""Connect per-drone autonomy to the shared swarm message bus.

The swarm brain owns allocation and swarm-wide decisions.

This adapter owns the boundary between those decisions and one drone's
local execution safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autonomy.deconfliction import Conflict, detect_conflicts
from autonomy.drone_controller import DroneController, MissionExecutionResult
from autonomy.health_manager import evaluate_drone_health
from contracts.entities import Alert, DroneState, Hotspot, Task
from contracts.geometry import Position
from contracts.enums import AlertKind, MessageType, Severity, TaskStatus, TaskType
from autonomy.path_claim_manager import PathClaimManager
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
from contracts.topics import ROUTES, topic_for


@dataclass
class SwarmAutonomyState:
    """Local state retained by one autonomous drone."""

    current_task: Task | None = None
    current_hotspot: Hotspot | None = None
    latest_heartbeat: Heartbeat | None = None
    latest_refill_grant: RefillGrant | None = None
    latest_path_claim: PathClaim | None = None

    # Path claims belonging to other drones are retained separately from
    # this drone's own claim. Local autonomy uses these claims only for
    # deconfliction; they do not grant permission to modify another drone.
    remote_path_claims: list[PathClaim] = field(default_factory=list)


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

        # The adapter receives time through the DroneState contract.
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
            ROUTES[MessageType.PATH_CLAIM],
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

        # Health is the highest local safety gate. A lower-level mission
        # objective cannot override an unsafe drone state.
        if not health.safe:
            if self.controller.autopilot is not None:
                if health.action == "RETURN":
                    self.controller.autopilot.return_to_launch()
                else:
                    self.controller.autopilot.hold()

            self._publish_alert(
                None,
                f"Autonomy blocked mission: {health.reason}",
            )
            return None

        if task.type is not TaskType.EXTINGUISH:
            return None

        # DroneState.position.z is the local altitude reference used by
        # the existing contract. The controller's drop limits are applied
        # immediately before release.
        altitude_agl_m = max(0.0, float(self.drone.position.z))

        result = self.controller.step_extinguish(
            self.drone,
            hotspot,
            task,
            altitude_agl_m=altitude_agl_m,
        )

        if result.status in ("DONE", "FAILED", "REJECTED"):
            self._publish_task_complete(task, result)

        return result

    def check_path_conflicts(
        self,
        proposed_path: list,
        minimum_separation_m: float,
    ) -> list[Conflict]:
        """Check a proposed local path against active remote path claims.

        Path claims are advisory coordination data. They never override
        local health, mission, or resource safety.
        """

        other_paths = [
            claim.waypoints
            for claim in self.state.remote_path_claims
            if self._claim_is_current(claim)
        ]

        return detect_conflicts(
            proposed_path,
            other_paths,
            minimum_separation_m,
        )

    def publish_planned_path(
        self,
        path_claim_manager: PathClaimManager,
        waypoints: list[Position],
        *,
        valid_from: float,
        time_step_s: float,
        altitude_band_m: tuple[float, float] | None,
        minimum_separation_m: float,
    ) -> SwarmMessage | None:
        """Publish a planned route only when active claims permit it."""

        conflicts = self.check_path_conflicts(
            waypoints,
            minimum_separation_m,
        )

        if conflicts:
            self._publish_alert(
                None,
                (
                    "Rejected planned path because it conflicts "
                    "with an active remote path claim"
                ),
            )
            return None

        return path_claim_manager.publish_from_planned_path(
            self.drone,
            waypoints,
            valid_from=valid_from,
            time_step_s=time_step_s,
            altitude_band_m=altitude_band_m,
        )

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
        """Record a path claim for local deconfliction."""

        if claim.drone_id == self.drone.drone_id:
            self.state.latest_path_claim = claim
            return

        # Replace a previous claim from the same drone so the local
        # deconfliction view contains only its latest known route.
        self.state.remote_path_claims = [
            existing
            for existing in self.state.remote_path_claims
            if existing.drone_id != claim.drone_id
        ]
        self.state.remote_path_claims.append(claim)

    def _claim_is_current(self, claim: PathClaim) -> bool:
        """Return whether a remote claim overlaps the adapter's current time."""

        return claim.valid_from <= self.now <= claim.valid_until

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