"""Create prioritized mission tasks from the swarm's current hotspot picture."""

from __future__ import annotations

import math
from collections.abc import Iterable

from contracts import Hotspot, HotspotStatus, Task, TaskStatus, TaskType

from .prioritization import prioritize_hotspots

__all__ = ["create_tasks_from_hotspots"]

_TASK_FOR_STATUS = {
    HotspotStatus.DETECTED: TaskType.INVESTIGATE,
    HotspotStatus.CONFIRMED: TaskType.EXTINGUISH,
}
_ACTIVE_TASK_STATUSES = {
    TaskStatus.PENDING,
    TaskStatus.ASSIGNED,
    TaskStatus.IN_PROGRESS,
}


def create_tasks_from_hotspots(
    hotspots: Iterable[Hotspot],
    now: float,
    *,
    existing_tasks: Iterable[Task] = (),
    deadline_after_s: float | None = None,
) -> list[Task]:
    """Create missing investigation/extinguish tasks from the latest hotspot state.

    DETECTED hotspots create INVESTIGATE tasks and CONFIRMED hotspots create
    EXTINGUISH tasks. Other statuses do not create work: in particular, ASSIGNED
    prevents duplicate assignment and terminal statuses are not actionable.

    Priority is copied from the weighted hotspot urgency score. IDs are deterministic
    from task type, hotspot ID, and the hotspot's last observation time. An optional
    deadline is a duration from `now`; no deadline is imposed by default.
    """
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    if deadline_after_s is not None and (
        not math.isfinite(deadline_after_s) or deadline_after_s <= 0
    ):
        raise ValueError("deadline_after_s must be finite and > 0 when provided")
    deadline = None if deadline_after_s is None else now + deadline_after_s
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("resulting task deadline must be finite")

    active_targets = {
        (task.type, task.target_hotspot_id)
        for task in existing_tasks
        if task.status in _ACTIVE_TASK_STATUSES and task.target_hotspot_id is not None
    }

    tasks: list[Task] = []
    for hotspot in prioritize_hotspots(hotspots, now):
        task_type = _TASK_FOR_STATUS.get(hotspot.status)
        if task_type is None or (task_type, hotspot.hotspot_id) in active_targets:
            continue

        task_id = (
            f"{task_type.value.lower()}-{hotspot.hotspot_id}-"
            f"{hotspot.last_seen.hex()}"
        )
        task = Task(
            task_id=task_id,
            type=task_type,
            priority=min(1.0, hotspot.priority_score),
            target_hotspot_id=hotspot.hotspot_id,
            created_at=now,
            deadline=deadline,
        )
        tasks.append(task)
        active_targets.add((task_type, hotspot.hotspot_id))

    return tasks
