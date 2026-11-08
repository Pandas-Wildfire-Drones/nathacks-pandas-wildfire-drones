"""Greedy, contract-backed task bidding and assignment."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from contracts import (
    DroneKind,
    DroneState,
    DroneStatus,
    Position,
    Task,
    TaskBid,
    TaskStatus,
    TaskType,
)

__all__ = ["SSIAuctionRound", "allocate_tasks", "bid_for_task", "target_position_for"]

DEFAULT_TRAVEL_TIME_NORMALIZER_S = 600.0

_SEEKER_TASKS = {
    TaskType.SCAN_ZONE,
    TaskType.INVESTIGATE,
    TaskType.VERIFY,
    TaskType.SURVEY_WATER,
}


class SSIAuctionRound:
    """Collect at most one bid per drone for one task, then select its winner."""

    def __init__(self, task_id: str, bid_deadline: float) -> None:
        if not task_id:
            raise ValueError("task_id must not be empty")
        if not math.isfinite(bid_deadline):
            raise ValueError("bid_deadline must be finite")
        self.task_id = task_id
        self.bid_deadline = bid_deadline
        self._bids: dict[str, TaskBid] = {}
        self._closed = False
        self._winner: TaskBid | None = None

    @property
    def bids(self) -> tuple[TaskBid, ...]:
        """Return accepted bids in deterministic bidder-id order."""
        return tuple(self._bids[key] for key in sorted(self._bids))

    def submit_bid(self, bid: TaskBid, received_at: float) -> bool:
        """Accept a valid first bid from a drone before the deadline."""
        if not math.isfinite(received_at):
            raise ValueError("received_at must be finite")
        if bid.task_id != self.task_id:
            raise ValueError(f"bid is for {bid.task_id!r}, not {self.task_id!r}")
        if self._closed or received_at > self.bid_deadline or bid.bidder_id in self._bids:
            return False
        self._bids[bid.bidder_id] = bid
        return True

    def close(self, now: float) -> TaskBid | None:
        """Close the round at/after its deadline and return the lowest feasible bid."""
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if now < self.bid_deadline:
            raise ValueError("cannot close an auction round before its bid deadline")
        if not self._closed:
            feasible = [bid for bid in self._bids.values() if bid.feasible]
            self._winner = min(feasible, key=lambda bid: (bid.cost, bid.bidder_id), default=None)
            self._closed = True
        return self._winner


def target_position_for(task: Task, positions: Mapping[str, Position]) -> Position:
    """Resolve a task's target from the shared target-id-to-position map."""
    if task.type in (TaskType.INVESTIGATE, TaskType.EXTINGUISH, TaskType.VERIFY):
        target_id = task.target_hotspot_id
    elif task.type is TaskType.SCAN_ZONE:
        target_id = task.target_zone_id
    elif task.type is TaskType.SURVEY_WATER:
        target_id = task.target_water_source_id or task.target_zone_id
    elif task.type is TaskType.REFILL:
        target_id = task.target_station_id or task.target_water_source_id
    else:
        raise ValueError(f"unsupported task type: {task.type}")

    if target_id is None:
        raise ValueError(f"{task.type} task {task.task_id!r} has no target id")
    try:
        return positions[target_id]
    except KeyError as exc:
        raise KeyError(f"no target position supplied for {target_id!r}") from exc


def _kind_can_do(task: Task, drone: DroneState) -> bool:
    if task.type in _SEEKER_TASKS:
        return drone.kind is DroneKind.SEEKER
    if task.type is TaskType.EXTINGUISH:
        return drone.kind is DroneKind.EXTINGUISHER
    return True


def bid_for_task(
    task: Task,
    drone: DroneState,
    target: Position,
    *,
    now: float,
    reserve_s: float = 60.0,
    cruise_speed_mps: float = 10.0,
    return_position: Position | None = None,
    travel_time_normalizer_s: float = DEFAULT_TRAVEL_TIME_NORMALIZER_S,
) -> TaskBid:
    """Estimate feasibility and the normalized marginal-travel-time bid.

    The cruise-speed default is a planning heuristic, not a platform specification.
    Unless specified, the return leg ends at the drone's starting position.
    The normalizer makes travel cost bounded and comparable without changing its
    ordering; deployment should use the same value for every bidder.
    """
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    if not math.isfinite(reserve_s) or reserve_s < 0:
        raise ValueError("reserve_s must be finite and >= 0")
    if not math.isfinite(cruise_speed_mps) or cruise_speed_mps <= 0:
        raise ValueError("cruise_speed_mps must be finite and > 0")
    if not math.isfinite(travel_time_normalizer_s) or travel_time_normalizer_s <= 0:
        raise ValueError("travel_time_normalizer_s must be finite and > 0")

    feasible = (
        task.status is TaskStatus.PENDING
        and drone.status is DroneStatus.IDLE
        and drone.current_task_id is None
        and _kind_can_do(task, drone)
        and task.priority > 0.0
        and (
            task.cluster_id is None
            or (
                drone.cluster_id == task.cluster_id
                and drone.cluster_epoch == task.cluster_epoch
            )
        )
    )
    eta_s = drone.position.distance_to(target) / cruise_speed_mps
    return_target = drone.position if return_position is None else return_position
    return_s = target.distance_to(return_target) / cruise_speed_mps
    required_s = eta_s + return_s + reserve_s

    if drone.endurance_remaining_s is None or drone.endurance_remaining_s < required_s:
        feasible = False
    if task.deadline is not None and now + eta_s > task.deadline:
        feasible = False
    if task.type is TaskType.EXTINGUISH:
        requested_water = task.volume_l
        if drone.water_remaining_l is None or drone.water_remaining_l <= 0:
            feasible = False
        elif requested_water is not None and drone.water_remaining_l < requested_water:
            feasible = False

    normalized_travel_time = eta_s / (eta_s + travel_time_normalizer_s)
    cost = normalized_travel_time / task.priority if feasible else 0.0
    return TaskBid(
        task_id=task.task_id,
        bidder_id=drone.drone_id,
        cost=cost,
        feasible=feasible,
        eta_s=eta_s if feasible else None,
        energy_cost_pct=(
            min(100.0, 100.0 * required_s / drone.endurance_remaining_s)
            if feasible and drone.endurance_remaining_s
            else None
        ),
        cluster_id=task.cluster_id,
        cluster_epoch=task.cluster_epoch,
    )


def allocate_tasks(
    tasks: Iterable[Task],
    drones: Iterable[DroneState],
    target_positions: Mapping[str, Position],
    *,
    now: float,
    epoch: int,
    reserve_s: float = 60.0,
    cruise_speed_mps: float = 10.0,
    return_position: Position | None = None,
    travel_time_normalizer_s: float = DEFAULT_TRAVEL_TIME_NORMALIZER_S,
    cluster_id: str | None = None,
    cluster_epoch: int = 0,
) -> list[Task]:
    """Run one sequential single-item auction for each pending task.

    Tasks are auctioned from highest priority downward. Drones without sufficient
    endurance for the outbound leg, return leg, and reserve do not bid. Each round
    accepts one bid per drone; each drone wins at most one task per call. Returned
    tasks are updated copies; inputs remain unchanged.
    """
    if epoch < 0:
        raise ValueError("epoch must be >= 0")
    if cluster_epoch < 0:
        raise ValueError("cluster_epoch must be >= 0")
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    if not math.isfinite(reserve_s) or reserve_s < 0:
        raise ValueError("reserve_s must be finite and >= 0")
    if not math.isfinite(cruise_speed_mps) or cruise_speed_mps <= 0:
        raise ValueError("cruise_speed_mps must be finite and > 0")
    if not math.isfinite(travel_time_normalizer_s) or travel_time_normalizer_s <= 0:
        raise ValueError("travel_time_normalizer_s must be finite and > 0")

    available_drones = [
        drone for drone in drones
        if drone.status is DroneStatus.IDLE and drone.current_task_id is None
        and (
            cluster_id is None
            or (drone.cluster_id == cluster_id and drone.cluster_epoch == cluster_epoch)
        )
    ]
    pending_tasks = [
        task for task in tasks
        if task.status is TaskStatus.PENDING
        and (cluster_id is None or (
            task.cluster_id == cluster_id and task.cluster_epoch == cluster_epoch
        ))
    ]
    pending_tasks.sort(key=lambda task: (-task.priority, task.created_at, task.task_id))

    assigned: list[Task] = []
    used_drone_ids: set[str] = set()
    for task in pending_tasks:
        target = target_position_for(task, target_positions)
        round_ = SSIAuctionRound(task.task_id, bid_deadline=now)
        for drone in available_drones:
            if drone.drone_id in used_drone_ids:
                continue
            bid = bid_for_task(
                task,
                drone,
                target,
                now=now,
                reserve_s=reserve_s,
                cruise_speed_mps=cruise_speed_mps,
                return_position=return_position,
                travel_time_normalizer_s=travel_time_normalizer_s,
            )
            if bid.feasible:
                round_.submit_bid(bid, received_at=now)

        winner = round_.close(now)
        if winner is None:
            continue
        task_copy = task.model_copy(deep=True)
        task_copy.status = TaskStatus.ASSIGNED
        task_copy.assigned_to = winner.bidder_id
        task_copy.epoch = epoch
        if cluster_id is not None:
            task_copy.cluster_id = cluster_id
            task_copy.cluster_epoch = cluster_epoch
        assigned.append(task_copy)
        used_drone_ids.add(winner.bidder_id)

    return assigned