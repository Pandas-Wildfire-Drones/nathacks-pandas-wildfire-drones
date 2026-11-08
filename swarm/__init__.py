"""Leader coordination and tasking for the wildfire drone swarm."""

from .communication import SwarmCommunicator
from .failure_handling import FailureMonitor, FailureReport
from .leader_coordination import LeaderCoordinator
from .leader_election import elect_leader, leader_score, select_leader_claim
from .prioritization import prioritize_hotspots, score_hotspot
from .refill_assignment import RefillAssignmentCoordinator, RefillDecision
from .task_auction import AuctionOutcome, TaskAuctionCoordinator, TaskBidResponder
from .task_allocation import SSIAuctionRound, allocate_tasks, bid_for_task, target_position_for
from .task_creation import create_tasks_from_hotspots
from .task_lifecycle import TaskLifecycleCoordinator

__all__ = [
    "AuctionOutcome",
    "FailureMonitor",
    "FailureReport",
    "LeaderCoordinator",
    "RefillAssignmentCoordinator",
    "RefillDecision",
    "SSIAuctionRound",
    "SwarmCommunicator",
    "TaskAuctionCoordinator",
    "TaskBidResponder",
    "TaskLifecycleCoordinator",
    "allocate_tasks",
    "bid_for_task",
    "create_tasks_from_hotspots",
    "elect_leader",
    "leader_score",
    "prioritize_hotspots",
    "score_hotspot",
    "select_leader_claim",
    "target_position_for",
]
