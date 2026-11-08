"""Leader scoring and deterministic failover election helpers."""

from __future__ import annotations

from collections.abc import Iterable

from contracts import DroneState, DroneStatus, LeaderClaim

__all__ = ["elect_leader", "leader_score", "select_leader_claim"]


def leader_score(state: DroneState) -> float:
    """Score a viable candidate from normalized health, link, battery, and endurance."""
    if state.status in (DroneStatus.FAULT, DroneStatus.LOST):
        return 0.0

    endurance = state.endurance_remaining_s
    endurance_score = 0.5 if endurance is None else endurance / (endurance + 600.0)
    link_score = 0.5 if state.link_quality is None else state.link_quality
    sensor_values = tuple(state.sensors_ok.values())
    health_score = sum(sensor_values) / len(sensor_values) if sensor_values else 0.5
    score = (
        0.30 * (state.battery_pct / 100.0)
        + 0.30 * endurance_score
        + 0.20 * link_score
        + 0.20 * health_score
    )
    return min(1.0, max(0.0, score))


def elect_leader(
    candidates: Iterable[DroneState],
    *,
    epoch: int | None = None,
) -> LeaderClaim:
    """Elect the highest-scoring viable drone and return its claim at a newer epoch."""
    all_states = list(candidates)
    states = [
        state for state in all_states
        if state.status not in (DroneStatus.FAULT, DroneStatus.LOST)
    ]
    if not states:
        raise ValueError("cannot elect a leader without a viable candidate")

    current_epoch = max(state.epoch for state in all_states)
    election_epoch = current_epoch + 1 if epoch is None else epoch
    if election_epoch <= current_epoch:
        raise ValueError("election epoch must be greater than every candidate's current epoch")

    winner = max(states, key=lambda state: (leader_score(state), state.drone_id))
    return LeaderClaim(
        candidate_id=winner.drone_id,
        epoch=election_epoch,
        score=leader_score(winner),
    )


def select_leader_claim(claims: Iterable[LeaderClaim]) -> LeaderClaim:
    """Resolve competing claims using the contract's (epoch, score, candidate_id) order."""
    claim_list = list(claims)
    if not claim_list:
        raise ValueError("cannot select a leader from an empty claim set")
    return max(claim_list, key=lambda claim: (claim.epoch, claim.score, claim.candidate_id))