"""Simulated ground-base operator for the perception review queue.

Uses the REAL ground-base queue (groundbase.review_queue.GroundReviewQueue) with decision delivery on: every
ReviewReport a seeker sends after an inconclusive close-range look lands in the queue; after `delay_s` (a human
looking at the thermal + RGB evidence) the operator resolves it, approving `approve_frac` as FIRE and rejecting the
rest as NOT_FIRE. Resolving publishes GroundCommand(CONFIRM_FIRE | MARK_FALSE_POSITIVE) back to the swarm leader.

The operator does not look at ground truth: the split is a fixed, seeded coin (team spec: 70 % approve, 30 %
reject), so runs stay deterministic. The swarm never waits on it: the leader confirms autonomously when no
decision arrives in time (core.leader, review_timeout_s), and the whole operator can be switched off
(GameConfig.ground_link = False) to show the swarm working without a ground link.
"""
from __future__ import annotations

import random
from collections import Counter

from groundbase.review_queue import GroundReviewQueue, ReviewDisposition

__all__ = ["SimGroundOperator"]


class SimGroundOperator:
    def __init__(self, bus, clock, approve_frac: float = 0.7, delay_s: float = 6.0, seed: int = 0):
        self.queue = GroundReviewQueue(bus, clock.now, publish_decisions=True)
        self.approve_frac, self.delay_s = approve_frac, delay_s
        self.rng = random.Random(seed)
        self.stats: Counter = Counter()

    def tick(self, now: float) -> None:
        for rec in self.queue.pending():          # insertion order: deterministic
            if now - rec.received_at < self.delay_s:
                continue
            disp = ReviewDisposition.FIRE if self.rng.random() < self.approve_frac else ReviewDisposition.NOT_FIRE
            self.queue.resolve(rec.report.review_id, disp)
            self.stats[disp.value.lower()] += 1
