"""In-process, deterministic pub/sub bus (contracts.interfaces.Bus).

Hard rule 2: every message is encoded on publish and decoded separately for EVERY subscriber, so drones never
share Python objects. Delivery happens in `pump(now)` after a fixed latency, in publish order (deterministic).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from contracts import SwarmMessage, decode, encode, topic_matches, validate_pattern

__all__ = ["SimBus", "SimClock"]


class SimClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def now(self) -> float:
        return self.t


@dataclass
class _Sub:
    pattern: str
    handler: Callable[[SwarmMessage], object]
    active: bool = True

    def cancel(self) -> None:
        self.active = False


@dataclass
class SimBus:
    clock: SimClock
    latency_s: float = 0.05
    _queue: deque = field(default_factory=deque)
    _subs: list[_Sub] = field(default_factory=list)
    taps: list[Callable[[str, SwarmMessage], None]] = field(default_factory=list)  # read-only observers (UI log)
    delivered: int = 0

    def publish(self, topic: str, msg: SwarmMessage) -> None:
        self._queue.append((self.clock.now() + self.latency_s, topic, encode(msg)))

    def subscribe(self, pattern: str, handler: Callable[[SwarmMessage], object]) -> _Sub:
        validate_pattern(pattern)
        sub = _Sub(pattern, handler)
        self._subs.append(sub)
        return sub

    def pump(self, now: float, max_rounds: int = 4) -> None:
        """Deliver every message due by `now`. Messages published by handlers are delivered on a later pump."""
        n = len(self._queue)
        for _ in range(n):
            due, topic, raw = self._queue[0]
            if due > now:
                break
            self._queue.popleft()
            for tap in self.taps:
                tap(topic, decode(raw))
            for sub in list(self._subs):
                if sub.active and topic_matches(sub.pattern, topic):
                    msg = decode(raw)
                    if msg.is_expired(now):
                        continue
                    sub.handler(msg)
                    self.delivered += 1
