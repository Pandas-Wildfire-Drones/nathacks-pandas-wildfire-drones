"""Compatibility relay for the autonomy adapter's legacy swarm subscriptions."""

from __future__ import annotations

from contracts import (
    Bus,
    Clock,
    Heartbeat,
    HotspotReport,
    PathClaim,
    RefillGrant,
    SwarmMessage,
    Subscription,
    TaskAssign,
)

__all__ = ["SwarmAutonomyRelay"]


class SwarmAutonomyRelay:
    """Translate canonical task/hotspot routes to autonomy's existing topic names.

    Run exactly once per shared bus. The legacy topics are broadcast, so this
    provides compatibility rather than scalable per-cluster autonomy routing.
    Remove it when autonomy subscribes to canonical routes directly. Keep
    ``relay_telemetry`` disabled at scale because global heartbeat/path-claim
    fanout grows with the entire fleet.
    """

    def __init__(
        self,
        bus: Bus,
        clock: Clock,
        *,
        relay_telemetry: bool = False,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._relay_telemetry = relay_telemetry
        self._subscriptions: list[Subscription] = []

    def start(self) -> None:
        if self._subscriptions:
            raise RuntimeError("autonomy relay is already started")
        self._subscriptions = [
            self._bus.subscribe("drone/*/hotspots", self._on_message),
            self._bus.subscribe("drone/*/cmd", self._on_message),
        ]
        if self._relay_telemetry:
            self._subscriptions.extend([
                self._bus.subscribe("drone/*/heartbeat", self._on_message),
                self._bus.subscribe("swarm/path_claims", self._on_message),
            ])

    def _on_message(self, message: SwarmMessage) -> None:
        if message.is_expired(self._clock.now()):
            return
        body = message.body
        if isinstance(body, Heartbeat) and self._relay_telemetry:
            topic = "swarm/heartbeat"
        elif isinstance(body, HotspotReport):
            topic = "swarm/hotspot_report"
        elif isinstance(body, TaskAssign):
            topic = "swarm/task_assign"
        elif isinstance(body, RefillGrant):
            topic = "swarm/refill_grant"
        elif isinstance(body, PathClaim) and self._relay_telemetry:
            topic = "swarm/path_claim"
        else:
            return
        self._bus.publish(topic, message)

    def close(self) -> None:
        for subscription in self._subscriptions:
            subscription.cancel()
        self._subscriptions.clear()
