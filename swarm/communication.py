"""Small typed facade over the shared Bus and message contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from contracts import Body, Bus, Clock, MessageType, Subscription, SwarmMessage, topic_for

__all__ = ["SwarmCommunicator"]

SwarmHandler = Callable[[SwarmMessage], None]


class SwarmCommunicator:
    """Build routed envelopes and maintain per-sender message sequence numbers."""

    def __init__(
        self,
        bus: Bus,
        clock: Clock,
        sender_id: str,
        *,
        initial_seq: int = 0,
    ) -> None:
        if initial_seq < 0:
            raise ValueError("initial_seq must be >= 0")
        self._bus = bus
        self._clock = clock
        self._sender_id = sender_id
        self._next_seq = initial_seq

    def publish(
        self,
        body: Body,
        *,
        receiver_id: str | None = None,
        epoch: int | None = None,
    ) -> SwarmMessage:
        """Publish one message; transport/serialization errors propagate to the caller."""
        seq = self._next_seq
        self._next_seq += 1
        message = SwarmMessage.create(
            sender_id=self._sender_id,
            body=body,
            now=self._clock.now(),
            seq=seq,
            receiver_id=receiver_id,
            epoch=epoch,
        )
        self._bus.publish(topic_for(message), message)
        return message

    def subscribe(
        self,
        pattern: str,
        handler: SwarmHandler,
        *,
        message_types: Iterable[MessageType] | None = None,
    ) -> Subscription:
        """Subscribe to a topic, optionally restricting delivery by message type."""
        allowed_types = None if message_types is None else frozenset(message_types)

        def dispatch(message: SwarmMessage) -> None:
            if allowed_types is None or message.msg_type in allowed_types:
                handler(message)

        return self._bus.subscribe(pattern, dispatch)