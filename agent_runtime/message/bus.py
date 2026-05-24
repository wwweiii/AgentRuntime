from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Message:
    msg_id: str
    sender: str
    recipient: str | None
    msg_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    context_ref: str | None = None
    topic: str | None = None
    created_at: float = field(default_factory=time.time)


class MessageBus:
    def __init__(self) -> None:
        self._seq = 0
        self.mailboxes: dict[str, deque[Message]] = defaultdict(deque)
        self.topics: dict[str, list[Message]] = defaultdict(list)

    def send(
        self,
        sender: str,
        recipient: str,
        msg_type: str,
        payload: dict[str, Any] | None = None,
        context_ref: str | None = None,
    ) -> Message:
        self._seq += 1
        msg = Message(
            msg_id=f"msg-{self._seq:06d}",
            sender=sender,
            recipient=recipient,
            msg_type=msg_type,
            payload=payload or {},
            context_ref=context_ref,
        )
        self.mailboxes[recipient].append(msg)
        return msg

    def receive(self, agent_id: str, limit: int = 10) -> list[Message]:
        messages: list[Message] = []
        queue = self.mailboxes[agent_id]
        while queue and len(messages) < limit:
            messages.append(queue.popleft())
        return messages

    def publish(
        self,
        sender: str,
        topic: str,
        msg_type: str,
        payload: dict[str, Any] | None = None,
        context_ref: str | None = None,
    ) -> Message:
        self._seq += 1
        msg = Message(
            msg_id=f"msg-{self._seq:06d}",
            sender=sender,
            recipient=None,
            topic=topic,
            msg_type=msg_type,
            payload=payload or {},
            context_ref=context_ref,
        )
        self.topics[topic].append(msg)
        return msg

    def topic_messages(self, topic: str) -> list[Message]:
        return list(self.topics[topic])

