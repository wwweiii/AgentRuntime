"""Tests for message bus: pub/sub, mailbox, peek."""

from agent_runtime.message.bus import MessageBus


def test_send_and_receive() -> None:
    bus = MessageBus()
    bus.send("agent_a", "agent_b", "task_result", payload={"result": "ok"})
    msgs = bus.receive("agent_b")
    assert len(msgs) == 1
    assert msgs[0].sender == "agent_a"
    assert msgs[0].recipient == "agent_b"
    assert msgs[0].payload == {"result": "ok"}


def test_receive_clears_mailbox() -> None:
    bus = MessageBus()
    bus.send("a", "b", "test", payload={})
    assert len(bus.receive("b")) == 1
    assert len(bus.receive("b")) == 0  # Already consumed


def test_peek_does_not_consume() -> None:
    bus = MessageBus()
    bus.send("a", "b", "test", payload={})
    peeked = bus.peek("b")
    assert len(peeked) == 1
    # Peek again — still there
    peeked2 = bus.peek("b")
    assert len(peeked2) == 1
    # Receive consumes
    assert len(bus.receive("b")) == 1
    assert len(bus.peek("b")) == 0


def test_clear_mailbox() -> None:
    bus = MessageBus()
    bus.send("a", "b", "t1", payload={})
    bus.send("a", "b", "t2", payload={})
    assert len(bus.peek("b")) == 2
    bus.clear_mailbox("b")
    assert len(bus.peek("b")) == 0


def test_publish_and_topic_messages() -> None:
    bus = MessageBus()
    bus.publish("agent_a", "agent.outputs", "output", payload={"data": 42})
    bus.publish("agent_b", "agent.outputs", "output", payload={"data": 99})

    msgs = bus.topic_messages("agent.outputs")
    assert len(msgs) == 2
    assert msgs[0].sender == "agent_a"
    assert msgs[1].sender == "agent_b"


def test_context_ref_in_message() -> None:
    bus = MessageBus()
    bus.send("a", "b", "test", payload={}, context_ref="ctx-000001")
    msgs = bus.receive("b")
    assert msgs[0].context_ref == "ctx-000001"


def test_receive_limit() -> None:
    bus = MessageBus()
    for i in range(20):
        bus.send("a", "b", f"msg_{i}", payload={})
    msgs = bus.receive("b", limit=5)
    assert len(msgs) == 5
    # Remaining 15 still in mailbox
    assert len(bus.peek("b", limit=50)) == 15


def test_empty_mailbox_for_unknown_agent() -> None:
    bus = MessageBus()
    assert bus.receive("unknown") == []
    assert bus.peek("unknown") == []


def test_message_id_sequence() -> None:
    bus = MessageBus()
    m1 = bus.send("a", "b", "t1", payload={})
    m2 = bus.send("a", "b", "t2", payload={})
    assert m1.msg_id == "msg-000001"
    assert m2.msg_id == "msg-000002"
