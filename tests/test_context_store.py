from agent_runtime.context.store import ContextStore


def test_context_dedupe_and_materialize() -> None:
    store = ContextStore()
    a = store.create_segment("goal", "hello world", owner="user")
    b = store.create_segment("goal", "hello world", owner="user")
    ctx = store.create_snapshot([a, b])
    text, metrics = store.materialize(ctx, "agent", token_budget=100)

    assert a == b
    assert "hello world" in text
    assert metrics["segments"] == 2
    assert store.metrics()["dedupe_hits"] == 1

