"""Tests for DAG scheduler: dependency resolution, priority, resource admission."""

from agent_runtime.context.store import ContextStore
from agent_runtime.core.models import AgentTask, ResourceRequest, RuntimeConfig
from agent_runtime.core.state import TaskState
from agent_runtime.resource.quota import ResourceManager
from agent_runtime.scheduler.dag_scheduler import ContextAwareDAGScheduler


def _make_task(task_id: str, deps: list[str] | None = None, priority: int = 5, memory_mb: int = 512) -> AgentTask:
    return AgentTask(
        task_id=task_id,
        agent_id="test_agent",
        objective=f"Objective for {task_id}",
        dependencies=deps or [],
        priority=priority,
        resource_request=ResourceRequest(memory_mb=memory_mb),
    )


def test_dependency_order() -> None:
    store = ContextStore()
    rm = ResourceManager(RuntimeConfig(max_workers=10, max_parallel_model_calls=10))
    scheduler = ContextAwareDAGScheduler(store, rm)

    tasks = {
        "a": _make_task("a"),
        "b": _make_task("b", deps=["a"]),
        "c": _make_task("c", deps=["a", "b"]),
    }
    # Only "a" has no deps — it's ready immediately
    ready = scheduler.ready_tasks(tasks)
    assert len(ready) == 1
    assert ready[0].task_id == "a"

    # Complete task "a" — now "b" is ready
    tasks["a"].state = TaskState.COMPLETED
    ready = scheduler.ready_tasks(tasks)
    assert len(ready) == 1
    assert ready[0].task_id == "b"

    # Complete task "b" — now "c" is ready
    tasks["b"].state = TaskState.COMPLETED
    ready = scheduler.ready_tasks(tasks)
    assert len(ready) == 1
    assert ready[0].task_id == "c"


def test_priority_ordering() -> None:
    store = ContextStore()
    rm = ResourceManager(RuntimeConfig(max_workers=10, max_parallel_model_calls=10))
    scheduler = ContextAwareDAGScheduler(store, rm)

    tasks = {
        "low": _make_task("low", priority=3),
        "mid": _make_task("mid", priority=5),
        "high": _make_task("high", priority=9),
    }
    ready = scheduler.ready_tasks(tasks)
    assert ready[0].task_id == "high"
    assert ready[1].task_id == "mid"
    assert ready[2].task_id == "low"


def test_retrying_tasks_are_ready() -> None:
    store = ContextStore()
    rm = ResourceManager(RuntimeConfig(max_workers=10, max_parallel_model_calls=10))
    scheduler = ContextAwareDAGScheduler(store, rm)

    task = _make_task("retry_me")
    task.state = TaskState.RETRYING
    tasks = {"retry_me": task}
    ready = scheduler.ready_tasks(tasks)
    assert len(ready) == 1
    assert ready[0].task_id == "retry_me"


def test_resource_admission_and_rejection() -> None:
    store = ContextStore()
    rm = ResourceManager(RuntimeConfig(max_workers=1, max_parallel_model_calls=1))
    scheduler = ContextAwareDAGScheduler(store, rm)

    tasks = {
        "first": _make_task("first", priority=10),
        "second": _make_task("second", priority=5),
    }
    # select() admits only 1 due to max_workers=1
    selected = scheduler.select(tasks)
    assert len(selected) == 1
    assert selected[0].task_id == "first"

    # ready_tasks still returns both (it only checks dependencies, not resources)
    ready = scheduler.ready_tasks(tasks)
    assert len(ready) == 2

    # Release first and mark it completed — second should now be selected
    rm.release(tasks["first"])
    tasks["first"].state = TaskState.COMPLETED
    selected2 = scheduler.select(tasks)
    assert len(selected2) == 1
    assert selected2[0].task_id == "second"


def test_random_scheduling() -> None:
    store = ContextStore()
    rm = ResourceManager(RuntimeConfig(max_workers=10, max_parallel_model_calls=10))
    scheduler = ContextAwareDAGScheduler(store, rm, random_scheduling=True)

    tasks = {f"t{i}": _make_task(f"t{i}", priority=5) for i in range(20)}
    # With random scheduling, all should still be ready (just shuffled)
    ready = scheduler.ready_tasks(tasks)
    assert len(ready) == 20
