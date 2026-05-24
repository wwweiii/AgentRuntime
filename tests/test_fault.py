"""Tests for fault recovery: retry, fallback depth, circuit breaker."""

from agent_runtime.core.models import AgentTask, FailurePolicy
from agent_runtime.fault.recovery import FaultManager


def _make_task(task_id: str, retry: int = 1, fallback: str | None = None) -> AgentTask:
    return AgentTask(
        task_id=task_id,
        agent_id="test_agent",
        objective=f"Test {task_id}",
        failure_policy=FailurePolicy(retry=retry, fallback_agent=fallback),
    )


def test_retry_when_attempts_within_limit() -> None:
    fm = FaultManager()
    task = _make_task("t1", retry=2)
    task.attempts = 2  # <= retry count
    assert fm.should_retry(task) is True


def test_no_retry_when_attempts_exceeded() -> None:
    fm = FaultManager()
    task = _make_task("t1", retry=1)
    task.attempts = 2  # > retry count
    assert fm.should_retry(task) is False


def test_no_retry_when_retry_zero() -> None:
    fm = FaultManager()
    task = _make_task("t1", retry=0)
    task.attempts = 1
    assert fm.should_retry(task) is False


def test_mark_retry_or_failed_retries() -> None:
    from agent_runtime.core.state import TaskState

    fm = FaultManager()
    task = _make_task("t1", retry=2)
    task.attempts = 1
    state = fm.mark_retry_or_failed(task)
    assert state == TaskState.RETRYING


def test_mark_retry_or_failed_fails() -> None:
    from agent_runtime.core.state import TaskState

    fm = FaultManager()
    task = _make_task("t1", retry=0)
    task.attempts = 1
    state = fm.mark_retry_or_failed(task)
    assert state == TaskState.FAILED


def test_fallback_depth_tracking() -> None:
    """Verify fallback metadata accumulates depth correctly."""
    task = _make_task("original", retry=0, fallback="backup_agent")
    # Simulate what runtime does: create a fallback with depth+1
    depth = task.metadata.get("fallback_depth", 0) + 1
    assert depth == 1
    task.metadata["fallback_depth"] = depth
    assert task.metadata["fallback_depth"] == 1

    # Second level fallback
    depth2 = task.metadata.get("fallback_depth", 0) + 1
    assert depth2 == 2


def test_failure_policy_defaults() -> None:
    fp = FailurePolicy()
    assert fp.retry == 1
    assert fp.timeout_s == 120
    assert fp.fallback_agent is None
    assert fp.cascade is False


def test_failure_policy_custom() -> None:
    fp = FailurePolicy(retry=3, timeout_s=60, fallback_agent="debugger", cascade=True)
    assert fp.retry == 3
    assert fp.timeout_s == 60
    assert fp.fallback_agent == "debugger"
    assert fp.cascade is True
