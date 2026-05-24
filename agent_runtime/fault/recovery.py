from __future__ import annotations

from agent_runtime.core.models import AgentTask
from agent_runtime.core.state import TaskState


class FaultManager:
    def should_retry(self, task: AgentTask) -> bool:
        return task.attempts <= task.failure_policy.retry

    def mark_retry_or_failed(self, task: AgentTask) -> TaskState:
        if self.should_retry(task):
            task.state = TaskState.RETRYING
        else:
            task.state = TaskState.FAILED
        return task.state

