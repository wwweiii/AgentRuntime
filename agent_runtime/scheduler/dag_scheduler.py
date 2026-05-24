from __future__ import annotations

from agent_runtime.context.store import ContextStore
from agent_runtime.core.models import AgentTask
from agent_runtime.core.state import TaskState
from agent_runtime.resource.quota import ResourceManager


class ContextAwareDAGScheduler:
    """DAG scheduler with priority, resources, and context reuse preference."""

    def __init__(self, context_store: ContextStore, resource_manager: ResourceManager) -> None:
        self.context_store = context_store
        self.resource_manager = resource_manager

    def ready_tasks(self, tasks: dict[str, AgentTask]) -> list[AgentTask]:
        completed = {task_id for task_id, task in tasks.items() if task.state == TaskState.COMPLETED}
        ready: list[AgentTask] = []
        for task in tasks.values():
            if task.state in {TaskState.CREATED, TaskState.RETRYING} and all(dep in completed for dep in task.dependencies):
                ready.append(task)
        return sorted(ready, key=self._score, reverse=True)

    def select(self, tasks: dict[str, AgentTask]) -> list[AgentTask]:
        admitted: list[AgentTask] = []
        for task in self.ready_tasks(tasks):
            if self.resource_manager.can_admit(task):
                admitted.append(task)
                self.resource_manager.admit(task)
        return admitted

    def _score(self, task: AgentTask) -> float:
        score = float(task.priority) * 10.0
        score -= len(task.dependencies)
        if task.context_id and task.context_id in self.context_store.snapshots:
            snapshot = self.context_store.snapshots[task.context_id]
            reused = sum(max(0, self.context_store.ref_counts.get(seg_id, 0) - 1) for seg_id in snapshot.segment_ids)
            score += min(20.0, reused)
        score -= task.resource_request.memory_mb / 2048.0
        score -= task.attempts * 5.0
        return score

