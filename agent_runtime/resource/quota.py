from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.core.models import AgentTask, RuntimeConfig


@dataclass(slots=True)
class ResourceSnapshot:
    running_workers: int
    model_calls: int
    memory_mb: int
    cpu_slots: int


class ResourceManager:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.running_workers = 0
        self.model_calls = 0
        self.memory_mb = 0
        self.cpu_slots = 0

    def can_admit(self, task: AgentTask) -> bool:
        req = task.resource_request
        if self.running_workers + 1 > self.config.max_workers:
            return False
        if self.model_calls + req.model_calls > self.config.max_parallel_model_calls:
            return False
        if self.memory_mb + req.memory_mb > self.config.memory_budget_mb:
            return False
        return True

    def admit(self, task: AgentTask) -> None:
        req = task.resource_request
        self.running_workers += 1
        self.model_calls += req.model_calls
        self.memory_mb += req.memory_mb
        self.cpu_slots += req.cpu_slots

    def release(self, task: AgentTask) -> None:
        req = task.resource_request
        self.running_workers = max(0, self.running_workers - 1)
        self.model_calls = max(0, self.model_calls - req.model_calls)
        self.memory_mb = max(0, self.memory_mb - req.memory_mb)
        self.cpu_slots = max(0, self.cpu_slots - req.cpu_slots)

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            running_workers=self.running_workers,
            model_calls=self.model_calls,
            memory_mb=self.memory_mb,
            cpu_slots=self.cpu_slots,
        )

