from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_runtime.core.state import TaskState


@dataclass(slots=True)
class ResourceRequest:
    cpu_slots: int = 1
    memory_mb: int = 512
    model_tokens: int = 4096
    model_calls: int = 1


@dataclass(slots=True)
class FailurePolicy:
    retry: int = 1
    retry_backoff_ms: int = 500
    timeout_s: int = 120
    fallback_agent: str | None = None
    cascade: bool = False


@dataclass(slots=True)
class AgentSpec:
    agent_id: str
    role: str
    system_prompt: str
    model: str = "qwen3.5:4b"
    tools: list[str] = field(default_factory=list)
    resource_profile: ResourceRequest = field(default_factory=ResourceRequest)
    private_context: bool = False


@dataclass(slots=True)
class AgentTask:
    task_id: str
    agent_id: str
    objective: str
    dependencies: list[str] = field(default_factory=list)
    priority: int = 5
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    failure_policy: FailurePolicy = field(default_factory=FailurePolicy)
    context_id: str | None = None
    state: TaskState = TaskState.CREATED
    attempts: int = 0
    created_by: str = "user"
    dynamic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeConfig:
    model: str = "qwen3.5:4b"
    ollama_url: str = "http://localhost:11434"
    max_workers: int = 2
    max_parallel_model_calls: int = 1
    memory_budget_mb: int = 4096
    token_budget_per_task: int = 12000
    context_compression_threshold: int = 5000
    run_root: str = "runs"
    mock_model: bool = False
    enable_dynamic_tasks: bool = True
    max_dynamic_depth: int = 2
    max_fallback_depth: int = 3
    failure_rate_threshold: float = 0.5
    embedding_model: str | None = "qwen3-embedding:4b"
    semantic_threshold: float = 0.90
    disable_context_reuse: bool = False
    random_scheduling: bool = False


@dataclass(slots=True)
class WorkerResult:
    task_id: str
    ok: bool
    output: str
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    dynamic_tasks: list[dict[str, Any]] = field(default_factory=list)
    message_directives: list[dict[str, Any]] = field(default_factory=list)

