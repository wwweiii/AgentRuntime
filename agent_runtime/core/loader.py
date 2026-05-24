from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_runtime.core.models import AgentSpec, AgentTask, FailurePolicy, ResourceRequest, RuntimeConfig


def load_scenario(path: Path) -> tuple[RuntimeConfig, list[AgentSpec], list[AgentTask], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = _runtime_config(data.get("runtime", {}))
    agents = [_agent_spec(item, config) for item in data.get("agents", [])]
    tasks = [_agent_task(item) for item in data.get("tasks", [])]
    root = data.get("root_context", {})
    return config, agents, tasks, root


def _runtime_config(data: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        model=data.get("model", "qwen3.5:9b"),
        ollama_url=data.get("ollama_url", "http://localhost:11434"),
        max_workers=int(data.get("max_workers", 2)),
        max_parallel_model_calls=int(data.get("max_parallel_model_calls", 1)),
        memory_budget_mb=int(data.get("memory_budget_mb", 4096)),
        token_budget_per_task=int(data.get("token_budget_per_task", 12000)),
        context_compression_threshold=int(data.get("context_compression_threshold", 5000)),
        run_root=data.get("run_root", "runs"),
        mock_model=bool(data.get("mock_model", False)),
        enable_dynamic_tasks=bool(data.get("enable_dynamic_tasks", True)),
        max_dynamic_depth=int(data.get("max_dynamic_depth", 2)),
        embedding_model=data.get("embedding_model", "qwen3-embedding:4b"),
        semantic_threshold=float(data.get("semantic_threshold", 0.90)),
    )


def _resource_request(data: dict[str, Any] | None) -> ResourceRequest:
    data = data or {}
    return ResourceRequest(
        cpu_slots=int(data.get("cpu_slots", data.get("cpu", 1))),
        memory_mb=int(data.get("memory_mb", 512)),
        model_tokens=int(data.get("model_tokens", data.get("tokens", 4096))),
        model_calls=int(data.get("model_calls", 1)),
    )


def _failure_policy(data: dict[str, Any] | None) -> FailurePolicy:
    data = data or {}
    return FailurePolicy(
        retry=int(data.get("retry", 1)),
        retry_backoff_ms=int(data.get("retry_backoff_ms", 500)),
        timeout_s=int(data.get("timeout_s", 120)),
        fallback_agent=data.get("fallback_agent"),
        cascade=bool(data.get("cascade", False)),
    )


def _agent_spec(data: dict[str, Any], config: RuntimeConfig) -> AgentSpec:
    return AgentSpec(
        agent_id=data["agent_id"],
        role=data.get("role", data["agent_id"]),
        system_prompt=data.get("system_prompt", ""),
        model=data.get("model", config.model),
        tools=list(data.get("tools", [])),
        resource_profile=_resource_request(data.get("resource_profile")),
        private_context=bool(data.get("private_context", False)),
    )


def _agent_task(data: dict[str, Any]) -> AgentTask:
    return AgentTask(
        task_id=data["task_id"],
        agent_id=data["agent_id"],
        objective=data["objective"],
        dependencies=list(data.get("dependencies", [])),
        priority=int(data.get("priority", 5)),
        resource_request=_resource_request(data.get("resource_request")),
        failure_policy=_failure_policy(data.get("failure_policy")),
        created_by=data.get("created_by", "user"),
        dynamic=bool(data.get("dynamic", False)),
        metadata=dict(data.get("metadata", {})),
    )

