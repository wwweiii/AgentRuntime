from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_runtime.context.store import ContextStore
from agent_runtime.core.models import AgentSpec, AgentTask, FailurePolicy, RuntimeConfig, WorkerResult
from agent_runtime.core.state import TERMINAL_STATES, TaskState
from agent_runtime.fault.recovery import FaultManager
from agent_runtime.message.bus import MessageBus
from agent_runtime.model.embedding_client import EmbeddingClient
from agent_runtime.observability.event_log import EventLog, Metrics
from agent_runtime.osadapter.factory import get_os_adapter
from agent_runtime.resource.quota import ResourceManager
from agent_runtime.scheduler.dag_scheduler import ContextAwareDAGScheduler
from agent_runtime.worker.process_worker import ProcessWorkerManager, RunningWorker


class AgentRuntime:
    def __init__(
        self,
        config: RuntimeConfig,
        agents: list[AgentSpec],
        tasks: list[AgentTask],
        root_context: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.run_dir = Path(config.run_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.event_log = EventLog(self.run_dir)
        self.metrics = Metrics()

        embedding_client = None
        if config.embedding_model and not config.mock_model:
            try:
                embedding_client = EmbeddingClient(base_url=config.ollama_url, model=config.embedding_model)
            except Exception:
                embedding_client = None

        if embedding_client:
            self.event_log.emit("embedding.client_created", model=config.embedding_model, threshold=config.semantic_threshold)

        self.context_store = ContextStore(
            compression_threshold_tokens=config.context_compression_threshold,
            embedding_client=embedding_client,
            semantic_threshold=config.semantic_threshold,
        )
        self.message_bus = MessageBus()
        self.resource_manager = ResourceManager(config)
        self.scheduler = ContextAwareDAGScheduler(self.context_store, self.resource_manager)
        self.worker_manager = ProcessWorkerManager(self.run_dir, config)
        self.fault_manager = FaultManager()
        self.os_adapter = get_os_adapter()

        self.agents = {agent.agent_id: agent for agent in agents}
        self.tasks = {task.task_id: task for task in tasks}
        self.running: dict[str, RunningWorker] = {}
        self.task_output_segments: dict[str, list[str]] = {}

        self.root_context_id = self._init_root_context(root_context or {})
        for task in self.tasks.values():
            task.context_id = self.root_context_id

        self.event_log.emit(
            "runtime.created",
            run_id=self.run_id,
            os_adapter=self.os_adapter.name,
            isolation=self.os_adapter.describe_isolation(),
            agents=list(self.agents),
            tasks=list(self.tasks),
        )

    def run(self) -> dict[str, Any]:
        self.event_log.emit("runtime.started", run_id=self.run_id)
        while not self._all_terminal():
            progress = False
            progress = self._collect_workers() or progress
            progress = self._schedule_ready_tasks() or progress
            if not progress:
                if not self.running and self._has_unsatisfied_tasks():
                    self._fail_blocked_tasks()
                    break
                time.sleep(0.1)

        self._finalize()
        self.event_log.emit("runtime.completed", run_id=self.run_id, summary=self.summary())
        return self.summary()

    def summary(self) -> dict[str, Any]:
        states = {state.value: 0 for state in TaskState}
        for task in self.tasks.values():
            states[task.state.value] = states.get(task.state.value, 0) + 1
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "states": states,
            "context": self.context_store.metrics(),
            "metrics": self.metrics.snapshot(),
            "tasks": {task_id: task.state.value for task_id, task in self.tasks.items()},
        }

    def _init_root_context(self, root_context: dict[str, Any]) -> str:
        segment_ids: list[str] = []
        goal = root_context.get("goal", "No goal provided.")
        segment_ids.append(self.context_store.create_segment("user_goal", goal, owner="user", visibility="public"))
        for idx, item in enumerate(root_context.get("segments", []), start=1):
            segment_ids.append(
                self.context_store.create_segment(
                    item.get("kind", f"root-{idx}"),
                    item.get("content", ""),
                    owner=item.get("owner", "user"),
                    visibility=item.get("visibility", "shared"),
                    allowed_agents=list(item.get("allowed_agents", [])),
                )
            )
        return self.context_store.create_snapshot(segment_ids, owner="runtime")

    def _all_terminal(self) -> bool:
        return all(task.state in TERMINAL_STATES for task in self.tasks.values()) and not self.running

    def _has_unsatisfied_tasks(self) -> bool:
        return any(task.state not in TERMINAL_STATES for task in self.tasks.values())

    def _fail_blocked_tasks(self) -> None:
        completed = {task_id for task_id, task in self.tasks.items() if task.state == TaskState.COMPLETED}
        for task in self.tasks.values():
            if task.state not in TERMINAL_STATES and not all(dep in completed for dep in task.dependencies):
                task.state = TaskState.FAILED
                self.event_log.emit(
                    "task.failed",
                    task_id=task.task_id,
                    reason="dependency_not_satisfied",
                    dependencies=task.dependencies,
                )

    def _schedule_ready_tasks(self) -> bool:
        self._prepare_contexts_for_ready_tasks()
        selected = self.scheduler.select(self.tasks)
        for task in selected:
            self._start_task(task)
        return bool(selected)

    def _prepare_contexts_for_ready_tasks(self) -> None:
        completed = {task_id for task_id, task in self.tasks.items() if task.state == TaskState.COMPLETED}
        for task in self.tasks.values():
            if task.state not in {TaskState.CREATED, TaskState.RETRYING}:
                continue
            if not all(dep in completed for dep in task.dependencies):
                continue
            segment_ids = list(self.context_store.snapshots[self.root_context_id].segment_ids)
            for dep in task.dependencies:
                dep_segments = self.task_output_segments.get(dep)
                if dep_segments:
                    segment_ids.extend(dep_segments)
            task.context_id = self.context_store.create_snapshot(segment_ids, owner=task.agent_id)

    def _start_task(self, task: AgentTask) -> None:
        agent = self.agents.get(task.agent_id)
        if not agent:
            task.state = TaskState.FAILED
            self.resource_manager.release(task)
            self.event_log.emit("task.failed", task_id=task.task_id, reason="agent_not_found", agent_id=task.agent_id)
            return

        task.state = TaskState.SCHEDULED
        task.attempts += 1
        context_text, context_metrics = self.context_store.materialize(
            task.context_id or self.root_context_id,
            task.agent_id,
            token_budget=min(self.config.token_budget_per_task, task.resource_request.model_tokens),
        )
        self.event_log.emit(
            "task.scheduled",
            task_id=task.task_id,
            agent_id=task.agent_id,
            attempt=task.attempts,
            context_id=task.context_id,
            context_metrics=context_metrics,
            resource=asdict(task.resource_request),
        )
        task.state = TaskState.RUNNING
        worker = self.worker_manager.start(task, agent, context_text, context_metrics)
        self.running[task.task_id] = worker
        self.event_log.emit("task.started", task_id=task.task_id, pid=worker.process.pid, work_dir=str(worker.work_dir))

    def _collect_workers(self) -> bool:
        progress = False
        for task_id, worker in list(self.running.items()):
            task = worker.task
            elapsed = time.time() - worker.started_at
            if elapsed > task.failure_policy.timeout_s:
                self.worker_manager.terminate(worker)
                self.running.pop(task_id, None)
                self.resource_manager.release(task)
                self.event_log.emit("task.timeout", task_id=task_id, timeout_s=task.failure_policy.timeout_s)
                self._handle_failure(task, f"timeout after {task.failure_policy.timeout_s}s")
                progress = True
                continue

            result = self.worker_manager.collect(worker)
            if result is None and worker.process.poll() is None:
                continue
            if result is None:
                exit_code = worker.process.returncode
                self.running.pop(task_id, None)
                self.resource_manager.release(task)
                self._handle_failure(task, f"worker exited without result, exit_code={exit_code}")
                progress = True
                continue

            try:
                worker.process.wait(timeout=2)
            except Exception:
                worker.process.terminate()
            self.running.pop(task_id, None)
            self.resource_manager.release(task)
            self._handle_result(task, result)
            progress = True
        return progress

    def _handle_result(self, task: AgentTask, result: WorkerResult) -> None:
        if result.ok:
            task.state = TaskState.COMPLETED
            visibility = "private" if self.agents[task.agent_id].private_context else "shared"
            segment_ids = self.context_store.create_segments_batch(
                kind_prefix=f"agent_output:{task.agent_id}",
                content=result.output,
                owner=task.agent_id,
                visibility=visibility,
            )
            self.task_output_segments[task.task_id] = segment_ids
            self.metrics.inc("tasks.completed")
            self.metrics.inc("model.prompt_tokens_est", result.metrics.get("prompt_tokens_est", 0))
            self.metrics.inc("model.output_tokens_est", result.metrics.get("output_tokens_est", 0))
            self.metrics.inc("model.context_tokens_est", result.metrics.get("context_tokens_est", 0))
            self.event_log.emit(
                "task.completed",
                task_id=task.task_id,
                agent_id=task.agent_id,
                output_segments=segment_ids,
                metrics=result.metrics,
            )
            for message in result.messages:
                self.message_bus.publish(
                    sender=message.get("from", task.agent_id),
                    topic="agent.outputs",
                    msg_type=message.get("type", "agent_output"),
                    payload=message.get("payload", {}),
                    context_ref=task.context_id,
                )
                self.event_log.emit("message.published", topic="agent.outputs", task_id=task.task_id)
            self._add_dynamic_tasks(task, result.dynamic_tasks)
        else:
            self._handle_failure(task, result.error or "unknown worker error")

    def _handle_failure(self, task: AgentTask, error: str) -> None:
        self.metrics.inc("tasks.failed_attempts")
        self.event_log.emit("task.failed_attempt", task_id=task.task_id, attempt=task.attempts, error=error)
        if self.fault_manager.should_retry(task):
            task.state = TaskState.RETRYING
            self.event_log.emit("task.retrying", task_id=task.task_id, next_attempt=task.attempts + 1)
            return

        if (
            task.failure_policy.fallback_agent
            and task.failure_policy.fallback_agent in self.agents
            and not task.dynamic
            and not task.task_id.endswith("-fallback")
        ):
            fallback_agent_id = task.failure_policy.fallback_agent
            fallback_id = f"{task.task_id}-fallback"
            fallback_policy = FailurePolicy(
                retry=task.failure_policy.retry,
                retry_backoff_ms=task.failure_policy.retry_backoff_ms,
                timeout_s=task.failure_policy.timeout_s,
                fallback_agent=None,
                cascade=False,
            )
            fallback = AgentTask(
                task_id=fallback_id,
                agent_id=fallback_agent_id,
                objective=f"Fallback recovery for failed task {task.task_id}: {task.objective}",
                dependencies=[],
                priority=task.priority + 1,
                resource_request=task.resource_request,
                failure_policy=fallback_policy,
                context_id=task.context_id,
                created_by=task.task_id,
                dynamic=True,
            )
            self.tasks[fallback_id] = fallback
            task.state = TaskState.FAILED
            self.event_log.emit("task.fallback_created", task_id=task.task_id, fallback_task_id=fallback_id)
            return

        task.state = TaskState.FAILED
        self.event_log.emit("task.failed", task_id=task.task_id, final=True)

    def _add_dynamic_tasks(self, parent: AgentTask, dynamic_tasks: list[dict[str, Any]]) -> None:
        parent_depth = parent.metadata.get("dynamic_depth", 0)
        if parent_depth >= self.config.max_dynamic_depth:
            self.event_log.emit(
                "dynamic_task.skipped",
                parent_task_id=parent.task_id,
                reason=f"max_dynamic_depth({self.config.max_dynamic_depth}) reached",
                current_depth=parent_depth,
            )
            return

        # Track which agent_ids we've already spawned from this parent
        existing_agents = {
            t.agent_id
            for t in self.tasks.values()
            if t.dynamic and t.created_by == parent.task_id
        }

        for idx, data in enumerate(dynamic_tasks, start=1):
            agent_id = data.get("agent_id")
            if agent_id not in self.agents:
                self.event_log.emit("dynamic_task.skipped", parent_task_id=parent.task_id, reason="agent_not_found", agent_id=agent_id)
                continue

            # Dedup: don't spawn the same agent type from the same parent twice
            if agent_id in existing_agents:
                self.event_log.emit("dynamic_task.skipped", parent_task_id=parent.task_id, reason="duplicate_agent", agent_id=agent_id)
                continue
            existing_agents.add(agent_id)

            task_id = f"{parent.task_id}-dyn-{idx}"
            if task_id in self.tasks:
                continue
            task = AgentTask(
                task_id=task_id,
                agent_id=agent_id,
                objective=data.get("objective", f"Dynamic task from {parent.task_id}"),
                dependencies=list(data.get("dependencies", [parent.task_id])),
                priority=int(data.get("priority", parent.priority)),
                failure_policy=parent.failure_policy,
                created_by=data.get("created_by", parent.task_id),
                dynamic=True,
                metadata={"dynamic_depth": parent_depth + 1},
            )
            self.tasks[task_id] = task
            self.event_log.emit("dynamic_task.created", parent_task_id=parent.task_id, task_id=task_id, agent_id=agent_id, depth=parent_depth + 1)

    def _finalize(self) -> None:
        for worker in list(self.running.values()):
            self.worker_manager.terminate(worker)
            self.resource_manager.release(worker.task)
        self.running.clear()
        self.context_store.export(self.run_dir / "contexts.json")
        summary = self.summary()
        (self.run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_path = Path(self.config.run_root) / "latest.json"
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.run_dir / "tasks.json").write_text(
            json.dumps({task_id: _task_record(task) for task_id, task in self.tasks.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _task_record(task: AgentTask) -> dict[str, Any]:
    data = asdict(task)
    data["state"] = task.state.value
    return data
