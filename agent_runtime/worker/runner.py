from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from agent_runtime.model.ollama_client import OllamaClient


def main() -> None:
    input_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = execute_agent_task(payload)
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def execute_agent_task(payload: dict[str, Any]) -> dict[str, Any]:
    work_dir = payload["work_dir"]
    os.chdir(work_dir)
    start = time.time()
    task_data = payload["task"]
    agent_data = payload["agent"]
    config_data = payload["config"]
    context_text = payload["context_text"]
    context_metrics = payload["context_metrics"]
    task_id = task_data["task_id"]
    try:
        agent_id = agent_data["agent_id"]
        pending_messages = payload.get("pending_messages", [])
        system_prompt = _build_system_prompt(agent_data)
        user_prompt = _build_user_prompt(task_data, context_text, context_metrics, pending_messages)
        client = OllamaClient(
            base_url=config_data["ollama_url"],
            model=agent_data.get("model") or config_data["model"],
            mock=config_data["mock_model"],
        )
        response = client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=agent_data.get("model") or config_data["model"],
            timeout_s=task_data.get("failure_policy", {}).get("timeout_s", 120),
        )
        output = response.content
        return {
            "task_id": task_id,
            "ok": True,
            "output": output,
            "error": None,
            "messages": [
                {
                    "from": agent_id,
                    "type": "agent_output",
                    "payload": {"task_id": task_id, "chars": len(output)},
                }
            ],
            "dynamic_tasks": _extract_dynamic_tasks(output, task_data, config_data),
            "message_directives": _extract_message_directives(output),
            "metrics": {
                "latency_s": time.time() - start,
                "model_latency_s": response.latency_s,
                "prompt_tokens_est": response.prompt_tokens_est,
                "output_tokens_est": response.output_tokens_est,
                "context_tokens_est": context_metrics.get("estimated_tokens", 0),
                "pid": os.getpid(),
            },
        }
    except Exception as exc:
        return {
            "task_id": task_id,
            "ok": False,
            "output": "",
            "error": f"{exc}\n{traceback.format_exc()}",
            "messages": [],
            "dynamic_tasks": [],
            "metrics": {"latency_s": time.time() - start, "pid": os.getpid()},
        }


def _build_system_prompt(agent_data: dict[str, Any]) -> str:
    return (
        f"You are agent {agent_data['agent_id']} in Agent Runtime. Role: {agent_data['role']}.\n"
        f"{agent_data['system_prompt']}\n"
        "Return a clear result that can be reused by other agents."
    )


def _build_user_prompt(
    task_data: dict[str, Any],
    context_text: str,
    context_metrics: dict[str, Any],
    pending_messages: list[dict[str, Any]] | None = None,
) -> str:
    parts = [
        f"Task ID: {task_data['task_id']}",
        f"Objective: {task_data['objective']}",
        f"Context snapshot: {task_data.get('context_id')}",
        f"Context metrics: {context_metrics}",
    ]

    if pending_messages:
        parts.append("\nIncoming messages from other agents:")
        for msg in pending_messages:
            sender = msg.get("from", msg.get("sender", "unknown"))
            payload = msg.get("payload", {})
            content = payload.get("content", json.dumps(payload, ensure_ascii=False))
            parts.append(f"[{sender}]: {content}")

    parts.append(f"\nVisible context:\n{context_text}")
    parts.append(
        "\nComplete this AgentTask. You may use these runtime directives:\n"
        "  RUNTIME_DYNAMIC_TASK:<agent_id>:<objective> — request a new task\n"
        "  RUNTIME_MESSAGE:<agent_id>:<content>    — send a message to another agent"
    )
    return "\n".join(parts)


def _extract_dynamic_tasks(output: str, task_data: dict[str, Any], config_data: dict[str, Any]) -> list[dict[str, Any]]:
    if not config_data.get("enable_dynamic_tasks", True):
        return []
    if task_data.get("agent_id") == "debugger" and task_data.get("dynamic"):
        return []
    tasks: list[dict[str, Any]] = []
    for line in output.split("\n"):
        line = line.strip()
        if not line.startswith("RUNTIME_DYNAMIC_TASK:"):
            continue
        # Format: RUNTIME_DYNAMIC_TASK:<agent_id>:<objective>
        rest = line[len("RUNTIME_DYNAMIC_TASK:"):]
        parts = rest.split(":", 1)
        agent_id = parts[0].strip()
        objective = parts[1].strip() if len(parts) > 1 else f"Dynamic task from {task_data['task_id']}"
        if not agent_id:
            continue
        tasks.append({
            "agent_id": agent_id,
            "objective": objective,
            "dependencies": [task_data["task_id"]],
            "priority": 8,
            "created_by": task_data["task_id"],
            "dynamic": True,
        })
    return tasks


def _extract_message_directives(output: str) -> list[dict[str, Any]]:
    directives: list[dict[str, Any]] = []
    for line in output.split("\n"):
        line = line.strip()
        if not line.startswith("RUNTIME_MESSAGE:"):
            continue
        # Format: RUNTIME_MESSAGE:<agent_id>:<content>
        rest = line[len("RUNTIME_MESSAGE:"):]
        parts = rest.split(":", 1)
        recipient = parts[0].strip()
        content = parts[1].strip() if len(parts) > 1 else ""
        if not recipient:
            continue
        directives.append({
            "recipient": recipient,
            "content": content,
            "msg_type": "agent_directive",
        })
    return directives


if __name__ == "__main__":
    main()

