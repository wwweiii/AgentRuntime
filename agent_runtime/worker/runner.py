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
        system_prompt = _build_system_prompt(agent_data)
        user_prompt = _build_user_prompt(task_data, context_text, context_metrics)
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


def _build_user_prompt(task_data: dict[str, Any], context_text: str, context_metrics: dict[str, Any]) -> str:
    return (
        f"Task ID: {task_data['task_id']}\n"
        f"Objective: {task_data['objective']}\n"
        f"Context snapshot: {task_data.get('context_id')}\n"
        f"Context metrics: {context_metrics}\n\n"
        f"Visible context:\n{context_text}\n\n"
        "Complete this AgentTask. If a debugger task is needed, include RUNTIME_DYNAMIC_TASK:debugger."
    )


def _extract_dynamic_tasks(output: str, task_data: dict[str, Any], config_data: dict[str, Any]) -> list[dict[str, Any]]:
    if not config_data.get("enable_dynamic_tasks", True):
        return []
    # Guard: if this is already a dynamic debug task, don't spawn another
    if task_data.get("agent_id") == "debugger" and task_data.get("dynamic"):
        return []
    if "RUNTIME_DYNAMIC_TASK:debugger" not in output:
        return []
    return [
        {
            "agent_id": "debugger",
            "objective": f"Fix the issue reported by task {task_data['task_id']}",
            "dependencies": [task_data["task_id"]],
            "priority": 8,
            "created_by": task_data["task_id"],
            "dynamic": True,
        }
    ]


if __name__ == "__main__":
    main()

