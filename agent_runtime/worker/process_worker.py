from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_runtime.core.models import AgentSpec, AgentTask, RuntimeConfig, WorkerResult


@dataclass(slots=True)
class RunningWorker:
    task: AgentTask
    process: subprocess.Popen
    started_at: float
    work_dir: Path
    result_path: Path
    log_path: Path


class ProcessWorkerManager:
    """Run every AgentTask in a child process and exchange data through files.

    File-based result passing avoids Windows named-pipe restrictions in some
    sandboxed environments while preserving process-level fault isolation.
    """

    def __init__(self, run_dir: Path, config: RuntimeConfig) -> None:
        self.run_dir = run_dir
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]

    def start(
        self,
        task: AgentTask,
        agent: AgentSpec,
        context_text: str,
        context_metrics: dict[str, Any],
        pending_messages: list[dict[str, Any]] | None = None,
    ) -> RunningWorker:
        work_dir = (self.run_dir / "workers" / task.task_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = (work_dir / "input.json").resolve()
        result_path = (work_dir / "result.json").resolve()
        log_path = work_dir / "worker.log"
        payload = {
            "task": asdict(task),
            "agent": asdict(agent),
            "context_text": context_text,
            "context_metrics": context_metrics,
            "config": asdict(self.config),
            "work_dir": str(work_dir),
            "pending_messages": pending_messages or [],
        }
        input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(self.project_root) + os.pathsep + env.get("PYTHONPATH", "")
        log_file = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "agent_runtime.worker.runner", str(input_path), str(result_path)],
            cwd=str(work_dir),
            stdout=log_file,
            stderr=log_file,
            env=env,
            text=True,
        )
        log_file.close()
        return RunningWorker(
            task=task,
            process=process,
            started_at=time.time(),
            work_dir=work_dir,
            result_path=result_path,
            log_path=log_path,
        )

    def collect(self, worker: RunningWorker) -> WorkerResult | None:
        if worker.result_path.exists():
            payload = json.loads(worker.result_path.read_text(encoding="utf-8"))
            return WorkerResult(**payload)
        return None

    def terminate(self, worker: RunningWorker) -> None:
        if worker.process.poll() is None:
            worker.process.terminate()
            try:
                worker.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=3)
