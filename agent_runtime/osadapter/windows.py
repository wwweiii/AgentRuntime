from __future__ import annotations

import os

from agent_runtime.osadapter.base import OSAdapter, ProcessUsage


class WindowsAdapter(OSAdapter):
    name = "windows"

    def get_usage(self, pid: int) -> ProcessUsage:
        try:
            import psutil  # type: ignore

            proc = psutil.Process(pid)
            mem = proc.memory_info().rss / (1024 * 1024)
            return ProcessUsage(cpu_percent=proc.cpu_percent(interval=0.0), memory_mb=mem, alive=proc.is_running())
        except Exception:
            return ProcessUsage(alive=self._pid_exists(pid))

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def describe_isolation(self) -> dict[str, str]:
        return {
            "process": "multiprocessing.Process per AgentTask",
            "filesystem": "isolated run/task working directory",
            "resource_limit": "timeout + runtime quotas; psutil monitoring when installed",
            "linux_extension": "cgroups/namespace/seccomp implemented by LinuxAdapter",
        }

