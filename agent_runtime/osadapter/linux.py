from __future__ import annotations

from agent_runtime.osadapter.base import OSAdapter, ProcessUsage


class LinuxAdapter(OSAdapter):
    name = "linux"

    def get_usage(self, pid: int) -> ProcessUsage:
        try:
            import psutil  # type: ignore

            proc = psutil.Process(pid)
            mem = proc.memory_info().rss / (1024 * 1024)
            return ProcessUsage(cpu_percent=proc.cpu_percent(interval=0.0), memory_mb=mem, alive=proc.is_running())
        except Exception:
            return ProcessUsage(alive=True)

    def describe_isolation(self) -> dict[str, str]:
        return {
            "process": "multiprocessing.Process per AgentTask",
            "filesystem": "per-task working directory; mount namespace can be added",
            "resource_limit": "cgroups v2 adapter placeholder",
            "security": "seccomp/namespace hooks reserved for openEuler/openKylin",
        }

