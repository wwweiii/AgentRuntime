from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProcessUsage:
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    alive: bool = False


class OSAdapter:
    name = "base"

    def get_usage(self, pid: int) -> ProcessUsage:
        return ProcessUsage(alive=False)

    def describe_isolation(self) -> dict[str, str]:
        return {
            "process": "subprocess/multiprocessing worker",
            "filesystem": "per-run working directory",
            "resource_limit": "runtime quota accounting",
        }

