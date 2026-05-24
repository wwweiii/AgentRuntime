from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    execute: Callable[..., ToolResult] | None = None


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    ok: bool
    output: str
    error: str | None = None
    latency_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
