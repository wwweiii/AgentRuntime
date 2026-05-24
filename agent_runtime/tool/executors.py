from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from agent_runtime.tool.registry import ToolRegistry
from agent_runtime.tool.spec import ToolResult, ToolSpec


def _shell_executor(cmd: str, work_dir: str = ".", timeout: int = 30) -> ToolResult:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
        )
        output = proc.stdout.strip() or proc.stderr.strip() or "(no output)"
        return ToolResult(
            tool_name="shell_cmd",
            ok=proc.returncode == 0,
            output=output,
            latency_s=time.time() - start,
            metadata={"exit_code": proc.returncode, "cmd": cmd},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="shell_cmd",
            ok=False,
            output="",
            error=f"Command timed out after {timeout}s",
            latency_s=time.time() - start,
            metadata={"cmd": cmd},
        )
    except Exception as exc:
        return ToolResult(
            tool_name="shell_cmd",
            ok=False,
            output="",
            error=str(exc),
            latency_s=time.time() - start,
            metadata={"cmd": cmd},
        )


def _read_file_executor(path: str) -> ToolResult:
    start = time.time()
    try:
        content = Path(path).read_text(encoding="utf-8")
        return ToolResult(
            tool_name="read_file",
            ok=True,
            output=content[:5000],
            latency_s=time.time() - start,
            metadata={"path": path, "size": len(content), "truncated": len(content) > 5000},
        )
    except Exception as exc:
        return ToolResult(
            tool_name="read_file",
            ok=False,
            output="",
            error=str(exc),
            latency_s=time.time() - start,
            metadata={"path": path},
        )


def _write_file_executor(path: str, content: str) -> ToolResult:
    start = time.time()
    try:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_name="write_file",
            ok=True,
            output=f"Written {len(content)} bytes to {path}",
            latency_s=time.time() - start,
            metadata={"path": path, "size": len(content)},
        )
    except Exception as exc:
        return ToolResult(
            tool_name="write_file",
            ok=False,
            output="",
            error=str(exc),
            latency_s=time.time() - start,
            metadata={"path": path},
        )


def register_builtin_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(ToolSpec(
        name="shell_cmd",
        description="Execute a shell command. Args: {\"cmd\": \"<command>\"}",
        parameters={"cmd": "string (required)", "timeout": "int (default 30s)"},
        execute=lambda **kw: _shell_executor(**kw),
    ))
    registry.register(ToolSpec(
        name="read_file",
        description="Read a file. Args: {\"path\": \"<file_path>\"}",
        parameters={"path": "string (required)"},
        execute=lambda **kw: _read_file_executor(**kw),
    ))
    registry.register(ToolSpec(
        name="write_file",
        description="Write content to a file. Args: {\"path\": \"<file_path>\", \"content\": \"<text>\"}",
        parameters={"path": "string (required)", "content": "string (required)"},
        execute=lambda **kw: _write_file_executor(**kw),
    ))
    return registry
