from __future__ import annotations

from agent_runtime.tool.spec import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def tool_descriptions(self) -> str:
        lines: list[str] = []
        for tool in self._tools.values():
            params_desc = tool.description
            if tool.parameters:
                params_desc += f" Parameters: {tool.parameters}"
            lines.append(f"  {tool.name}: {params_desc}")
        return "\n".join(lines)
