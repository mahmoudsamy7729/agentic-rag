from __future__ import annotations

from typing import Iterable

from src.shared.interfaces.llm import ToolCall
from src.shared.interfaces.tool import Tool, ToolContext, ToolExecutionResult


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool '{name}' is not registered.") from exc

    def list_openai_tools(self) -> list[dict]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

    async def execute(
        self,
        tool_name: str,
        arguments: dict,
        *,
        context: ToolContext | None = None,
    ) -> ToolExecutionResult:
        tool = self.get(tool_name)
        return await tool.run(arguments, context=context)

    async def execute_tool_call(
        self,
        tool_call: ToolCall,
        *,
        context: ToolContext | None = None,
    ) -> ToolExecutionResult:
        return await self.execute(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            context=context,
        )
