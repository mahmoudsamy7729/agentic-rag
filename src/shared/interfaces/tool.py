from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolContext:
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionResult:
    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Provider-agnostic tool contract for agent tool-calling."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short tool description shown to the LLM."""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON schema for tool arguments."""

    @abstractmethod
    async def run(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolContext | None = None,
    ) -> ToolExecutionResult:
        """Execute tool logic with validated arguments."""

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
