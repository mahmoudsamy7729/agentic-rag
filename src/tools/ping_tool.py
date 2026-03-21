from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.shared.interfaces.tool import Tool, ToolContext, ToolExecutionResult


class PingTool(Tool):
    @property
    def name(self) -> str:
        return "ping"

    @property
    def description(self) -> str:
        return "Check that the tool system is alive and return pong."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Optional message echoed back in the response.",
                }
            },
            "additionalProperties": False,
        }

    async def run(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolContext | None = None,
    ) -> ToolExecutionResult:
        message = arguments.get("message") if arguments else None
        return ToolExecutionResult(
            success=True,
            output={
                "reply": "pong",
                "echo": message,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "session_id": context.session_id if context else None,
                "user_id": context.user_id if context else None,
            },
        )
