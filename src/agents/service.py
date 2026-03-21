from __future__ import annotations

from dataclasses import dataclass

from src.shared.interfaces.llm import ChatMessage, GenerationConfig, LLM, MessageRole
from src.shared.interfaces.tool import ToolContext
from src.tools import ToolRegistry


@dataclass(slots=True)
class AgentResult:
    answer: str
    steps: int
    tools_used: list[str]
    status: str


class AgentService:
    def __init__(
        self,
        *,
        llm: LLM,
        registry: ToolRegistry,
        max_steps: int = 6,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout_s: float | None = 30.0,
        system_prompt: str = (
            "You are an agentic assistant. "
            "Use available tools when needed, then provide a final answer."
        ),
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._generation_config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        self._system_prompt = system_prompt

    async def run(self, *, question: str, session_id: str | None = None,
                  user_id: str | None = None) -> AgentResult:
        messages: list[ChatMessage] = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=self._system_prompt,
            ),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        tools_used: list[str] = []

        for step in range(1, self._max_steps + 1):
            response = await self._llm.generate(
                messages=messages,
                config=self._generation_config,
                tools=self._registry.list_openai_tools(),
            )

            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            if not response.tool_calls:
                return AgentResult(
                    answer=response.content,
                    steps=step,
                    tools_used=tools_used,
                    status="ok",
                )

            for tool_call in response.tool_calls:
                tool_result = await self._registry.execute_tool_call(
                    tool_call,
                    context=ToolContext(session_id=session_id, user_id=user_id),
                )
                tools_used.append(tool_call.name)

                payload = (
                    tool_result.output
                    if tool_result.success
                    else {"error": tool_result.error or "Tool execution failed."}
                )
                messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                        content=str(payload),
                    )
                )

        return AgentResult(
            answer="Agent stopped before final answer (max steps reached).",
            steps=self._max_steps,
            tools_used=tools_used,
            status="max_steps_reached",
        )
