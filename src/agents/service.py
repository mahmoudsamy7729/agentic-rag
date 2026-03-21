from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.shared.interfaces.llm import ChatMessage, GenerationConfig, LLM, MessageRole
from src.shared.interfaces.tool import ToolContext
from src.tools import ToolRegistry


@dataclass(slots=True)
class AgentCitation:
    source: str
    doc_id: str
    chunk_id: str
    snippet: str


@dataclass(slots=True)
class AgentResult:
    answer: str
    steps: int
    tools_used: list[str]
    status: str
    citations: list[AgentCitation] = field(default_factory=list)


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

    async def run(
        self,
        *,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AgentResult:
        messages: list[ChatMessage] = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=self._system_prompt,
            ),
            ChatMessage(role=MessageRole.USER, content=question),
        ]
        tools_used: list[str] = []
        latest_retrieval_payload: dict[str, Any] | None = None

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
                    citations=self._extract_citations(latest_retrieval_payload),
                )

            for tool_call in response.tool_calls:
                tool_result = await self._registry.execute_tool_call(
                    tool_call,
                    context=ToolContext(session_id=session_id, user_id=user_id),
                )
                tools_used.append(tool_call.name)

                if (
                    tool_call.name == "retrieve_context"
                    and tool_result.success
                    and isinstance(tool_result.output, dict)
                ):
                    latest_retrieval_payload = tool_result.output

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
            citations=self._extract_citations(latest_retrieval_payload),
        )

    @staticmethod
    def _extract_citations(
        retrieval_payload: dict[str, Any] | None,
    ) -> list[AgentCitation]:
        if not retrieval_payload:
            return []
        raw_results = retrieval_payload.get("results", [])
        if not isinstance(raw_results, list):
            return []

        citations: list[AgentCitation] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            snippet = str(item.get("text", "")).strip()
            if not snippet:
                continue

            citations.append(
                AgentCitation(
                    source=str(item.get("source", "unknown")),
                    doc_id=str(item.get("doc_id", "")),
                    chunk_id=str(item.get("chunk_id", "")),
                    snippet=snippet[:280],
                )
            )
        return citations
