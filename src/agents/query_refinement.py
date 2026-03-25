from __future__ import annotations

from dataclasses import dataclass

from src.shared.interfaces.llm import ChatMessage, GenerationConfig, LLM, MessageRole

SYSTEM_PROMPT = (
    "You rewrite user questions for retrieval quality in a RAG system.\n"
    "Rules:\n"
    "- Preserve user intent exactly.\n"
    "- Return one standalone, explicit question.\n"
    "- Do not answer the question.\n"
    "- Do not invent facts or assumptions.\n"
    "- Keep key entities and constraints from the original query.\n"
    "- If the query is ambiguous, clarify wording without changing intent.\n"
    "- Output only the rewritten query.\n"
)


@dataclass(slots=True)
class QueryRefinementResult:
    refined_query: str
    used_fallback: bool


class QueryRefinementService:
    def __init__(
        self,
        *,
        llm: LLM,
        enabled: bool,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._llm = llm
        self._enabled = enabled
        self._config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def refine(self, *, question: str, doc_id: str) -> QueryRefinementResult:
        fallback = question.strip()
        if not self._enabled:
            return QueryRefinementResult(
                refined_query=fallback,
                used_fallback=True,
            )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    f"doc_id: {doc_id}\n"
                    f"question: {question}\n"
                    "Return the rewritten retrieval query only."
                ),
            ),
        ]
        try:
            response = await self._llm.generate(
                messages=messages,
                config=self._config,
                tools=None,
            )
        except Exception:
            return QueryRefinementResult(
                refined_query=fallback,
                used_fallback=True,
            )

        refined = response.content.strip()
        if not refined:
            return QueryRefinementResult(
                refined_query=fallback,
                used_fallback=True,
            )
        return QueryRefinementResult(
            refined_query=refined,
            used_fallback=False,
        )
