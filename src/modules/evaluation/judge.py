from __future__ import annotations

import json
from dataclasses import dataclass

from src.shared.interfaces.llm import ChatMessage, GenerationConfig, LLM, MessageRole


@dataclass(frozen=True, slots=True)
class ContextRelevanceJudgeResult:
    score: int
    explanation: str


class ContextRelevanceJudge:
    def __init__(
        self,
        *,
        llm: LLM,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_s = timeout_s

    async def judge(
        self,
        *,
        question: str,
        retrieved_chunks: list[dict[str, str]],
    ) -> ContextRelevanceJudgeResult:
        response = await self._llm.generate(
            messages=[
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You are a retrieval evaluation judge. "
                        "Score whether the provided retrieved context is sufficient to answer the question. "
                        "Return strict JSON only with keys: score, explanation. "
                        "Score must be an integer from 1 to 5."
                    ),
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=self._build_prompt(
                        question=question,
                        retrieved_chunks=retrieved_chunks,
                    ),
                ),
            ],
            config=GenerationConfig(
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout_s=self._timeout_s,
                response_format={"type": "json_object"},
            ),
        )
        return self._parse_response(response.content)

    @staticmethod
    def _build_prompt(*, question: str, retrieved_chunks: list[dict[str, str]]) -> str:
        payload = {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "instructions": [
                "Score 1 if the context is irrelevant or clearly insufficient.",
                "Score 2 if the context is weakly relevant and missing major evidence.",
                "Score 3 if the context is partially sufficient but still missing important evidence.",
                "Score 4 if the context is mostly sufficient with only minor missing details.",
                "Score 5 if the context is fully sufficient to answer confidently.",
                "Explain briefly whether the context is relevant, sufficient, and what evidence is missing if any.",
            ],
        }
        return json.dumps(payload, ensure_ascii=True)

    @staticmethod
    def _parse_response(content: str) -> ContextRelevanceJudgeResult:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Context relevance judge returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Context relevance judge returned invalid payload.")
        score = payload.get("score")
        explanation = payload.get("explanation")
        if not isinstance(score, int) or score < 1 or score > 5:
            raise ValueError("Context relevance judge score must be an integer from 1 to 5.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("Context relevance judge explanation must be a non-empty string.")
        return ContextRelevanceJudgeResult(score=score, explanation=explanation.strip())
