import asyncio

import pytest

from src.modules.evaluation.judge import ContextRelevanceJudge
from src.shared.interfaces.llm import LLMResponse


class FakeJudgeLLM:
    async def generate(self, messages, *, config=None, tools=None):
        return LLMResponse(
            content='{"score":4,"explanation":"The context is mostly sufficient."}',
            model="fake-judge",
        )


class BadJudgeLLM:
    async def generate(self, messages, *, config=None, tools=None):
        return LLMResponse(content="not-json", model="fake-judge")


def test_context_relevance_judge_parses_strict_json():
    judge = ContextRelevanceJudge(
        llm=FakeJudgeLLM(),
        max_tokens=128,
        temperature=0.0,
        timeout_s=5.0,
    )

    result = asyncio.run(
        judge.judge(
            question="What is the refund policy?",
            retrieved_chunks=[{"chunk_id": "chunk-1", "text": "refund within 30 days"}],
        )
    )

    assert result.score == 4
    assert result.explanation == "The context is mostly sufficient."


def test_context_relevance_judge_rejects_invalid_json():
    judge = ContextRelevanceJudge(
        llm=BadJudgeLLM(),
        max_tokens=128,
        temperature=0.0,
        timeout_s=5.0,
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        asyncio.run(
            judge.judge(
                question="What is the refund policy?",
                retrieved_chunks=[{"chunk_id": "chunk-1", "text": "refund within 30 days"}],
            )
        )
