from __future__ import annotations

import asyncio

from src.agents.query_refinement import QueryRefinementService
from src.shared.interfaces.llm import (
    ChatMessage,
    GenerationConfig,
    LLM,
    LLMResponse,
)


class FakeLLM(LLM):
    def __init__(self, *, content: str = "", should_raise: bool = False) -> None:
        self._content = content
        self._should_raise = should_raise
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-llm"

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        config: GenerationConfig | None = None,
        tools=None,
    ) -> LLMResponse:
        self.calls += 1
        if self._should_raise:
            raise RuntimeError("boom")
        return LLMResponse(content=self._content, model=self.model_name)

    async def stream(self, messages, *, config=None, tools=None):
        if False:
            yield ""


def test_refine_success_returns_rewritten_query():
    async def _run():
        service = QueryRefinementService(
            llm=FakeLLM(content="refund policy conditions"),
            enabled=True,
            temperature=0.0,
            max_tokens=64,
        )
        result = await service.refine(
            question="what is refund rule?",
            doc_id="doc-1",
        )
        assert result.refined_query == "refund policy conditions"
        assert result.used_fallback is False

    asyncio.run(_run())


def test_refine_falls_back_when_empty_output():
    async def _run():
        service = QueryRefinementService(
            llm=FakeLLM(content="   "),
            enabled=True,
            temperature=0.0,
            max_tokens=64,
        )
        result = await service.refine(
            question="what is refund rule?",
            doc_id="doc-1",
        )
        assert result.refined_query == "what is refund rule?"
        assert result.used_fallback is True

    asyncio.run(_run())


def test_refine_falls_back_when_llm_fails():
    async def _run():
        service = QueryRefinementService(
            llm=FakeLLM(should_raise=True),
            enabled=True,
            temperature=0.0,
            max_tokens=64,
        )
        result = await service.refine(
            question="what is refund rule?",
            doc_id="doc-1",
        )
        assert result.refined_query == "what is refund rule?"
        assert result.used_fallback is True

    asyncio.run(_run())


def test_refine_falls_back_when_disabled_without_llm_call():
    async def _run():
        llm = FakeLLM(content="rewritten")
        service = QueryRefinementService(
            llm=llm,
            enabled=False,
            temperature=0.0,
            max_tokens=64,
        )
        result = await service.refine(
            question="what is refund rule?",
            doc_id="doc-1",
        )
        assert result.refined_query == "what is refund rule?"
        assert result.used_fallback is True
        assert llm.calls == 0

    asyncio.run(_run())
