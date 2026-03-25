from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from src.agents.cache_policy import is_cacheable_rag_answer, is_no_answer_fallback
from src.modules.semantic_cache.service import SemanticCacheService


@dataclass
class FakeEntry:
    answer: str
    citations: list[dict]


class FakeRepository:
    def __init__(self, entry: FakeEntry | None = None) -> None:
        self.entry = entry
        self.lookup_calls = 0
        self.store_calls = 0
        self.commits = 0
        self.last_store_kwargs: dict | None = None

    async def lookup(self, **kwargs):
        self.lookup_calls += 1
        return self.entry

    async def store(self, **kwargs):
        self.store_calls += 1
        self.last_store_kwargs = kwargs
        return None

    async def commit(self):
        self.commits += 1


def test_normalize_question_collapses_spaces_and_lowercases():
    normalized = SemanticCacheService.normalize_question("  What   IS   Refund   Rule?  ")
    assert normalized == "what is refund rule?"


def test_lookup_returns_hit_when_enabled():
    async def _run():
        repo = FakeRepository(entry=FakeEntry(answer="a", citations=[{"doc_id": "x"}]))
        service = SemanticCacheService(
            repository=repo,
            enabled=True,
            similarity_threshold=0.92,
        )
        hit = await service.lookup(
            owner_user_id=uuid4(),
            doc_id="doc-1",
            doc_version=datetime.now(timezone.utc),
            model_name="gpt-test",
            query_embedding=[0.1, 0.2],
        )
        assert hit is not None
        assert hit.answer == "a"
        assert hit.citations == [{"doc_id": "x"}]
        assert repo.lookup_calls == 1

    asyncio.run(_run())


def test_lookup_skips_when_disabled():
    async def _run():
        repo = FakeRepository(entry=FakeEntry(answer="a", citations=[]))
        service = SemanticCacheService(
            repository=repo,
            enabled=False,
            similarity_threshold=0.92,
        )
        hit = await service.lookup(
            owner_user_id=uuid4(),
            doc_id="doc-1",
            doc_version=datetime.now(timezone.utc),
            model_name="gpt-test",
            query_embedding=[0.1, 0.2],
        )
        assert hit is None
        assert repo.lookup_calls == 0

    asyncio.run(_run())


def test_store_persists_and_commits_when_enabled():
    async def _run():
        repo = FakeRepository()
        service = SemanticCacheService(
            repository=repo,
            enabled=True,
            similarity_threshold=0.92,
        )
        await service.store(
            owner_user_id=uuid4(),
            doc_id="doc-1",
            doc_version=datetime.now(timezone.utc),
            model_name="gpt-test",
            question_normalized="q",
            question_embedding=[0.1, 0.2],
            answer="answer",
            citations=[{"doc_id": "doc-1"}],
        )
        assert repo.store_calls == 1
        assert repo.commits == 1

    asyncio.run(_run())


def test_store_skips_when_disabled():
    async def _run():
        repo = FakeRepository()
        service = SemanticCacheService(
            repository=repo,
            enabled=False,
            similarity_threshold=0.92,
        )
        await service.store(
            owner_user_id=uuid4(),
            doc_id="doc-1",
            doc_version=datetime.now(timezone.utc),
            model_name="gpt-test",
            question_normalized="q",
            question_embedding=[0.1, 0.2],
            answer="answer",
            citations=[{"doc_id": "doc-1"}],
        )
        assert repo.store_calls == 0
        assert repo.commits == 0

    asyncio.run(_run())


def test_cacheable_policy_only_retriever_and_non_empty_citations():
    assert is_cacheable_rag_answer(
        tools_used=["retrieve_context"],
        citations=[{"doc_id": "doc-1"}],
    )
    assert not is_cacheable_rag_answer(
        tools_used=[],
        citations=[{"doc_id": "doc-1"}],
    )
    assert not is_cacheable_rag_answer(
        tools_used=["retrieve_context", "query_user_data"],
        citations=[{"doc_id": "doc-1"}],
    )
    assert not is_cacheable_rag_answer(
        tools_used=["retrieve_context"],
        citations=[],
    )


def test_no_answer_fallback_policy_detects_canonical_phrase():
    assert is_no_answer_fallback(
        "  I could not find the answer in the provided documents. "
    )
    assert is_no_answer_fallback("No   relevant   context found.")


def test_no_answer_fallback_policy_allows_grounded_answers():
    assert not is_no_answer_fallback("Paris is the capital of France.")
