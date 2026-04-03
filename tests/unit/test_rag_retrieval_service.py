import asyncio
import json
import logging

import pytest

from src.rag.models import RetrievedChunk
from src.rag.pipeline.services import RAGRetrievalService
from src.shared.tracing import TRACE_LOGGER_NAME, TraceContext


class FakeEmbeddingProvider:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.last_top_k: int | None = None
        self.last_doc_id: str | None = None

    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        doc_id: str | None = None,
    ):
        self.last_top_k = top_k
        self.last_doc_id = doc_id
        filtered = [item for item in self._results if (not doc_id or item.doc_id == doc_id)]
        return filtered[:top_k]


class FlakyReranker:
    def __init__(self, *, fail_times: int, response: list[RetrievedChunk]) -> None:
        self.fail_times = fail_times
        self.response = response
        self.calls = 0
        self.last_top_n: int | None = None

    async def rerank(self, *, query: str, chunks: list[RetrievedChunk], top_n: int):
        self.calls += 1
        self.last_top_n = top_n
        if self.calls <= self.fail_times:
            raise RuntimeError("temporary cohere failure")
        return self.response[:top_n]


def _candidates() -> list[RetrievedChunk]:
    return [
        RetrievedChunk("doc-1", "chunk-1", "source-a", "text a", 0.6),
        RetrievedChunk("doc-2", "chunk-2", "source-b", "text b", 0.5),
        RetrievedChunk("doc-2", "chunk-3", "source-c", "text c", 0.4),
        RetrievedChunk("doc-3", "chunk-4", "source-d", "text d", 0.3),
        RetrievedChunk("doc-4", "chunk-5", "source-e", "text e", 0.2),
        RetrievedChunk("doc-5", "chunk-6", "source-f", "text f", 0.1),
    ]


def _trace_events(caplog) -> list[dict]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == TRACE_LOGGER_NAME
    ]


def test_retrieval_uses_prefetch_and_reranks_to_final_top_k(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER_NAME)
    candidates = _candidates()
    vector_store = FakeVectorStore(candidates)
    reranker = FlakyReranker(fail_times=0, response=list(reversed(candidates)))

    service = RAGRetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        default_top_k=5,
        prefetch_k=50,
        reranker=reranker,
    )

    result = asyncio.run(
        service.retrieve(
            query="refund policy",
            doc_id="doc-2",
            trace_context=TraceContext(request_id="req-1", doc_id="doc-2"),
        )
    )

    assert vector_store.last_top_k == 50
    assert vector_store.last_doc_id == "doc-2"
    assert reranker.last_top_n == 5
    assert len(result) == 5
    events = _trace_events(caplog)
    assert [event["event"] for event in events] == [
        "retrieval.started",
        "retrieval.vector_search.completed",
        "retrieval.rerank.succeeded",
        "retrieval.completed",
    ]
    assert events[2]["reranker_status"] == "succeeded_first_try"
    assert events[3]["request_id"] == "req-1"


def test_retrieval_retries_reranker_once_then_succeeds(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER_NAME)
    candidates = _candidates()
    vector_store = FakeVectorStore(candidates)
    reranker = FlakyReranker(fail_times=1, response=candidates)

    service = RAGRetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        default_top_k=5,
        prefetch_k=50,
        reranker=reranker,
    )

    result = asyncio.run(
        service.retrieve(
            query="refund policy",
            trace_context=TraceContext(request_id="req-2", doc_id="doc-1"),
        )
    )

    assert reranker.calls == 2
    assert len(result) == 5
    events = _trace_events(caplog)
    assert [event["event"] for event in events] == [
        "retrieval.started",
        "retrieval.vector_search.completed",
        "retrieval.rerank.retry",
        "retrieval.rerank.succeeded",
        "retrieval.completed",
    ]
    assert events[3]["reranker_status"] == "succeeded_on_retry"
    assert events[3]["reranker_attempt_count"] == 2


def test_retrieval_raises_after_retry_exhausted(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER_NAME)
    candidates = _candidates()
    vector_store = FakeVectorStore(candidates)
    reranker = FlakyReranker(fail_times=2, response=candidates)

    service = RAGRetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        default_top_k=5,
        prefetch_k=50,
        reranker=reranker,
    )

    with pytest.raises(RuntimeError, match="Reranker failed"):
        asyncio.run(
            service.retrieve(
                query="refund policy",
                trace_context=TraceContext(request_id="req-3", doc_id="doc-1"),
            )
        )
    assert reranker.calls == 2
    events = _trace_events(caplog)
    assert [event["event"] for event in events] == [
        "retrieval.started",
        "retrieval.vector_search.completed",
        "retrieval.rerank.retry",
        "retrieval.rerank.failed",
    ]
    assert events[-1]["reranker_status"] == "failed"
    assert events[-1]["request_id"] == "req-3"


def test_retrieval_without_reranker_returns_base_results_truncated(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER_NAME)
    candidates = _candidates()
    vector_store = FakeVectorStore(candidates)

    service = RAGRetrievalService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        default_top_k=5,
        prefetch_k=50,
        reranker=None,
    )

    result = asyncio.run(
        service.retrieve(
            query="refund policy",
            top_k=3,
            doc_id="doc-2",
            trace_context=TraceContext(request_id="req-4", doc_id="doc-2"),
        )
    )

    assert vector_store.last_top_k == 50
    assert vector_store.last_doc_id == "doc-2"
    assert len(result) == 2
    assert [item.doc_id for item in result] == ["doc-2", "doc-2"]
    events = _trace_events(caplog)
    assert [event["event"] for event in events] == [
        "retrieval.started",
        "retrieval.vector_search.completed",
        "retrieval.completed",
    ]
    assert events[-1]["reranker_status"] == "disabled"
