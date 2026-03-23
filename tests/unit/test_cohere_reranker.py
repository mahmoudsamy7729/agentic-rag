import asyncio

import pytest

from src.infrastructure.reranker.cohere_reranker import CohereReranker
from src.rag.models import RetrievedChunk


class FakeDocument:
    def __init__(self, *, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def _sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            doc_id="doc-1",
            chunk_id="chunk-0",
            source="policy.md",
            text="Refunds for digital products are limited.",
            score=0.30,
            page_number=4,
        ),
        RetrievedChunk(
            doc_id="doc-2",
            chunk_id="chunk-0",
            source="terms.md",
            text="Users can request refunds within 7 days.",
            score=0.20,
            page_number=9,
        ),
        RetrievedChunk(
            doc_id="doc-3",
            chunk_id="chunk-0",
            source="faq.md",
            text="Support can approve exceptions.",
            score=0.10,
            page_number=12,
        ),
    ]


def test_cohere_reranker_returns_ranked_chunks(monkeypatch):
    class FakeCohereRerank:
        def __init__(self, *, model: str, cohere_api_key: str, top_n: int):
            self.model = model
            self.cohere_api_key = cohere_api_key
            self.top_n = top_n

        def compress_documents(self, *, documents, query):
            top_docs = [documents[1], documents[0]][: self.top_n]
            top_docs[0].metadata["relevance_score"] = 0.98
            top_docs[1].metadata["relevance_score"] = 0.87
            return top_docs

    monkeypatch.setattr(
        "src.infrastructure.reranker.cohere_reranker._CohereRerank",
        FakeCohereRerank,
    )
    monkeypatch.setattr(
        "src.infrastructure.reranker.cohere_reranker._Document",
        FakeDocument,
    )

    reranker = CohereReranker(api_key="cohere-key", model="rerank-v4.0-fast")
    result = asyncio.run(
        reranker.rerank(
            query="refund rules",
            chunks=_sample_chunks(),
            top_n=2,
        )
    )

    assert len(result) == 2
    assert result[0].doc_id == "doc-2"
    assert result[0].score == pytest.approx(0.98)
    assert result[0].page_number == 9
    assert result[1].doc_id == "doc-1"
    assert result[1].score == pytest.approx(0.87)
    assert result[1].page_number == 4


def test_cohere_reranker_raises_on_malformed_payload(monkeypatch):
    class FakeCohereRerank:
        def __init__(self, *, model: str, cohere_api_key: str, top_n: int):
            self.top_n = top_n

        def compress_documents(self, *, documents, query):
            return "invalid-payload"

    monkeypatch.setattr(
        "src.infrastructure.reranker.cohere_reranker._CohereRerank",
        FakeCohereRerank,
    )
    monkeypatch.setattr(
        "src.infrastructure.reranker.cohere_reranker._Document",
        FakeDocument,
    )

    reranker = CohereReranker(api_key="cohere-key", model="rerank-v4.0-fast")
    with pytest.raises(ValueError, match="invalid payload"):
        asyncio.run(
            reranker.rerank(
                query="refund rules",
                chunks=_sample_chunks(),
                top_n=2,
            )
        )
