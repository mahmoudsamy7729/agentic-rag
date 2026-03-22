import asyncio

from src.rag.models import RetrievedChunk
from src.shared.interfaces.tool import ToolContext
from src.tools.retriever_tool import RetrieverTool


class FakeRetrievalService:
    def __init__(self) -> None:
        self.last_top_k: int | None = None
        self.last_doc_id: str | None = None

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        doc_id: str | None = None,
    ):
        self.last_top_k = top_k
        self.last_doc_id = doc_id
        return [
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="chunk-0",
                source="policy.md",
                text=f"Found context for: {query}",
                score=0.91,
            )
        ]


def test_retriever_tool_returns_structured_results():
    retrieval_service = FakeRetrievalService()

    async def _run():
        tool = RetrieverTool(retrieval_service=retrieval_service, default_top_k=4)
        return await tool.run(
            {"query": "refund policy"},
            context=ToolContext(doc_id="doc-1"),
        )

    result = asyncio.run(_run())

    assert result.success is True
    assert result.output["query"] == "refund policy"
    assert result.output["doc_id"] == "doc-1"
    assert len(result.output["results"]) == 1
    assert result.output["results"][0]["doc_id"] == "doc-1"
    assert result.output["results"][0]["source"] == "policy.md"
    assert retrieval_service.last_top_k == 4
    assert retrieval_service.last_doc_id == "doc-1"


def test_retriever_tool_requires_doc_id_in_context():
    retrieval_service = FakeRetrievalService()

    async def _run():
        tool = RetrieverTool(retrieval_service=retrieval_service, default_top_k=4)
        return await tool.run({"query": "refund policy"})

    result = asyncio.run(_run())

    assert result.success is False
    assert "doc_id" in (result.error or "")
