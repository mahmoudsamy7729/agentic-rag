import asyncio

from src.rag.models import RetrievedChunk
from src.tools.retriever_tool import RetrieverTool


class FakeRetrievalService:
    def __init__(self) -> None:
        self.last_top_k: int | None = None

    async def retrieve(self, *, query: str, top_k: int | None = None):
        self.last_top_k = top_k
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
        return await tool.run({"query": "refund policy", "top_k": 2})

    result = asyncio.run(_run())

    assert result.success is True
    assert result.output["query"] == "refund policy"
    assert len(result.output["results"]) == 1
    assert result.output["results"][0]["doc_id"] == "doc-1"
    assert result.output["results"][0]["source"] == "policy.md"
    assert retrieval_service.last_top_k == 4
