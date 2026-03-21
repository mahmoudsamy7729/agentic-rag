import asyncio

from src.rag.models import RetrievedChunk
from src.tools.retriever_tool import RetrieverTool


class FakeRetrievalService:
    async def retrieve(self, *, query: str, top_k: int | None = None):
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
    async def _run():
        tool = RetrieverTool(retrieval_service=FakeRetrievalService(), default_top_k=4)
        return await tool.run({"query": "refund policy", "top_k": 2})

    result = asyncio.run(_run())

    assert result.success is True
    assert result.output["query"] == "refund policy"
    assert len(result.output["results"]) == 1
    assert result.output["results"][0]["doc_id"] == "doc-1"
    assert result.output["results"][0]["source"] == "policy.md"
