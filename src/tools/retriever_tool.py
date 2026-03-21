from __future__ import annotations

from typing import Any

from src.rag.pipeline import RAGRetrievalService
from src.shared.interfaces.tool import Tool, ToolContext, ToolExecutionResult


class RetrieverTool(Tool):
    def __init__(self, *, retrieval_service: RAGRetrievalService, default_top_k: int) -> None:
        self._retrieval_service = retrieval_service
        self._default_top_k = default_top_k

    @property
    def name(self) -> str:
        return "retrieve_context"

    @property
    def description(self) -> str:
        return "Retrieve relevant context chunks from indexed documents for a query."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve relevant document chunks.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of chunks to retrieve.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def run(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolContext | None = None,
    ) -> ToolExecutionResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolExecutionResult(success=False, error="Missing required argument: query")

        top_k_value = arguments.get("top_k", self._default_top_k)
        try:
            top_k = int(top_k_value)
        except (TypeError, ValueError):
            return ToolExecutionResult(success=False, error="top_k must be an integer")

        if top_k < 1:
            return ToolExecutionResult(success=False, error="top_k must be >= 1")

        chunks = await self._retrieval_service.retrieve(query=query, top_k=top_k)
        return ToolExecutionResult(
            success=True,
            output={
                "query": query,
                "results": [
                    {
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "text": chunk.text,
                        "score": chunk.score,
                    }
                    for chunk in chunks
                ],
                "session_id": context.session_id if context else None,
                "user_id": context.user_id if context else None,
            },
        )
