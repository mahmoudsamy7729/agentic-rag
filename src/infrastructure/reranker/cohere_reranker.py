from __future__ import annotations

import asyncio
from typing import Any

from src.rag.models import RetrievedChunk
from src.rag.reranker import Reranker

try:
    from langchain_cohere import CohereRerank as _CohereRerank
    from langchain_core.documents import Document as _Document
except ImportError:
    _CohereRerank = None
    _Document = None


class CohereReranker(Reranker):
    def __init__(self, *, api_key: str, model: str) -> None:
        if _CohereRerank is None or _Document is None:
            raise RuntimeError(
                "langchain-cohere is not installed. Install it to use RERANKER_ENABLED=true."
            )
        self._api_key = api_key
        self._model = model

    async def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if not chunks:
            return []

        target_top_n = min(top_n, len(chunks))
        return await asyncio.to_thread(
            self._rerank_sync,
            query=query,
            chunks=chunks,
            top_n=target_top_n,
        )

    def _rerank_sync(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        reranker = _CohereRerank(
            model=self._model,
            cohere_api_key=self._api_key,
            top_n=top_n,
        )

        docs = [
            _Document(
                page_content=chunk.text,
                metadata={
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "score": chunk.score,
                    "page_number": chunk.page_number,
                    "_candidate_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]

        reranked_docs = reranker.compress_documents(documents=docs, query=query)
        if not isinstance(reranked_docs, list):
            raise ValueError("Cohere reranker returned invalid payload.")

        reranked_chunks: list[RetrievedChunk] = []
        for doc in reranked_docs:
            if not hasattr(doc, "page_content") or not hasattr(doc, "metadata"):
                raise ValueError("Cohere reranker returned malformed documents.")
            metadata = self._normalize_metadata(doc.metadata)

            index = metadata.get("_candidate_index")
            if isinstance(index, int) and 0 <= index < len(chunks):
                original = chunks[index]
                doc_id = original.doc_id
                chunk_id = original.chunk_id
                source = original.source
                page_number = original.page_number
            else:
                doc_id = str(metadata.get("doc_id", ""))
                chunk_id = str(metadata.get("chunk_id", ""))
                source = str(metadata.get("source", "unknown"))
                page_number = self._parse_page_number(metadata.get("page_number"))

            score = metadata.get("relevance_score", metadata.get("score", 0.0))
            reranked_chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    source=source,
                    text=str(doc.page_content),
                    score=float(score),
                    page_number=page_number,
                )
            )

        return reranked_chunks

    @staticmethod
    def _normalize_metadata(metadata: Any) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        return metadata

    @staticmethod
    def _parse_page_number(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
