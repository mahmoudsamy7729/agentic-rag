from __future__ import annotations

from pathlib import Path

import chromadb

from src.rag.models import RAGChunk, RetrievedChunk
from src.rag.vectorstore.interface import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, *, persist_dir: str, collection_name: str) -> None:
        path = Path(persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    async def upsert_chunks(
        self,
        *,
        chunks: list[RAGChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ids = [f"{chunk.doc_id}:{chunk.chunk_id}" for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
            }
            for chunk in chunks
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        where = {"doc_id": doc_id} if doc_id else None
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            results.append(
                RetrievedChunk(
                    doc_id=str((metadata or {}).get("doc_id", "")),
                    chunk_id=str((metadata or {}).get("chunk_id", "")),
                    source=str((metadata or {}).get("source", "unknown")),
                    text=str(document),
                    score=score,
                )
            )
        return results
