import ast
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("chromadb")

from main import app
from src.agents import AgentCitation, AgentResult, AgentService
from src.api.v1 import dependencies as deps
from src.modules.documents.dependencies import get_documents_repository
from src.modules.users.dependencies import active_user
from src.rag.models import RetrievedChunk
from src.rag.pipeline import IngestionResult, PDFIngestionResult
from src.shared.interfaces.llm import (
    ChatMessage,
    GenerationConfig,
    LLM,
    LLMResponse,
    MessageRole,
    ToolCall,
)
from src.tools import PingTool, RetrieverTool, ToolRegistry


class InMemoryRAGStore:
    def __init__(self) -> None:
        self.items: list[dict] = []


class FakeVectorStore:
    async def delete_by_doc_id(self, *, doc_id: str) -> None:
        return None


class FakeIngestionService:
    def __init__(self, store: InMemoryRAGStore) -> None:
        self._store = store

    async def ingest_text(self, *, text: str, source: str | None = None, doc_id: str | None = None):
        resolved_doc_id = doc_id or f"doc-{len(self._store.items) + 1}"
        self._store.items.append(
            {
                "doc_id": resolved_doc_id,
                "chunk_id": "chunk-0",
                "source": source or "inline-text",
                "text": text,
                "page_number": None,
            }
        )
        return IngestionResult(doc_id=resolved_doc_id, chunks_ingested=1)

    async def ingest_pdf(self, *, pdf_bytes: bytes, source: str | None = None, doc_id: str | None = None):
        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty.")
        resolved_doc_id = doc_id or f"doc-{len(self._store.items) + 1}"
        self._store.items.append(
            {
                "doc_id": resolved_doc_id,
                "chunk_id": "chunk-0",
                "source": source or "uploaded-pdf",
                "text": "Refund rules for digital products are 7 days.",
                "page_number": 1,
            }
        )
        return PDFIngestionResult(
            doc_id=resolved_doc_id,
            chunks_ingested=1,
            pages_total=2,
            pages_ingested=1,
            skipped_pages=[2],
            warnings=["Page 2: no extractable text or tables."],
        )


class FakeRetrievalService:
    def __init__(self, store: InMemoryRAGStore) -> None:
        self._store = store

    async def retrieve(self, *, query: str, top_k: int | None = None, doc_id: str | None = None):
        lowered_query = query.lower()
        hits = [
            item
            for item in self._store.items
            if (not doc_id or item["doc_id"] == doc_id)
            and any(token in item["text"].lower() for token in lowered_query.split())
        ]
        limit = top_k or 4
        return [
            RetrievedChunk(
                doc_id=item["doc_id"],
                chunk_id=item["chunk_id"],
                source=item["source"],
                text=item["text"],
                score=0.9,
                page_number=item.get("page_number"),
            )
            for item in hits[:limit]
        ]


class FakeLLM(LLM):
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
        has_tool_result = any(
            msg.role == MessageRole.TOOL and msg.name == "retrieve_context"
            for msg in messages
        )

        if not has_tool_result:
            user_question = next(
                (msg.content for msg in reversed(messages) if msg.role == MessageRole.USER),
                "",
            )
            return LLMResponse(
                content="",
                model=self.model_name,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="retrieve_context",
                        arguments={"query": user_question},
                    )
                ],
            )

        tool_message = next(
            (
                msg
                for msg in reversed(messages)
                if msg.role == MessageRole.TOOL and msg.name == "retrieve_context"
            ),
            None,
        )

        results: list[dict] = []
        if tool_message:
            try:
                payload = ast.literal_eval(tool_message.content)
                if isinstance(payload, dict):
                    results = payload.get("results", []) or []
            except (ValueError, SyntaxError):
                results = []

        if results:
            answer = f"Based on retrieved context: {results[0].get('text', '')}"
        else:
            answer = "No relevant context found."

        return LLMResponse(content=answer, model=self.model_name)

    async def stream(self, messages, *, config=None, tools=None):
        if False:
            yield ""



class FakeNoAnswerAgentService:
    async def run(
        self,
        *,
        question: str,
        doc_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            answer="I could not find the answer in the provided documents.",
            steps=1,
            tools_used=["retrieve_context"],
            status="ok",
            citations=[
                AgentCitation(
                    source="inline-text",
                    doc_id=doc_id,
                    chunk_id="chunk-0",
                    snippet="Some snippet",
                    page_number=None,
                )
            ],
        )

class FakeEmbeddingProvider:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        checksum = float(sum(ord(char) for char in normalized))
        return [checksum, float(len(normalized) or 1)]


class FakeQueryRefinementService:
    @staticmethod
    async def refine(*, question: str, doc_id: str):
        lowered = question.lower()
        if "capital" in lowered or "city" in lowered:
            refined = f"capital city for {doc_id}"
        elif "refund" in lowered:
            refined = f"refund policy for {doc_id}"
        else:
            refined = question.strip()
        return type(
            "Refinement",
            (),
            {"refined_query": refined, "used_fallback": refined == question.strip()},
        )()


@dataclass
class FakeSemanticCacheHit:
    answer: str
    citations: list[dict]


class FakeSemanticCacheService:
    def __init__(self, *, enabled: bool = True):
        self._enabled = enabled
        self._entries: dict[tuple, FakeSemanticCacheHit] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question.strip().lower())

    async def lookup(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        doc_version: datetime,
        model_name: str,
        query_embedding: list[float],
    ) -> FakeSemanticCacheHit | None:
        key = (
            owner_user_id,
            doc_id,
            doc_version,
            model_name,
            tuple(query_embedding),
        )
        return self._entries.get(key)

    async def store(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        doc_version: datetime,
        model_name: str,
        question_normalized: str,
        question_embedding: list[float],
        answer: str,
        citations: list[dict],
    ) -> None:
        key = (
            owner_user_id,
            doc_id,
            doc_version,
            model_name,
            tuple(question_embedding),
        )
        self._entries[key] = FakeSemanticCacheHit(answer=answer, citations=citations)


@dataclass
class FakeUser:
    id: UUID
    is_active: bool = True


@dataclass
class FakeDocument:
    id: str
    owner_user_id: UUID
    source: str | None
    created_at: datetime
    updated_at: datetime
    last_indexed_at: datetime | None = None
    deleted_at: datetime | None = None


class FakeDocumentsRepository:
    def __init__(self) -> None:
        self._items: dict[str, FakeDocument] = {}

    async def create_document(self, *, owner_user_id: UUID, doc_id: str, source: str | None):
        now = datetime.now(timezone.utc)
        doc = FakeDocument(
            id=doc_id,
            owner_user_id=owner_user_id,
            source=source,
            created_at=now,
            updated_at=now,
            last_indexed_at=None,
            deleted_at=None,
        )
        self._items[doc_id] = doc
        return doc

    async def get_owned_document(self, *, owner_user_id: UUID, doc_id: str, include_deleted: bool = False):
        item = self._items.get(doc_id)
        if item is None:
            return None
        if item.owner_user_id != owner_user_id:
            return None
        if not include_deleted and item.deleted_at is not None:
            return None
        return item

    async def list_owned_documents(self, *, owner_user_id: UUID, limit: int, offset: int):
        items = [
            item
            for item in self._items.values()
            if item.owner_user_id == owner_user_id and item.deleted_at is None
        ]
        items.sort(key=lambda item: item.created_at, reverse=True)
        return items[offset : offset + limit]

    async def soft_delete_owned_document(self, *, owner_user_id: UUID, doc_id: str):
        item = await self.get_owned_document(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            include_deleted=True,
        )
        if item is None:
            return None
        if item.deleted_at is None:
            now = datetime.now(timezone.utc)
            item.deleted_at = now
            item.updated_at = now
        return item

    async def doc_id_exists(self, *, doc_id: str, include_deleted: bool = True) -> bool:
        item = self._items.get(doc_id)
        if item is None:
            return False
        if include_deleted:
            return True
        return item.deleted_at is None

    async def mark_document_indexed(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        indexed_at: datetime | None = None,
    ):
        item = await self.get_owned_document(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            include_deleted=False,
        )
        if item is None:
            return None
        now = indexed_at or datetime.now(timezone.utc)
        item.last_indexed_at = now
        item.updated_at = now
        return item

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


def _build_client():
    store = InMemoryRAGStore()
    ingestion_service = FakeIngestionService(store)
    retrieval_service = FakeRetrievalService(store)
    docs_repo = FakeDocumentsRepository()
    owner = FakeUser(id=uuid4())
    embedding_provider = FakeEmbeddingProvider()
    query_refinement_service = FakeQueryRefinementService()
    semantic_cache_service = FakeSemanticCacheService(enabled=True)
    registry = ToolRegistry(
        [
            PingTool(),
            RetrieverTool(retrieval_service=retrieval_service, default_top_k=4),
        ]
    )
    llm = FakeLLM()

    app.dependency_overrides[deps.get_rag_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[deps.get_tool_registry] = lambda: registry
    app.dependency_overrides[deps.get_llm] = lambda: llm
    app.dependency_overrides[deps.get_query_refinement_service] = lambda: query_refinement_service
    app.dependency_overrides[deps.get_embedding_provider] = lambda: embedding_provider
    app.dependency_overrides[deps.get_semantic_cache_service] = lambda: semantic_cache_service
    app.dependency_overrides[deps.get_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[get_documents_repository] = lambda: docs_repo
    app.dependency_overrides[active_user] = lambda: owner
    app.dependency_overrides[deps.get_agent_service] = lambda: AgentService(
        llm=llm,
        registry=registry,
        max_steps=4,
    )

    return TestClient(app)


def test_end_to_end_rag_flow_with_doc_scoped_citations():
    client = _build_client()
    try:
        llm_health = client.get("/llm/health")
        tools_health = client.get("/tools/health")

        assert llm_health.status_code == 200
        assert llm_health.json()["llm_ok"] is True
        assert tools_health.status_code == 200
        assert tools_health.json()["tools_ok"] is True

        ingest_a = client.post(
            "/rag/ingest/text",
            json={
                "text": "Paris is the capital of France.",
                "source": "wiki-france",
                "doc_id": "doc-fr",
            },
        )
        assert ingest_a.status_code == 200

        ingest_b = client.post(
            "/rag/ingest/text",
            json={
                "text": "Cairo is the capital of Egypt.",
                "source": "wiki-egypt",
                "doc_id": "doc-eg",
            },
        )
        assert ingest_b.status_code == 200

        ask_fr = client.post(
            "/agent/ask",
            json={
                "question": "What is the capital?",
                "doc_id": "doc-fr",
                "session_id": "s-1",
            },
        )
        body_fr = ask_fr.json()
        assert ask_fr.status_code == 200
        assert body_fr["status"] == "ok"
        assert body_fr["cache_status"] == "miss"
        assert body_fr["refined_query"] == "capital city for doc-fr"
        assert len(body_fr["citations"]) >= 1
        assert all(c["doc_id"] == "doc-fr" for c in body_fr["citations"])

        ask_fr_again = client.post(
            "/agent/ask",
            json={
                "question": "Capital city?",
                "doc_id": "doc-fr",
                "session_id": "s-1",
            },
        )
        body_fr_again = ask_fr_again.json()
        assert ask_fr_again.status_code == 200
        assert body_fr_again["cache_status"] == "hit"
        assert body_fr_again["refined_query"] == "capital city for doc-fr"
        assert body_fr_again["steps"] == 0
        assert body_fr_again["tools_used"] == []

        ask_eg = client.post(
            "/agent/ask",
            json={
                "question": "What is the capital?",
                "doc_id": "doc-eg",
                "session_id": "s-1",
            },
        )
        body_eg = ask_eg.json()
        assert ask_eg.status_code == 200
        assert body_eg["status"] == "ok"
        assert body_eg["cache_status"] == "miss"
        assert body_eg["refined_query"] == "capital city for doc-eg"
        assert len(body_eg["citations"]) >= 1
        assert all(c["doc_id"] == "doc-eg" for c in body_eg["citations"])
    finally:
        app.dependency_overrides.clear()



def test_no_answer_fallback_is_never_cached():
    client = _build_client()
    try:
        ingest = client.post(
            "/rag/ingest/text",
            json={
                "text": "Paris is the capital of France.",
                "source": "wiki-france",
                "doc_id": "doc-no-answer",
            },
        )
        assert ingest.status_code == 200

        app.dependency_overrides[deps.get_agent_service] = lambda: FakeNoAnswerAgentService()

        ask_first = client.post(
            "/agent/ask",
            json={
                "question": "unknown query",
                "doc_id": "doc-no-answer",
                "session_id": "s-1",
            },
        )
        body_first = ask_first.json()
        assert ask_first.status_code == 200
        assert body_first["cache_status"] == "miss"
        assert body_first["answer"] == "I could not find the answer in the provided documents."

        ask_second = client.post(
            "/agent/ask",
            json={
                "question": "unknown query",
                "doc_id": "doc-no-answer",
                "session_id": "s-1",
            },
        )
        body_second = ask_second.json()
        assert ask_second.status_code == 200
        assert body_second["cache_status"] == "miss"
        assert body_second["steps"] > 0
        assert body_second["answer"] == "I could not find the answer in the provided documents."
    finally:
        app.dependency_overrides.clear()

def test_ingest_pdf_and_page_number_citations():
    client = _build_client()
    try:
        ingest = client.post(
            "/rag/ingest/pdf",
            files={"file": ("policy.pdf", b"%PDF-sample", "application/pdf")},
            data={"source": "policy-pdf", "doc_id": "doc-pdf"},
        )
        body = ingest.json()

        assert ingest.status_code == 200
        assert body["status"] == "ok"
        assert body["doc_id"] == "doc-pdf"
        assert body["pages_total"] == 2
        assert body["pages_ingested"] == 1
        assert body["skipped_pages"] == [2]
        assert len(body["warnings"]) == 1

        ask = client.post(
            "/agent/ask",
            json={
                "question": "refund rules",
                "doc_id": "doc-pdf",
                "session_id": "s-1",
            },
        )
        ask_body = ask.json()

        assert ask.status_code == 200
        assert ask_body["status"] == "ok"
        assert ask_body["cache_status"] == "miss"
        assert ask_body["refined_query"] == "refund policy for doc-pdf"
        assert len(ask_body["citations"]) >= 1
        assert ask_body["citations"][0]["doc_id"] == "doc-pdf"
        assert ask_body["citations"][0]["source"] == "policy-pdf"
        assert ask_body["citations"][0]["page_number"] == 1
    finally:
        app.dependency_overrides.clear()


def test_documents_routes_soft_delete_and_agent_ownership_enforcement():
    client = _build_client()
    try:
        ingest = client.post(
            "/rag/ingest/text",
            json={"text": "hello", "source": "s", "doc_id": "doc-owned"},
        )
        assert ingest.status_code == 200

        listing = client.get("/documents")
        assert listing.status_code == 200
        assert len(listing.json()["items"]) == 1

        get_doc = client.get("/documents/doc-owned")
        assert get_doc.status_code == 200

        deleted = client.delete("/documents/doc-owned")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

        get_after = client.get("/documents/doc-owned")
        assert get_after.status_code == 404

        ask_after = client.post(
            "/agent/ask",
            json={"question": "hello?", "doc_id": "doc-owned", "session_id": "s-1"},
        )
        assert ask_after.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_ingest_duplicate_doc_id_returns_409_even_if_soft_deleted():
    client = _build_client()
    try:
        first = client.post(
            "/rag/ingest/text",
            json={"text": "v1", "source": "s", "doc_id": "doc-fixed"},
        )
        assert first.status_code == 200

        deleted = client.delete("/documents/doc-fixed")
        assert deleted.status_code == 200

        second = client.post(
            "/rag/ingest/text",
            json={"text": "v2", "source": "s", "doc_id": "doc-fixed"},
        )
        assert second.status_code == 409
    finally:
        app.dependency_overrides.clear()


def test_agent_ask_without_doc_id_returns_422():
    client = _build_client()
    try:
        response = client.post(
            "/agent/ask",
            json={"question": "What is the capital of France?", "session_id": "s-1"},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_ingest_empty_text_returns_422():
    client = _build_client()
    try:
        response = client.post(
            "/rag/ingest/text",
            json={"text": "", "source": "invalid"},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()

