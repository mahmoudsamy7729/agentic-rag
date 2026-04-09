from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

pytest.importorskip("fastapi_users")

from main import app
from src.api.v1 import dependencies as deps
from src.infrastructure.database import Base
from src.modules.documents.dependencies import get_documents_repository
from src.modules.documents.models import Document
from src.modules.evaluation.judge import ContextRelevanceJudgeResult
from src.modules.evaluation.service import RetrievalEvaluationService
from src.modules.users.dependencies import active_user
from src.modules.users.models import User
from src.rag.models import RetrievedChunk


class FakeRetriever:
    async def retrieve(self, *, question: str, file_id: str, k: int):
        if "refund" in question.lower():
            return [
                RetrievedChunk(
                    doc_id=file_id,
                    chunk_id="chunk-1",
                    source="policy",
                    text="Customers can request a refund within 30 days.",
                    score=0.9,
                ),
                RetrievedChunk(
                    doc_id=file_id,
                    chunk_id="chunk-2",
                    source="policy",
                    text="A refund applies only if the subscription was not used.",
                    score=0.8,
                ),
            ][:k]
        return [
            RetrievedChunk(
                doc_id=file_id,
                chunk_id="chunk-3",
                source="policy",
                text="Support hours are Monday to Friday.",
                score=0.2,
            )
        ][:k]


class FakeJudge:
    async def judge(self, *, question: str, retrieved_chunks: list[dict[str, str]]):
        return ContextRelevanceJudgeResult(
            score=4,
            explanation="The retrieved context is mostly sufficient.",
        )


@dataclass
class FakeUser:
    id: UUID
    is_active: bool = True


@dataclass
class FakeDoc:
    id: str
    owner_user_id: UUID
    source: str | None = None
    created_at: datetime = datetime.now(timezone.utc)
    updated_at: datetime = datetime.now(timezone.utc)
    deleted_at: datetime | None = None


class FakeDocumentsRepository:
    def __init__(self, *, owner_user_id: UUID, doc_id: str) -> None:
        self._owner_user_id = owner_user_id
        self._doc_id = doc_id

    async def get_owned_document(self, *, owner_user_id: UUID, doc_id: str, include_deleted: bool = False):
        if owner_user_id != self._owner_user_id or doc_id != self._doc_id:
            return None
        return FakeDoc(id=doc_id, owner_user_id=owner_user_id, source="policy")


def test_retrieval_evaluation_api_flow_and_judge_disabled():
    temp_db = Path(tempfile.mkdtemp(prefix="eval-api-", dir=".")) / "test.db"
    database_url = f"sqlite+aiosqlite:///{temp_db.as_posix()}"
    dataset_dir = Path(tempfile.mkdtemp(prefix="eval-api-data-", dir="."))
    owner_id = uuid4()

    async def _seed(session_factory):
        async with session_factory() as session:
            session.add(
                User(
                    id=owner_id,
                    email="owner@example.com",
                    hashed_password="hashed",
                    is_active=True,
                    is_superuser=False,
                    is_verified=False,
                )
            )
            session.add(Document(id="doc-1", owner_user_id=owner_id, source="policy"))
            await session.commit()

    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    import asyncio

    asyncio.run(_prepare_db(engine, session_factory, _seed))

    service = RetrievalEvaluationService(
        session_factory=session_factory,
        dataset_storage_dir=str(dataset_dir),
        retriever_factory=lambda: FakeRetriever(),
        judge_factory=lambda: FakeJudge(),
    )
    app.dependency_overrides[deps.get_retrieval_evaluation_service] = lambda: service
    app.dependency_overrides[active_user] = lambda: FakeUser(id=owner_id)
    app.dependency_overrides[get_documents_repository] = lambda: FakeDocumentsRepository(
        owner_user_id=owner_id,
        doc_id="doc-1",
    )

    client = TestClient(app)
    dataset_bytes = (
        b'{"question":"What is the refund policy?","answer":"Users can request a refund within 30 days if the subscription was not used.","must_include_keywords":["refund","30","days","subscription","used"],"must_include_phrases":["refund within 30 days","subscription was not used"],"difficulty":"easy","category":"billing"}\n'
    )

    try:
        create = client.post(
            "/evaluations/retrieval",
            data={"file_id": "doc-1", "k": "2", "judge_enabled": "false"},
            files={"dataset_file": ("dataset.jsonl", dataset_bytes, "application/json")},
        )
        assert create.status_code == 202
        run_id = create.json()["item"]["run_id"]

        listing = client.get("/evaluations")
        assert listing.status_code == 200
        assert len(listing.json()["items"]) == 1

        detail = client.get(f"/evaluations/{run_id}")
        assert detail.status_code == 200
        detail_body = detail.json()
        assert detail_body["item"]["file_id"] == "doc-1"
        assert detail_body["item"]["metrics"]["hit_at_k_avg"] == 1.0

        cases = client.get(f"/evaluations/{run_id}/cases")
        assert cases.status_code == 200
        case_body = cases.json()
        assert case_body["total"] == 1
        assert case_body["items"][0]["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]
        assert case_body["items"][0]["context_relevance_score"] is None
        assert case_body["items"][0]["context_relevance_explanation"] is None

        bad_dataset = client.post(
            "/evaluations/retrieval",
            data={"file_id": "doc-1", "k": "2"},
            files={"dataset_file": ("bad.jsonl", b"not-json", "application/json")},
        )
        assert bad_dataset.status_code == 422

        missing_doc = client.post(
            "/evaluations/retrieval",
            data={"file_id": "missing-doc", "k": "2"},
            files={"dataset_file": ("dataset.jsonl", dataset_bytes, "application/json")},
        )
        assert missing_doc.status_code == 404
    finally:
        app.dependency_overrides.clear()
        asyncio.run(engine.dispose())


async def _prepare_db(engine, session_factory, seed_fn):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_fn(session_factory)
