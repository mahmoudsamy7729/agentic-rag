from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from fastapi.testclient import TestClient

pytest.importorskip("fastapi_users")

from main import app
from src.api.v1 import dependencies as deps
from src.modules.documents.dependencies import get_documents_repository
from src.modules.users.dependencies import active_user


@dataclass
class FakeUser:
    id: UUID
    is_active: bool = True


@dataclass
class FakeDocument:
    id: str
    owner_user_id: UUID
    deleted_at: datetime | None = None


@dataclass
class FakeRun:
    id: UUID
    owner_user_id: UUID
    doc_id: str
    status: str
    dataset_name: str
    dataset_sha256: str
    total_cases: int
    processed_cases: int
    hit_at_k: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    accuracy_avg: float | None = None
    completeness_avg: float | None = None
    relevance_avg: float | None = None
    groundedness_avg: float | None = None
    error_message: str | None = None
    created_at: datetime = datetime.now(timezone.utc)
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass
class FakeCase:
    id: UUID
    run_id: UUID
    case_index: int
    question: str
    reference_answer: str
    expected_chunk_ids: list[str]
    difficulty: str | None = None
    category: str | None = None
    retrieved_chunk_ids: list[str] | None = None
    hit: bool | None = None
    recall: float | None = None
    first_relevant_rank: int | None = None
    reciprocal_rank: float | None = None
    generated_answer: str | None = None
    citations: list[dict] | None = None
    accuracy: int | None = None
    completeness: int | None = None
    relevance: int | None = None
    groundedness: int | None = None
    judge_feedback: str | None = None
    status: str = "queued"
    error_message: str | None = None


class FakeDocumentsRepository:
    def __init__(self, owner_user_id: UUID) -> None:
        self.owner_user_id = owner_user_id

    async def get_owned_document(self, *, owner_user_id: UUID, doc_id: str, include_deleted: bool = False):
        if owner_user_id != self.owner_user_id:
            return None
        return FakeDocument(id=doc_id, owner_user_id=owner_user_id)


class FakeEvaluationService:
    def __init__(self, owner_user_id: UUID) -> None:
        self.owner_user_id = owner_user_id
        self.last_uploaded_bytes: bytes | None = None
        self.run = FakeRun(
            id=uuid4(),
            owner_user_id=owner_user_id,
            doc_id="doc-eval",
            status="queued",
            dataset_name="eval.jsonl",
            dataset_sha256="abc123",
            total_cases=1,
            processed_cases=0,
        )
        self.case = FakeCase(
            id=uuid4(),
            run_id=self.run.id,
            case_index=0,
            question="What is policy?",
            reference_answer="Reference",
            expected_chunk_ids=["chunk-1"],
        )

    async def create_run_from_dataset(self, *, owner_user_id: UUID, doc_id: str, dataset_name: str, dataset_bytes: bytes):
        self.last_uploaded_bytes = dataset_bytes
        self.run.doc_id = doc_id
        self.run.dataset_name = dataset_name
        return self.run

    async def get_run_status(self, *, owner_user_id: UUID, run_id: UUID):
        if owner_user_id != self.owner_user_id or run_id != self.run.id:
            return None
        return self.run

    async def list_run_cases(self, *, owner_user_id: UUID, run_id: UUID, limit: int, offset: int):
        if owner_user_id != self.owner_user_id or run_id != self.run.id:
            return None
        return self.run, [self.case], 1

    async def get_run_report(self, *, owner_user_id: UUID, run_id: UUID):
        if owner_user_id != self.owner_user_id or run_id != self.run.id:
            return None
        return self.run, [self.case]


class FakeRunner:
    def __init__(self) -> None:
        self.called_with: UUID | None = None

    async def __call__(self, run_id: UUID) -> None:
        self.called_with = run_id


def test_evaluation_endpoints_status_and_report():
    owner = FakeUser(id=uuid4())
    docs_repo = FakeDocumentsRepository(owner.id)
    eval_service = FakeEvaluationService(owner.id)
    runner = FakeRunner()

    app.dependency_overrides[active_user] = lambda: owner
    app.dependency_overrides[get_documents_repository] = lambda: docs_repo
    app.dependency_overrides[deps.get_evaluation_service] = lambda: eval_service
    app.dependency_overrides[deps.get_evaluation_job_runner] = lambda: runner

    client = TestClient(app)
    try:
        create = client.post(
            "/evaluations/rag",
            data={"doc_id": "doc-eval"},
            files={"file": ("eval.jsonl", b'{"question":"q","answer":"a","expected_chunk_ids":["chunk-1"]}', "application/json")},
        )
        assert create.status_code == 202
        body = create.json()
        assert body["run_status"] == "queued"
        assert body["total_cases"] == 1
        run_id = body["run_id"]
        assert runner.called_with is not None
        assert str(runner.called_with) == run_id

        status = client.get(f"/evaluations/{run_id}")
        assert status.status_code == 200
        assert status.json()["run_id"] == run_id

        cases = client.get(f"/evaluations/{run_id}/cases?limit=10&offset=0")
        assert cases.status_code == 200
        assert cases.json()["total"] == 1
        assert len(cases.json()["items"]) == 1

        report = client.get(f"/evaluations/{run_id}/report")
        assert report.status_code == 200
        assert report.json()["run"]["run_id"] == run_id
        assert len(report.json()["cases"]) == 1
    finally:
        app.dependency_overrides.clear()


def test_evaluation_ui_page_renders():
    client = TestClient(app)
    response = client.get("/evaluation-ui")
    assert response.status_code == 200
    assert "RAG Evaluation Console" in response.text


