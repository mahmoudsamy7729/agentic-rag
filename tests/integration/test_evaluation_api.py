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
    chunking_strategy: str | None = "fixed_window"
    chunk_size: int | None = 800
    chunk_overlap: int | None = 120


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
    cfg_rag_top_k: int | None = 12
    cfg_rag_prefetch_k: int | None = 80
    cfg_embedding_provider: str | None = "huggingface"
    cfg_embedding_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    cfg_reranker_enabled: bool | None = True
    cfg_reranker_model: str | None = "rerank-v4.0-fast"
    cfg_answer_model: str | None = "gpt-oss:120b-cloud"
    cfg_chunk_strategy: str | None = "fixed_window"
    cfg_chunk_size: int | None = 800
    cfg_chunk_overlap: int | None = 120
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

    async def get_owned_document(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        include_deleted: bool = False,
    ):
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
            status="failed",
            dataset_name="eval.jsonl",
            dataset_sha256="abc123",
            total_cases=1,
            processed_cases=1,
        )
        self.case = FakeCase(
            id=uuid4(),
            run_id=self.run.id,
            case_index=0,
            question="What is policy?",
            reference_answer="Reference",
            expected_chunk_ids=["chunk-1"],
            status="failed",
            error_message="failed once",
        )
        self.failed_count = 1

    async def create_run_from_dataset(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        dataset_name: str,
        dataset_bytes: bytes,
    ):
        self.last_uploaded_bytes = dataset_bytes
        self.run.doc_id = doc_id
        self.run.dataset_name = dataset_name
        self.run.status = "queued"
        self.run.total_cases = 1
        return self.run

    async def list_runs(self, *, owner_user_id: UUID, doc_id: str | None, limit: int, offset: int):
        if owner_user_id != self.owner_user_id:
            return [], 0
        if doc_id and self.run.doc_id != doc_id:
            return [], 0
        return [self.run], 1

    async def get_run_status(self, *, owner_user_id: UUID, run_id: UUID):
        if owner_user_id != self.owner_user_id or run_id != self.run.id:
            return None
        return self.run

    async def get_owned_run_failed_count(self, *, owner_user_id: UUID, run_id: UUID):
        if owner_user_id != self.owner_user_id or run_id != self.run.id:
            return None
        return self.run, self.failed_count

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


def _build_client(eval_service: FakeEvaluationService, runner: FakeRunner, rerun_runner: FakeRunner):
    owner = FakeUser(id=eval_service.owner_user_id)
    docs_repo = FakeDocumentsRepository(owner.id)

    app.dependency_overrides[active_user] = lambda: owner
    app.dependency_overrides[get_documents_repository] = lambda: docs_repo
    app.dependency_overrides[deps.get_evaluation_service] = lambda: eval_service
    app.dependency_overrides[deps.get_evaluation_job_runner] = lambda: runner
    app.dependency_overrides[deps.get_evaluation_rerun_failed_job_runner] = lambda: rerun_runner
    return TestClient(app)


def test_evaluation_endpoints_status_report_list_and_rerun():
    owner_id = uuid4()
    eval_service = FakeEvaluationService(owner_id)
    runner = FakeRunner()
    rerun_runner = FakeRunner()
    client = _build_client(eval_service, runner, rerun_runner)

    try:
        create = client.post(
            "/evaluations/rag",
            data={"doc_id": "doc-eval"},
            files={
                "file": (
                    "eval.jsonl",
                    b'{"question":"q","answer":"a","expected_chunk_ids":["chunk-1"]}',
                    "application/json",
                )
            },
        )
        assert create.status_code == 202
        run_id = create.json()["run_id"]
        assert str(runner.called_with) == run_id

        status = client.get(f"/evaluations/{run_id}")
        assert status.status_code == 200
        assert status.json()["config"]["chunk_strategy"] == "fixed_window"

        listing = client.get("/evaluations?limit=10&offset=0")
        assert listing.status_code == 200
        assert listing.json()["total"] == 1

        cases = client.get(f"/evaluations/{run_id}/cases?limit=10&offset=0")
        assert cases.status_code == 200
        assert len(cases.json()["items"]) == 1

        report = client.get(f"/evaluations/{run_id}/report")
        assert report.status_code == 200
        assert report.json()["run"]["run_id"] == run_id

        rerun = client.post(f"/evaluations/{run_id}/rerun-failed")
        assert rerun.status_code == 202
        assert rerun.json()["queued_failed_cases"] == 1
        assert str(rerun_runner.called_with) == run_id
    finally:
        app.dependency_overrides.clear()


def test_rerun_rejects_running_run():
    owner_id = uuid4()
    eval_service = FakeEvaluationService(owner_id)
    eval_service.run.status = "running"
    runner = FakeRunner()
    rerun_runner = FakeRunner()
    client = _build_client(eval_service, runner, rerun_runner)

    try:
        response = client.post(f"/evaluations/{eval_service.run.id}/rerun-failed")
        assert response.status_code == 409
    finally:
        app.dependency_overrides.clear()


def test_evaluation_pages_render():
    client = TestClient(app)
    assert client.get("/documents-ui").status_code == 200
    assert client.get("/evaluation-ui").status_code == 200
    assert client.get("/evaluation-history-ui").status_code == 200
    assert client.get(f"/evaluation-runs/{uuid4()}/ui").status_code == 200
