from src.modules.evaluation.matching import TextNormalizationConfig, UsefulChunkConfig
from src.modules.evaluation.metrics import compute_retrieval_metrics
from src.rag.models import RetrievedChunk


def test_compute_retrieval_metrics_tracks_phrase_keyword_and_useful_chunks():
    chunks = [
        RetrievedChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            source="policy",
            text="Customers may request a refund within 30 days of purchase.",
            score=0.9,
        ),
        RetrievedChunk(
            doc_id="doc-1",
            chunk_id="chunk-2",
            source="policy",
            text="A refund applies only if the subscription was not used.",
            score=0.8,
        ),
        RetrievedChunk(
            doc_id="doc-1",
            chunk_id="chunk-3",
            source="policy",
            text="Office hours are Monday to Friday.",
            score=0.1,
        ),
    ]

    result = compute_retrieval_metrics(
        retrieved_chunks=chunks,
        must_include_phrases=[
            "refund within 30 days",
            "subscription was not used",
        ],
        must_include_keywords=["refund", "30", "days", "subscription", "used"],
        k=3,
        normalization_config=TextNormalizationConfig(strip_punctuation=True),
        useful_chunk_config=UsefulChunkConfig(min_keyword_hits=2, min_keyword_ratio=0.4),
    )

    assert result.hit_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 2 / 3
    assert result.mrr == 1.0
    assert result.keyword_coverage == 1.0
    assert result.first_correct_rank == 1
    assert result.useful_chunk_count == 2
    assert result.matched_phrases == [
        "refund within 30 days",
        "subscription was not used",
    ]
    assert result.matched_keywords == ["30", "days", "refund", "subscription", "used"]
