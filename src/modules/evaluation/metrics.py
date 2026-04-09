from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from src.modules.evaluation.matching import (
    TextNormalizationConfig,
    UsefulChunkConfig,
    chunk_phrase_matches,
    is_useful_chunk,
    matched_keywords,
    matched_phrases,
)
from src.rag.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrievalMetricResult:
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float
    mrr: float
    keyword_coverage: float
    matched_phrases: list[str]
    matched_keywords: list[str]
    first_correct_rank: int | None
    useful_chunk_count: int


def compute_retrieval_metrics(
    *,
    retrieved_chunks: list[RetrievedChunk],
    must_include_phrases: list[str],
    must_include_keywords: list[str],
    k: int,
    normalization_config: TextNormalizationConfig,
    useful_chunk_config: UsefulChunkConfig,
) -> RetrievalMetricResult:
    chunk_texts = [chunk.text for chunk in retrieved_chunks]
    matched_phrase_values = matched_phrases(
        phrases=must_include_phrases,
        chunk_texts=chunk_texts,
        config=normalization_config,
    )
    matched_keyword_values = matched_keywords(
        keywords=must_include_keywords,
        chunk_texts=chunk_texts,
        config=normalization_config,
    )
    first_correct_rank = _first_correct_rank(
        retrieved_chunks=retrieved_chunks,
        must_include_phrases=must_include_phrases,
        normalization_config=normalization_config,
    )
    useful_chunk_count = sum(
        1
        for chunk in retrieved_chunks
        if is_useful_chunk(
            chunk_text=chunk.text,
            phrases=must_include_phrases,
            keywords=must_include_keywords,
            normalization_config=normalization_config,
            useful_chunk_config=useful_chunk_config,
        )
    )
    total_phrase_count = len(must_include_phrases)
    total_keyword_count = len(set(must_include_keywords))
    recall = (
        len(matched_phrase_values) / total_phrase_count
        if total_phrase_count > 0
        else 0.0
    )
    keyword_coverage = (
        len(set(matched_keyword_values)) / total_keyword_count
        if total_keyword_count > 0
        else 0.0
    )
    return RetrievalMetricResult(
        hit_at_k=1.0 if matched_phrase_values else 0.0,
        recall_at_k=recall,
        precision_at_k=(useful_chunk_count / k) if k > 0 else 0.0,
        mrr=(1.0 / first_correct_rank) if first_correct_rank is not None else 0.0,
        keyword_coverage=keyword_coverage,
        matched_phrases=matched_phrase_values,
        matched_keywords=sorted(set(matched_keyword_values)),
        first_correct_rank=first_correct_rank,
        useful_chunk_count=useful_chunk_count,
    )


def aggregate_metric_average(values: Iterable[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return mean(filtered)


def grouped_metric_summary(*, cases: list[dict], field_name: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for case in cases:
        key = case.get(field_name)
        if key is None:
            continue
        grouped.setdefault(str(key), []).append(case)

    summary: dict[str, dict] = {}
    for key, items in grouped.items():
        summary[key] = {
            "count": len(items),
            "hit_at_k_avg": aggregate_metric_average(item.get("hit_at_k") for item in items),
            "recall_at_k_avg": aggregate_metric_average(item.get("recall_at_k") for item in items),
            "precision_at_k_avg": aggregate_metric_average(item.get("precision_at_k") for item in items),
            "mrr_avg": aggregate_metric_average(item.get("mrr") for item in items),
            "keyword_coverage_avg": aggregate_metric_average(
                item.get("keyword_coverage") for item in items
            ),
            "context_relevance_score_avg": aggregate_metric_average(
                item.get("context_relevance_score") for item in items
            ),
        }
    return summary


def _first_correct_rank(
    *,
    retrieved_chunks: list[RetrievedChunk],
    must_include_phrases: list[str],
    normalization_config: TextNormalizationConfig,
) -> int | None:
    for index, chunk in enumerate(retrieved_chunks, start=1):
        matches = chunk_phrase_matches(
            phrases=must_include_phrases,
            chunk_text=chunk.text,
            config=normalization_config,
        )
        if matches:
            return index
    return None
