from __future__ import annotations

from src.modules.evaluation.judge import EvaluationJudgeService


def test_parse_score_valid_json_payload():
    score = EvaluationJudgeService._parse_score(
        '{"accuracy":4,"completeness":5,"relevance":4,"groundedness":5,"feedback":"good"}'
    )
    assert score.accuracy == 4
    assert score.completeness == 5
    assert score.relevance == 4
    assert score.groundedness == 5
    assert score.feedback == "good"


def test_parse_score_extracts_json_from_wrapped_text():
    score = EvaluationJudgeService._parse_score(
        "Result:\n{\"accuracy\":1,\"completeness\":2,\"relevance\":3,\"groundedness\":4,\"feedback\":\"ok\"}"
    )
    assert score.accuracy == 1
    assert score.completeness == 2
    assert score.relevance == 3
    assert score.groundedness == 4


def test_parse_score_rejects_invalid_score_range():
    try:
        EvaluationJudgeService._parse_score(
            '{"accuracy":6,"completeness":5,"relevance":4,"groundedness":5,"feedback":"bad"}'
        )
        assert False, "Expected ValueError for out-of-range score."
    except ValueError as exc:
        assert "between 1 and 5" in str(exc)


def test_parse_score_rejects_non_json_payload():
    try:
        EvaluationJudgeService._parse_score("not-json")
        assert False, "Expected ValueError for invalid payload."
    except ValueError as exc:
        assert "valid JSON" in str(exc)
