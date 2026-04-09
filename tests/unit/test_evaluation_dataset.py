import pytest

from src.modules.evaluation.dataset import (
    DatasetValidationError,
    parse_retrieval_dataset_jsonl_bytes,
)


def test_parse_retrieval_dataset_jsonl_bytes_accepts_valid_items():
    raw = (
        b'{"question":"What is the refund policy?","answer":"Users can request a refund within 30 days if the subscription was not used.","must_include_keywords":["refund","30","days","subscription","used"],"must_include_phrases":["refund within 30 days","subscription was not used"],"difficulty":"easy","category":"billing"}\n'
    )

    dataset = parse_retrieval_dataset_jsonl_bytes(raw)

    assert len(dataset.items) == 1
    assert dataset.items[0].question == "What is the refund policy?"
    assert dataset.items[0].must_include_keywords == [
        "refund",
        "30",
        "days",
        "subscription",
        "used",
    ]
    assert dataset.sha256


def test_parse_retrieval_dataset_jsonl_bytes_reports_line_number_for_bad_json():
    with pytest.raises(DatasetValidationError, match="line 2"):
        parse_retrieval_dataset_jsonl_bytes(
            b'{"question":"ok","answer":"ok","must_include_keywords":["a"],"must_include_phrases":["b"]}\nnot-json\n'
        )


def test_parse_retrieval_dataset_jsonl_bytes_rejects_missing_required_fields():
    with pytest.raises(DatasetValidationError, match="must_include_phrases"):
        parse_retrieval_dataset_jsonl_bytes(
            b'{"question":"q","answer":"a","must_include_keywords":["a"]}\n'
        )
