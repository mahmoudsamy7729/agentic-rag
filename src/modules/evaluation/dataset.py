from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class DatasetValidationError(ValueError):
    pass


@dataclass(slots=True)
class RetrievalDatasetItem:
    question: str
    answer: str
    must_include_keywords: list[str]
    must_include_phrases: list[str]
    difficulty: str | None = None
    category: str | None = None


@dataclass(slots=True)
class LoadedRetrievalDataset:
    items: list[RetrievalDatasetItem]
    sha256: str


def load_retrieval_dataset_jsonl(path: str | Path) -> LoadedRetrievalDataset:
    dataset_path = Path(path)
    raw_bytes = dataset_path.read_bytes()
    return parse_retrieval_dataset_jsonl_bytes(raw_bytes)


def parse_retrieval_dataset_jsonl_bytes(raw_bytes: bytes) -> LoadedRetrievalDataset:
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("Dataset must be valid UTF-8 encoded JSONL.") from exc
    items: list[RetrievalDatasetItem] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"Invalid JSON on line {line_number}: {exc.msg}."
            ) from exc
        if not isinstance(payload, dict):
            raise DatasetValidationError(
                f"Invalid dataset item on line {line_number}: expected JSON object."
            )
        items.append(_validate_dataset_item(payload=payload, line_number=line_number))
    if not items:
        raise DatasetValidationError("Dataset is empty.")
    return LoadedRetrievalDataset(items=items, sha256=sha256)


def _validate_dataset_item(*, payload: dict, line_number: int) -> RetrievalDatasetItem:
    question = _require_string(payload, "question", line_number)
    answer = _require_string(payload, "answer", line_number)
    must_include_keywords = _require_string_list(
        payload,
        "must_include_keywords",
        line_number,
    )
    must_include_phrases = _require_string_list(
        payload,
        "must_include_phrases",
        line_number,
    )
    difficulty = _optional_string(payload, "difficulty", line_number)
    category = _optional_string(payload, "category", line_number)
    return RetrievalDatasetItem(
        question=question,
        answer=answer,
        must_include_keywords=must_include_keywords,
        must_include_phrases=must_include_phrases,
        difficulty=difficulty,
        category=category,
    )


def _require_string(payload: dict, field_name: str, line_number: int) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(
            f"Invalid field '{field_name}' on line {line_number}: expected non-empty string."
        )
    return value.strip()


def _optional_string(payload: dict, field_name: str, line_number: int) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetValidationError(
            f"Invalid field '{field_name}' on line {line_number}: expected string or null."
        )
    stripped = value.strip()
    return stripped or None


def _require_string_list(payload: dict, field_name: str, line_number: int) -> list[str]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise DatasetValidationError(
            f"Invalid field '{field_name}' on line {line_number}: expected array of strings."
        )
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DatasetValidationError(
                f"Invalid field '{field_name}' on line {line_number}: expected array of non-empty strings."
            )
        normalized.append(item.strip())
    return normalized
