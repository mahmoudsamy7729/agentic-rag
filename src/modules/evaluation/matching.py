from __future__ import annotations

import re
import string
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextNormalizationConfig:
    strip_punctuation: bool = True


@dataclass(frozen=True, slots=True)
class UsefulChunkConfig:
    min_keyword_hits: int = 2
    min_keyword_ratio: float = 0.4


def normalize_text(text: str, *, config: TextNormalizationConfig) -> str:
    normalized = text.strip().lower()
    if config.strip_punctuation:
        normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize_text(text: str, *, config: TextNormalizationConfig) -> set[str]:
    normalized = normalize_text(text, config=config)
    if not normalized:
        return set()
    return {token for token in normalized.split(" ") if token}


def matched_phrases(
    *,
    phrases: list[str],
    chunk_texts: list[str],
    config: TextNormalizationConfig,
) -> list[str]:
    normalized_chunks = [normalize_text(text, config=config) for text in chunk_texts]
    matches: list[str] = []
    for phrase in phrases:
        needle = normalize_text(phrase, config=config)
        if needle and any(needle in chunk for chunk in normalized_chunks):
            matches.append(phrase)
    return matches


def matched_keywords(
    *,
    keywords: list[str],
    chunk_texts: list[str],
    config: TextNormalizationConfig,
) -> list[str]:
    chunk_tokens: set[str] = set()
    for text in chunk_texts:
        chunk_tokens.update(tokenize_text(text, config=config))

    matches: list[str] = []
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword, config=config)
        if normalized_keyword and normalized_keyword in chunk_tokens:
            matches.append(keyword)
    return matches


def chunk_phrase_matches(
    *,
    phrases: list[str],
    chunk_text: str,
    config: TextNormalizationConfig,
) -> list[str]:
    normalized_chunk = normalize_text(chunk_text, config=config)
    matches: list[str] = []
    for phrase in phrases:
        needle = normalize_text(phrase, config=config)
        if needle and needle in normalized_chunk:
            matches.append(phrase)
    return matches


def chunk_keyword_matches(
    *,
    keywords: list[str],
    chunk_text: str,
    config: TextNormalizationConfig,
) -> list[str]:
    tokens = tokenize_text(chunk_text, config=config)
    matches: list[str] = []
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword, config=config)
        if normalized_keyword and normalized_keyword in tokens:
            matches.append(keyword)
    return matches


def is_useful_chunk(
    *,
    chunk_text: str,
    phrases: list[str],
    keywords: list[str],
    normalization_config: TextNormalizationConfig,
    useful_chunk_config: UsefulChunkConfig,
) -> bool:
    phrase_hits = chunk_phrase_matches(
        phrases=phrases,
        chunk_text=chunk_text,
        config=normalization_config,
    )
    if phrase_hits:
        return True

    keyword_hits = chunk_keyword_matches(
        keywords=keywords,
        chunk_text=chunk_text,
        config=normalization_config,
    )
    if not keyword_hits:
        return False

    unique_expected_keywords = {
        normalize_text(keyword, config=normalization_config)
        for keyword in keywords
        if normalize_text(keyword, config=normalization_config)
    }
    hit_count = len(
        {
            normalize_text(keyword, config=normalization_config)
            for keyword in keyword_hits
        }
    )
    ratio = (
        hit_count / len(unique_expected_keywords)
        if unique_expected_keywords
        else 0.0
    )
    return (
        hit_count >= useful_chunk_config.min_keyword_hits
        or ratio >= useful_chunk_config.min_keyword_ratio
    )
