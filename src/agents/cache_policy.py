import re


_NO_ANSWER_CANONICAL_PHRASES = (
    "i could not find the answer in the provided documents.",
    "no relevant context found.",
)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def is_no_answer_fallback(answer: str) -> bool:
    normalized_answer = _normalize_text(answer)
    return any(
        phrase in normalized_answer for phrase in _NO_ANSWER_CANONICAL_PHRASES
    )


def is_cacheable_rag_answer(*, tools_used: list[str], citations: list[dict]) -> bool:
    return (
        bool(citations)
        and bool(tools_used)
        and all(tool_name == "retrieve_context" for tool_name in tools_used)
    )
