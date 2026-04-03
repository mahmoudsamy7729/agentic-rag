from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from src.settings.config import settings

TRACE_LOGGER_NAME = "agentic_rag.trace"
_logger = logging.getLogger(TRACE_LOGGER_NAME)


@dataclass(slots=True)
class TraceContext:
    request_id: str
    doc_id: str | None = None
    owner_user_id: str | None = None
    session_id: str | None = None


def configure_trace_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


def trace_event(
    event: str,
    *,
    trace_context: TraceContext | None = None,
    **fields: Any,
) -> None:
    if not settings.tracing_enabled:
        return

    payload: dict[str, Any] = {"event": event}
    if trace_context is not None:
        payload.update({k: v for k, v in asdict(trace_context).items() if v is not None})
    payload.update({k: v for k, v in fields.items() if v is not None})

    if not settings.tracing_include_query_text:
        payload.pop("question", None)
        payload.pop("query", None)
        payload.pop("refined_query", None)

    _logger.info(json.dumps(payload, default=_json_default, sort_keys=True))


def chunk_metadata(chunks: list[Any]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata.append(
            {
                "chunk_id": getattr(chunk, "chunk_id", None),
                "doc_id": getattr(chunk, "doc_id", None),
                "page_number": getattr(chunk, "page_number", None),
                "score": getattr(chunk, "score", None),
                "source": getattr(chunk, "source", None),
            }
        )
    return metadata


def _json_default(value: Any) -> str:
    return str(value)
