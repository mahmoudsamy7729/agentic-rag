def is_cacheable_rag_answer(*, tools_used: list[str], citations: list[dict]) -> bool:
    return (
        bool(citations)
        and bool(tools_used)
        and all(tool_name == "retrieve_context" for tool_name in tools_used)
    )
