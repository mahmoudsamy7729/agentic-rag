from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.agents import AgentService
from src.infrastructure.llm.openai_llm import OpenAILLM
from src.settings.config import settings
from src.shared.interfaces.llm import LLM
from src.tools import PingTool, ToolRegistry


@lru_cache
def get_llm() -> LLM:
    if not settings.openai_key:
        raise RuntimeError("Missing OPENAI_KEY in environment.")
    if not settings.model:
        raise RuntimeError("Missing MODEL in environment.")

    return OpenAILLM(
        api_key=settings.openai_key,
        model=settings.model,
        base_url=settings.ollama_base_url,
    )


LLMDep = Annotated[LLM, Depends(get_llm)]


@lru_cache
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry([PingTool()])


ToolRegistryDep = Annotated[ToolRegistry, Depends(get_tool_registry)]


def get_agent_service(llm: LLMDep, registry: ToolRegistryDep) -> AgentService:
    return AgentService(
        llm=llm,
        registry=registry,
        max_steps=settings.agent_max_steps,
        temperature=settings.agent_temperature,
        max_tokens=settings.agent_max_tokens,
        timeout_s=settings.agent_timeout_s,
        system_prompt=settings.agent_system_prompt,
    )


AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
