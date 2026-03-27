from __future__ import annotations

import json
from typing import Any, AsyncIterator, Mapping

from openai import AsyncOpenAI

from src.shared.interfaces.llm import (
    ChatMessage,
    GenerationConfig,
    LLM,
    LLMResponse,
    MessageRole,
    TokenUsage,
    ToolCall,
)


class OpenAILLM(LLM):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        organization: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_tool_calls(self) -> bool:
        return True

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        config: GenerationConfig | None = None,
        tools: list[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        cfg = config or GenerationConfig()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_openai_message(msg) for msg in messages],
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        if cfg.stop:
            payload["stop"] = cfg.stop
        if cfg.timeout_s is not None:
            payload["timeout"] = cfg.timeout_s
        if cfg.response_format is not None:
            payload["response_format"] = cfg.response_format
        if tools:
            payload["tools"] = list(tools)

        completion = await self._client.chat.completions.create(**payload)
        choice = completion.choices[0]
        message = choice.message

        usage = TokenUsage(
            prompt_tokens=(completion.usage.prompt_tokens if completion.usage else 0),
            completion_tokens=(
                completion.usage.completion_tokens if completion.usage else 0
            ),
        )

        return LLMResponse(
            content=message.content or "",
            model=completion.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            tool_calls=self._extract_tool_calls(message),
            raw=completion,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        config: GenerationConfig | None = None,
        tools: list[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        cfg = config or GenerationConfig()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_openai_message(msg) for msg in messages],
            "temperature": cfg.temperature,
            "stream": True,
        }
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        if cfg.stop:
            payload["stop"] = cfg.stop
        if cfg.timeout_s is not None:
            payload["timeout"] = cfg.timeout_s
        if cfg.response_format is not None:
            payload["response_format"] = cfg.response_format
        if tools:
            payload["tools"] = list(tools)

        stream = await self._client.chat.completions.create(**payload)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                yield token

    @staticmethod
    def _to_openai_message(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role.value,
            "content": message.content,
        }
        if message.name:
            payload["name"] = message.name
        if message.role == MessageRole.TOOL and message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.role == MessageRole.ASSISTANT and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        return payload

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        raw_calls = getattr(message, "tool_calls", None) or []
        for call in raw_calls:
            arguments_raw = call.function.arguments or "{}"
            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments_raw}

            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )
        return tool_calls
