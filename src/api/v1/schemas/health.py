from pydantic import BaseModel, Field


class LLMHealthResponse(BaseModel):
    status: str = Field(description="Overall health status.")
    llm_ok: bool = Field(description="Whether the LLM is reachable and responsive.")
    provider: str = Field(description="LLM provider name.")
    model: str = Field(description="Configured model name.")
    detail: str | None = Field(
        default=None,
        description="Optional failure detail when llm_ok is false.",
    )


class ToolsHealthResponse(BaseModel):
    status: str = Field(description="Overall health status.")
    tools_ok: bool = Field(description="Whether tools registry and execution are healthy.")
    tool_name: str = Field(description="Health-checked tool name.")
    detail: str | None = Field(
        default=None,
        description="Optional failure detail when tools_ok is false.",
    )
    output: dict | None = Field(
        default=None,
        description="Tool output payload for successful checks.",
    )
