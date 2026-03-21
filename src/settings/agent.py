from pydantic import Field
from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    agent_max_steps: int = Field(default=6, ge=1, le=20)
    agent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    agent_max_tokens: int | None = Field(default=None, ge=1)
    agent_timeout_s: float | None = Field(default=30.0, gt=0.0)
    agent_system_prompt: str = Field(
        default=(
            "You are an agentic assistant. "
            "Use available tools when needed, then provide a final answer."
        )
    )
