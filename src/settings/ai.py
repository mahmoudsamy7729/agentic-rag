from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class AISettings(BaseSettings):
    openai_key: Optional[str] = Field(default=None)
    ollama_key: Optional[str] = Field(default=None)
    ollama_base_url: Optional[str]= Field(default=None)
    model: Optional[str] = "gpt-oss:120b-cloud"
