from pathlib import Path

from pydantic_settings import SettingsConfigDict

from src.settings.ai import AISettings
from src.settings.agent import AgentSettings
from src.settings.rag import RAGSettings
from src.settings.database import DatabaseSettings
from src.settings.auth import AuthSettings

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(
    AISettings,
    AgentSettings,
    RAGSettings,
    DatabaseSettings,
    AuthSettings,
):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings() #type: ignore
