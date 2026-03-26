from pydantic import Field
from pydantic_settings import BaseSettings


class EvaluationSettings(BaseSettings):
    eval_judge_model: str | None = Field(default="gpt-oss:120b-cloud")
    eval_max_cases: int = Field(default=500, ge=1, le=5000)
    eval_upload_max_mb: int = Field(default=10, ge=1, le=200)
    eval_poll_interval_ms: int = Field(default=2000, ge=250, le=30000)

