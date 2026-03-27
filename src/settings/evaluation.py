from pydantic import Field
from pydantic_settings import BaseSettings


class EvaluationSettings(BaseSettings):
    eval_judge_model: str | None = Field(default="gpt-oss:120b-cloud")
    eval_judge_max_tokens: int = Field(default=600, ge=50, le=4000)
    eval_judge_timeout_s: float = Field(default=60.0, gt=1.0, le=300.0)
    eval_max_cases: int = Field(default=500, ge=1, le=5000)
    eval_upload_max_mb: int = Field(default=10, ge=1, le=200)
    eval_poll_interval_ms: int = Field(default=2000, ge=250, le=30000)

