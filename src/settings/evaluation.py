from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class EvaluationSettings(BaseSettings):
    evaluation_data_dir: str = Field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[2] / "data" / "evaluations"
        )
    )
    evaluation_judge_enabled: bool = Field(default=True)
    evaluation_judge_model: str = Field(default="gpt-oss:120b-cloud")
    evaluation_judge_base_url: str | None = Field(default=None)
    evaluation_judge_max_tokens: int = Field(default=256, ge=32, le=2048)
    evaluation_judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    evaluation_judge_timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    evaluation_text_strip_punctuation: bool = Field(default=True)
    evaluation_useful_chunk_min_keyword_hits: int = Field(default=2, ge=0, le=100)
    evaluation_useful_chunk_min_keyword_ratio: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
    )
    evaluation_store_retrieved_chunk_texts: bool = Field(default=False)
