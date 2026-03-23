from pydantic_settings import BaseSettings
from pydantic import Field


class AuthSettings(BaseSettings):
    algorithm: str = Field(default="HS256")
    access_token_secret_key: str = Field(...)
    access_token_expire_time: int = Field(default=30)
    refresh_token_secret_key: str = Field(...)
    refresh_token_expire_time: int = Field(10080)
    reset_password_token_secret: str = Field(...)
    verification_token_secret: str = Field(...)
