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
    access_cookie_name: str = Field(default="access_token")
    refresh_cookie_name: str = Field(default="refresh_token")
    auth_cookie_secure: bool = Field(default=False)
    auth_cookie_samesite: str = Field(default="lax")
    auth_cookie_domain: str | None = Field(default=None)
    auth_cookie_path: str = Field(default="/")
