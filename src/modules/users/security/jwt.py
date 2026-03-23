from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jose import JWTError, ExpiredSignatureError, jwt

from src.settings.config import settings


class TokenExpiredError(Exception):
    """Raised when a token is expired."""


class TokenValidationError(Exception):
    """Raised when a token is malformed or invalid."""


def generate_token(
    *,
    data: dict[str, Any],
    mins: int,
    secret_key: str,
) -> tuple[str, str, datetime]:
    if mins <= 0:
        raise ValueError("mins must be greater than 0.")
    if not secret_key:
        raise ValueError("secret_key is required.")

    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=mins)
    jti = str(uuid4())
    to_encode.update({"exp": expire, "iat": now, "jti": jti})
    token = jwt.encode(to_encode, secret_key, algorithm=settings.algorithm)
    return token, jti, expire


def verify_token(*, token: str, secret_key: str) -> dict[str, Any]:
    if not token:
        raise TokenValidationError("Token is required.")
    if not secret_key:
        raise TokenValidationError("secret_key is required.")

    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            secret_key,
            algorithms=[settings.algorithm],
        )
    except ExpiredSignatureError as exc:
        raise TokenExpiredError("Token has expired.") from exc
    except JWTError as exc:
        raise TokenValidationError("Token is invalid.") from exc

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub.strip():
        raise TokenValidationError("Token payload is missing required 'sub'.")
    return payload


def generate_access_token(data: dict[str, Any]) -> tuple[str, str, datetime]:
    return generate_token(
        data=data,
        mins=settings.access_token_expire_time,
        secret_key=settings.access_token_secret_key,
    )


def generate_refresh_token(data: dict[str, Any]) -> tuple[str, str, datetime]:
    return generate_token(
        data=data,
        mins=settings.refresh_token_expire_time,
        secret_key=settings.refresh_token_secret_key,
    )


def verify_access_token(token: str) -> dict[str, Any]:
    return verify_token(token=token, secret_key=settings.access_token_secret_key)


def verify_refresh_token(token: str) -> dict[str, Any]:
    return verify_token(token=token, secret_key=settings.refresh_token_secret_key)
