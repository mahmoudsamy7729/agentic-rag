from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_users.exceptions import UserNotExists

from src.modules.users.dependencies import UserManagerDep, fastapi_users, get_jwt_strategy
from src.modules.users.schemas import TokenResponse, UserCreate, UserRead
from src.modules.users.security.jwt import (
    TokenExpiredError,
    TokenValidationError,
    generate_refresh_token,
    verify_refresh_token,
)
from src.settings.config import settings

router = APIRouter(tags=["auth"])


def _set_auth_cookies(response: JSONResponse, *, access_token: str, refresh_token: str) -> None:
    response.set_cookie(
        key=settings.access_cookie_name,
        value=access_token,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        domain=settings.auth_cookie_domain,
        path=settings.auth_cookie_path,
        max_age=settings.access_token_expire_time * 60,
    )
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=refresh_token,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        domain=settings.auth_cookie_domain,
        path=settings.auth_cookie_path,
        max_age=settings.refresh_token_expire_time * 60,
    )


def _clear_auth_cookies(response: JSONResponse) -> None:
    response.delete_cookie(
        key=settings.access_cookie_name,
        domain=settings.auth_cookie_domain,
        path=settings.auth_cookie_path,
    )
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        domain=settings.auth_cookie_domain,
        path=settings.auth_cookie_path,
    )


def _build_token_payload(*, user_id: UUID, email: str | None, token_type: str) -> dict[str, str]:
    payload: dict[str, str] = {"sub": str(user_id), "type": token_type}
    if email:
        payload["email"] = email
    return payload


@router.post("/auth/jwt/login", response_model=TokenResponse, tags=["auth"])
async def login(
    user_manager: UserManagerDep,
    credentials: OAuth2PasswordRequestForm = Depends(),
):
    user = await user_manager.authenticate(credentials)
    if user is None or user.is_active is False:
        response = JSONResponse(
            status_code=401,
            content={"detail": "Invalid credentials."},
        )
        _clear_auth_cookies(response)
        return response

    access_token = await get_jwt_strategy().write_token(user)
    refresh_token, _, _ = generate_refresh_token(
        _build_token_payload(user_id=user.id, email=user.email, token_type="refresh")
    )
    payload = TokenResponse(access_token=access_token).model_dump()
    response = JSONResponse(status_code=200, content=payload)
    _set_auth_cookies(response, access_token=access_token, refresh_token=refresh_token)
    return response


@router.post("/auth/jwt/refresh", response_model=TokenResponse, tags=["auth"])
async def refresh_access_token(
    request: Request,
    user_manager: UserManagerDep,
):
    refresh_token = request.cookies.get(settings.refresh_cookie_name)
    if not refresh_token:
        response = JSONResponse(status_code=401, content={"detail": "Missing refresh token."})
        _clear_auth_cookies(response)
        return response

    try:
        payload = verify_refresh_token(refresh_token)
    except (TokenExpiredError, TokenValidationError):
        response = JSONResponse(status_code=401, content={"detail": "Invalid refresh token."})
        _clear_auth_cookies(response)
        return response

    token_type = str(payload.get("type", "")).strip().lower()
    if token_type != "refresh":
        response = JSONResponse(status_code=401, content={"detail": "Invalid refresh token type."})
        _clear_auth_cookies(response)
        return response

    raw_sub = payload.get("sub")
    try:
        user_id = UUID(str(raw_sub))
    except Exception:
        response = JSONResponse(status_code=401, content={"detail": "Invalid token subject."})
        _clear_auth_cookies(response)
        return response

    try:
        user = await user_manager.get(user_id)
    except UserNotExists:
        response = JSONResponse(status_code=401, content={"detail": "User not found."})
        _clear_auth_cookies(response)
        return response

    if user.is_active is False:
        response = JSONResponse(status_code=401, content={"detail": "Inactive user."})
        _clear_auth_cookies(response)
        return response

    access_token = await get_jwt_strategy().write_token(user)
    # Stateless rotation: issue a new refresh token every refresh.
    refresh_token, _, _ = generate_refresh_token(
        _build_token_payload(user_id=user.id, email=user.email, token_type="refresh")
    )
    payload = TokenResponse(access_token=access_token).model_dump()
    response = JSONResponse(status_code=200, content=payload)
    _set_auth_cookies(response, access_token=access_token, refresh_token=refresh_token)
    return response


@router.post("/auth/jwt/logout", tags=["auth"])
async def logout():
    response = JSONResponse(
        status_code=200,
        content={"status": "ok", "message": "Logged out."},
    )
    _clear_auth_cookies(response)
    return response


router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
