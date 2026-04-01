from fastapi import APIRouter

from src.modules.users.dependencies import fastapi_users
from src.modules.users.schemas import UserRead, UserUpdate

router = APIRouter(tags=["users"])

router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

