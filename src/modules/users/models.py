from fastapi_users.db import SQLAlchemyBaseUserTableUUID

from src.infrastructure.database import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    pass

