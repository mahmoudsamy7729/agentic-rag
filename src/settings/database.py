from pydantic_settings import BaseSettings
from pydantic import Field



class DatabaseSettings(BaseSettings):
    database_url: str = Field(...)