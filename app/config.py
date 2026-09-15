from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    database_url: str = "sqlite:///./ai_stylist.db"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_name: str = "clothing_items"


@lru_cache
def get_settings() -> Settings:
    return Settings()
