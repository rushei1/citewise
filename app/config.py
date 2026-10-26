from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    arxiv_max_results: int = 10
    arxiv_timeout_seconds: float = 30.0
    default_sources: str = "arxiv"  # comma list: arxiv, semantic_scholar
    s2_api_key: str | None = None
    default_num_clusters: int = 3
    summarizer_model: str = "facebook/bart-large-cnn"
    embedding_model: str = "all-MiniLM-L6-v2"
    db_path: Path = Path("data/reviews.db")
    api_url: str = "http://127.0.0.1:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
