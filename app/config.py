from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    arxiv_max_results: int = 10
    arxiv_timeout_seconds: float = 30.0
    default_sources: str = "arxiv"  # comma list: arxiv, semantic_scholar
    s2_api_key: str | None = None
    default_num_clusters: int = 3
    summarizer_backend: Literal["hf", "llm"] = "hf"
    summarizer_model: str = "facebook/bart-large-cnn"
    # LLM backend (SUMMARIZER_BACKEND=llm). No default model: unset means unavailable.
    llm_provider: Literal["openai", "anthropic"] = "openai"  # openai = any compatible server
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    anthropic_api_key: str | None = None
    llm_timeout_seconds: float = 60.0
    llm_cache_path: Path = Path("data/llm_cache.db")
    embedding_model: str = "all-MiniLM-L6-v2"
    db_path: Path = Path("data/reviews.db")
    api_url: str = "http://127.0.0.1:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
