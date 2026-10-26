from datetime import datetime

from pydantic import BaseModel


class PaperIn(BaseModel):
    """A paper as fetched from a source, before summarization. Only the first three are required."""

    title: str
    abstract: str
    link: str
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source: str | None = None


class SearchFilters(BaseModel):
    """Applied after sources are merged. None means no constraint."""

    year_from: int | None = None
    year_to: int | None = None
    min_citations: int | None = None
    venue_contains: str | None = None


class Paper(BaseModel):
    title: str
    summary: str
    link: str
    # Additive metadata (v1.1 clients can ignore it)
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source: str | None = None


class ReviewResponse(BaseModel):
    review_id: int
    topic: str
    created_at: datetime
    cached: bool = False
    clusters: dict[str, list[Paper]]
    message: str | None = None
    warnings: list[str] = []


class ReviewListItem(BaseModel):
    id: int
    topic: str
    created_at: datetime


class ErrorResponse(BaseModel):
    detail: str
