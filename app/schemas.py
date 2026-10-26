from datetime import datetime

from pydantic import BaseModel


class PaperIn(BaseModel):
    """A paper as fetched from arXiv, before summarization."""

    title: str
    abstract: str
    link: str


class Paper(BaseModel):
    title: str
    summary: str
    link: str


class ReviewResponse(BaseModel):
    review_id: int
    topic: str
    created_at: datetime
    cached: bool = False
    clusters: dict[str, list[Paper]]
    message: str | None = None


class ReviewListItem(BaseModel):
    id: int
    topic: str
    created_at: datetime


class ErrorResponse(BaseModel):
    detail: str
