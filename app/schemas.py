from pydantic import BaseModel


class PaperIn(BaseModel):
    """A paper as fetched from arXiv, before summarization."""

    title: str
    abstract: str
    link: str
