import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, Query

from app.schemas import PaperIn
from app.services.cluster import Clusterer, Embedder
from app.services.fetch import fetch_papers
from app.services.summarize import Summarizer, SummarizerLike

log = logging.getLogger(__name__)


@lru_cache
def get_summarizer() -> SummarizerLike:
    return Summarizer()


@lru_cache
def get_clusterer() -> Clusterer:
    return Clusterer(Embedder())


def get_fetcher():
    return fetch_papers


def create_app() -> FastAPI:
    app = FastAPI(title="Lazy Researcher Toolkit")

    @app.get("/review")
    def generate_review(
        topic: str = Query(..., description="Enter research topic"),
        summarizer: SummarizerLike = Depends(get_summarizer),
        clusterer: Clusterer = Depends(get_clusterer),
        fetcher=Depends(get_fetcher),
    ):
        papers: list[PaperIn] = fetcher(topic, 10)
        entries = [
            {
                "title": p.title,
                "summary": summarizer.summarize(p.title, p.abstract),
                "link": p.link,
            }
            for p in papers
        ]
        labels = clusterer.cluster([f"{e['title']}\n{e['summary']}" for e in entries])

        clusters: dict[str, list[dict]] = {}
        for entry, label in zip(entries, labels, strict=True):
            clusters.setdefault(str(label), []).append(entry)
        return {"topic": topic, "clusters": clusters}

    return app


app = create_app()
