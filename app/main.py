import logging
from datetime import UTC, datetime
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query

from app.config import Settings, get_settings
from app.schemas import ErrorResponse, Paper, PaperIn, ReviewResponse
from app.services.cluster import Clusterer, Embedder
from app.services.fetch import FetchError, fetch_papers
from app.services.summarize import Summarizer, SummarizerLike

log = logging.getLogger(__name__)


@lru_cache
def get_summarizer() -> SummarizerLike:
    return Summarizer(get_settings().summarizer_model)


@lru_cache
def get_clusterer() -> Clusterer:
    return Clusterer(Embedder(get_settings().embedding_model))


def get_fetcher():
    return fetch_papers


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = FastAPI(title="Lazy Researcher Toolkit")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/review",
        response_model=ReviewResponse,
        responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    )
    def generate_review(
        topic: str = Query(..., description="Enter research topic"),
        num_clusters: int | None = Query(None, ge=1, le=8, description="Defaults to config"),
        max_results: int | None = Query(None, ge=1, le=30, description="Defaults to config"),
        summarizer: SummarizerLike = Depends(get_summarizer),
        clusterer: Clusterer = Depends(get_clusterer),
        fetcher=Depends(get_fetcher),
        settings: Settings = Depends(get_settings),
    ):
        topic = topic.strip()
        if not topic:
            raise HTTPException(status_code=422, detail="Topic must not be empty.")
        num_clusters = num_clusters or settings.default_num_clusters
        max_results = max_results or settings.arxiv_max_results

        try:
            papers: list[PaperIn] = fetcher(topic, max_results, settings.arxiv_timeout_seconds)
        except FetchError as e:
            log.error("arXiv fetch failed for %r: %s", topic, e)
            raise HTTPException(status_code=502, detail=f"Could not fetch papers: {e}") from e

        log.info("fetched %d papers for %r", len(papers), topic)
        entries = [
            Paper(title=p.title, summary=summarizer.summarize(p.title, p.abstract), link=p.link)
            for p in papers
        ]
        texts = [f"{e.title}\n{e.summary}" for e in entries]
        labels = clusterer.cluster(texts, num_clusters)

        clusters: dict[str, list[Paper]] = {}
        for entry, label in zip(entries, labels, strict=True):
            clusters.setdefault(str(label), []).append(entry)
        # review_id/created_at are placeholders until reviews are persisted.
        return ReviewResponse(
            review_id=0,
            topic=topic,
            created_at=datetime.now(UTC),
            clusters=clusters,
            message=None if papers else "No papers found for this topic.",
        )

    return app


app = create_app()
