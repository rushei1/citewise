import logging
from datetime import UTC, datetime
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query, Response

from app.config import Settings, get_settings
from app.db import ReviewDB, cache_params
from app.schemas import (
    ClusterInfo,
    ErrorResponse,
    Paper,
    ReviewListItem,
    ReviewResponse,
    SearchFilters,
    sorted_cluster_ids,
)
from app.services.cluster import CLUSTERERS, Embedder, EmbedderLike, keyword_labels
from app.services.export import build_bibtex, build_pdf
from app.services.fetch import ArxivSource, FetchError, PaperSource, SemanticScholarSource
from app.services.llm import LLMCache, LLMClient, LLMError, make_llm_client
from app.services.retrieve import retrieve
from app.services.summarize import (
    BACKENDS,
    HFSummarizer,
    LLMSummarizer,
    Summarizer,
    summarize_all,
)
from app.services.theme import ThemeWriter

log = logging.getLogger(__name__)


@lru_cache
def get_llm_client() -> LLMClient | None:
    return make_llm_client(get_settings())


@lru_cache
def get_llm_cache() -> LLMCache:
    return LLMCache(get_settings().llm_cache_path)


@lru_cache
def get_theme_writer() -> ThemeWriter | None:
    client = get_llm_client()
    return ThemeWriter(client, get_llm_cache()) if client else None


@lru_cache
def get_summarizers() -> dict[str, Summarizer]:
    """The summarizer backends that are usable right now, by name."""
    backends: dict[str, Summarizer] = {"hf": HFSummarizer(get_settings().summarizer_model)}
    if client := get_llm_client():
        backends["llm"] = LLMSummarizer(client, get_llm_cache())
    return backends


@lru_cache
def get_embedder() -> EmbedderLike:
    return Embedder(get_settings().embedding_model)


@lru_cache
def get_db() -> ReviewDB:
    return ReviewDB(get_settings().db_path)


@lru_cache
def get_sources() -> dict[str, PaperSource]:
    settings = get_settings()
    return {
        "arxiv": ArxivSource(settings.arxiv_timeout_seconds),
        "semantic_scholar": SemanticScholarSource(api_key=settings.s2_api_key),
    }


def _get_or_404(db: ReviewDB, review_id: int) -> ReviewResponse:
    review = db.get(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found.")
    return review


def _download(body: str | bytes, media_type: str, review_id: int, ext: str) -> Response:
    return Response(
        content=body.encode("utf-8") if isinstance(body, str) else body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="review_{review_id}.{ext}"'},
    )


def _parse_num_clusters(raw: str | None, default: int | str) -> int | None:
    """None means auto. Out-of-range or non-numeric values are a 422, like any bad parameter."""
    value = default if raw is None else raw.strip().lower()
    if value == "auto":
        return None
    try:
        k = int(value)
    except ValueError:
        k = 0
    if not 1 <= k <= 8:
        raise HTTPException(status_code=422, detail="num_clusters must be 1-8 or 'auto'.")
    return k


def _add_themes(
    writer: ThemeWriter, clusters: dict[str, list[Paper]], info: dict[str, ClusterInfo]
) -> list[str]:
    """Fill in theme summaries (skipping the unclustered group); returns warnings."""
    warnings = []
    for cid in sorted_cluster_ids(clusters):
        if cid == "-1":
            continue
        try:
            theme = writer.write(clusters[cid])
        except LLMError as e:
            warnings.append(f"theme for {info[cid].label!r} failed: {e}")
            continue
        if theme is None:
            continue
        info[cid].theme_summary = theme.text
        info[cid].invalid_citations = theme.invalid_citations
        if theme.text is None:
            warnings.append(f"theme for {info[cid].label!r} dropped: no valid citations")
    return warnings


def _parse_sources(raw: str, available: dict[str, PaperSource]) -> list[str]:
    names = list(dict.fromkeys(n.strip().lower() for n in raw.split(",") if n.strip()))
    unknown = [n for n in names if n not in available]
    if not names or unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown source(s) {unknown or raw!r}; choose from {sorted(available)}.",
        )
    return names


def _pick_summarizer(name: str, available: dict[str, Summarizer]) -> Summarizer:
    if name in available:
        return available[name]
    if name in BACKENDS:
        raise HTTPException(status_code=503, detail=f"Summarizer {name!r} is not configured.")
    raise HTTPException(
        status_code=422, detail=f"Unknown summarizer {name!r}; use one of {BACKENDS}."
    )


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
        num_clusters: str | None = Query(None, description="1-8 or auto; defaults to config"),
        cluster_method: str | None = Query(
            None, description=f"One of {', '.join(CLUSTERERS)}; defaults to CLUSTER_METHOD"
        ),
        max_results: int | None = Query(None, ge=1, le=30, description="Defaults to config"),
        year_from: int | None = Query(None, ge=1900, le=2100),
        year_to: int | None = Query(None, ge=1900, le=2100),
        min_citations: int | None = Query(None, ge=0),
        venue_contains: str | None = Query(None, max_length=100),
        sources_param: str | None = Query(
            None, alias="sources", description="Comma list: arxiv, semantic_scholar"
        ),
        summarizer_name: str | None = Query(
            None, alias="summarizer", description="hf or llm; defaults to SUMMARIZER_BACKEND"
        ),
        summarizers: dict[str, Summarizer] = Depends(get_summarizers),
        embedder: EmbedderLike = Depends(get_embedder),
        theme_writer: ThemeWriter | None = Depends(get_theme_writer),
        sources: dict[str, PaperSource] = Depends(get_sources),
        db: ReviewDB = Depends(get_db),
        settings: Settings = Depends(get_settings),
    ):
        topic = topic.strip()
        if not topic:
            raise HTTPException(status_code=422, detail="Topic must not be empty.")
        k = _parse_num_clusters(num_clusters, settings.default_num_clusters)
        method = (cluster_method or settings.cluster_method).strip().lower()
        if method not in CLUSTERERS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown cluster_method {method!r}; use one of {list(CLUSTERERS)}.",
            )
        max_results = max_results or settings.arxiv_max_results

        summarizer = _pick_summarizer(summarizer_name or settings.summarizer_backend, summarizers)
        names = _parse_sources(sources_param or settings.default_sources, sources)
        if year_from and year_to and year_from > year_to:
            raise HTTPException(status_code=422, detail="year_from must be <= year_to.")
        filters = SearchFilters(
            year_from=year_from,
            year_to=year_to,
            min_citations=min_citations,
            venue_contains=(venue_contains or "").strip() or None,
        )
        params = cache_params(
            max_results,
            None if method == "hdbscan" else (k or "auto"),  # hdbscan ignores k
            method,
            summarizer.id,
            settings.embedding_model,
            names,
            filters.model_dump(),
        )
        if cached := db.find_cached(topic, params):
            log.info("cache hit for %r (review %d)", topic, cached.review_id)
            return cached

        try:
            retrieval = retrieve(topic, sources, names, max_results, filters, embedder)
        except FetchError as e:
            log.error("arXiv fetch failed for %r: %s", topic, e)
            raise HTTPException(status_code=502, detail=f"Could not fetch papers: {e}") from e

        papers = retrieval.papers
        log.info("fetched %d papers for %r", len(papers), topic)
        summaries, errors = summarize_all(summarizer, papers)
        if papers and len(errors) == len(papers):
            raise HTTPException(status_code=502, detail=f"Summarization failed: {errors[0]}")
        entries = [
            Paper(
                **p.model_dump(exclude={"abstract"}),
                summary=s.text(),
                structured=s if s.is_structured else None,
            )
            for p, s in zip(papers, summaries, strict=True)
        ]
        # embed the source text, not the lossy summary
        vectors = embedder.encode([f"{p.title}. {p.abstract}" for p in papers])
        labels = CLUSTERERS[method].cluster(vectors, k)

        clusters: dict[str, list[Paper]] = {}
        docs: dict[str, list[str]] = {}
        for entry, p, label in zip(entries, papers, labels, strict=True):
            clusters.setdefault(str(label), []).append(entry)
            docs.setdefault(str(label), []).append(f"{p.title}. {p.abstract}")
        cluster_info = {cid: ClusterInfo(label=lab) for cid, lab in keyword_labels(docs).items()}
        warnings = retrieval.warnings + errors
        if theme_writer and summarizer.id.startswith("llm:"):
            warnings += _add_themes(theme_writer, clusters, cluster_info)
        review = ReviewResponse(
            review_id=0,  # assigned by db.save
            topic=topic,
            created_at=datetime.now(UTC),
            clusters=clusters,
            cluster_info=cluster_info,
            message=None if papers else "No papers found for this topic.",
            warnings=warnings,
        )
        return db.save(topic, params, review)

    @app.get("/reviews", response_model=list[ReviewListItem])
    def list_reviews(db: ReviewDB = Depends(get_db)):
        return db.list_reviews()

    @app.get(
        "/reviews/{review_id}",
        response_model=ReviewResponse,
        responses={404: {"model": ErrorResponse}},
    )
    def get_review(review_id: int, db: ReviewDB = Depends(get_db)):
        return _get_or_404(db, review_id)

    @app.get(
        "/reviews/{review_id}/pdf",
        response_class=Response,
        responses={200: {"content": {"application/pdf": {}}}, 404: {"model": ErrorResponse}},
    )
    def review_pdf(review_id: int, db: ReviewDB = Depends(get_db)):
        review = _get_or_404(db, review_id)
        return Response(
            content=build_pdf(review),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="review_{review_id}.pdf"'},
        )

    @app.get(
        "/reviews/{review_id}/bibtex",
        response_class=Response,
        responses={200: {"content": {"application/x-bibtex": {}}}, 404: {"model": ErrorResponse}},
    )
    def review_bibtex(review_id: int, db: ReviewDB = Depends(get_db)):
        review = _get_or_404(db, review_id)
        return _download(build_bibtex(review), "application/x-bibtex", review_id, "bib")

    @app.delete("/reviews/{review_id}", status_code=204, responses={404: {"model": ErrorResponse}})
    def delete_review(review_id: int, db: ReviewDB = Depends(get_db)):
        if not db.delete(review_id):
            raise HTTPException(status_code=404, detail="Review not found.")
        return Response(status_code=204)

    return app


app = create_app()
