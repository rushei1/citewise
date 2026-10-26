import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.schemas import PaperIn, SearchFilters
from app.services.cluster import EmbedderLike
from app.services.dedup import deduplicate
from app.services.fetch import FetchError, PaperSource
from app.services.rank import rank_by_relevance

CANDIDATE_FACTOR = 3  # fetch this many times max_results per source, then rank down
MAX_CANDIDATES = 90

log = logging.getLogger(__name__)


def apply_filters(papers: list[PaperIn], f: SearchFilters) -> list[PaperIn]:
    """Drop papers that fail a filter. A paper missing the field a filter needs (e.g. no
    citation count from arXiv) cannot be shown to satisfy it, so it is dropped too."""

    def ok(p: PaperIn) -> bool:
        if f.year_from is not None and (p.year is None or p.year < f.year_from):
            return False
        if f.year_to is not None and (p.year is None or p.year > f.year_to):
            return False
        if f.min_citations is not None and (p.citation_count or 0) < f.min_citations:
            return False
        if f.venue_contains and f.venue_contains.lower() not in (p.venue or "").lower():
            return False
        return True

    return [p for p in papers if ok(p)]


@dataclass
class Retrieval:
    papers: list[PaperIn]
    warnings: list[str] = field(default_factory=list)


def retrieve(
    query: str,
    sources: dict[str, PaperSource],
    names: list[str],
    max_results: int,
    filters: SearchFilters,
    embedder: EmbedderLike,
) -> Retrieval:
    """Query the named sources in parallel, merge duplicates and keep the most relevant papers.

    One source failing is a warning, not an error; only if every source fails do we raise."""

    pool_size = min(max_results * CANDIDATE_FACTOR, MAX_CANDIDATES)

    def run(name: str):
        try:
            return name, sources[name].search(query, pool_size, filters), None
        except FetchError as e:
            log.warning("source %s failed: %s", name, e)
            return name, [], str(e)

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        results = list(pool.map(run, names))  # keeps the order of `names`

    errors = [(name, err) for name, _, err in results if err]
    if len(errors) == len(names):
        raise FetchError("; ".join(f"{name}: {err}" for name, err in errors))
    papers = apply_filters(deduplicate([p for _, found, _ in results for p in found]), filters)
    return Retrieval(
        rank_by_relevance(query, papers, embedder, max_results),
        [f"{name}: {err}" for name, err in errors],
    )
