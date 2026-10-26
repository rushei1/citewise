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
    papers = deduplicate([p for _, found, _ in results for p in found])
    return Retrieval(
        rank_by_relevance(query, papers, embedder, max_results),
        [f"{name}: {err}" for name, err in errors],
    )
