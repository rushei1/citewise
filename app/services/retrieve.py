import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.schemas import PaperIn, SearchFilters
from app.services.fetch import FetchError, PaperSource

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
) -> Retrieval:
    """Query the named sources in parallel and combine their papers.

    One source failing is a warning, not an error; only if every source fails do we raise."""

    def run(name: str):
        try:
            return name, sources[name].search(query, max_results, filters), None
        except FetchError as e:
            log.warning("source %s failed: %s", name, e)
            return name, [], str(e)

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        results = list(pool.map(run, names))  # keeps the order of `names`

    errors = [(name, err) for name, _, err in results if err]
    if len(errors) == len(names):
        raise FetchError("; ".join(f"{name}: {err}" for name, err in errors))
    papers = [p for _, found, _ in results for p in found]
    return Retrieval(papers[:max_results], [f"{name}: {err}" for name, err in errors])
