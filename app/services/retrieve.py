import logging
from dataclasses import dataclass, field

from app.schemas import PaperIn, SearchFilters
from app.services.fetch import PaperSource

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
    """Query the named sources and return their combined papers."""
    papers: list[PaperIn] = []
    for name in names:
        papers += sources[name].search(query, max_results, filters)
    return Retrieval(papers[:max_results])
