import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Protocol

import arxiv

from app.schemas import PaperIn, SearchFilters
from app.services.http import Http, HttpError

log = logging.getLogger(__name__)


class FetchError(Exception):
    """A source could not be reached or returned garbage."""


class PaperSource(Protocol):
    name: str

    def search(self, query: str, max_results: int, filters: SearchFilters) -> list[PaperIn]: ...


def _arxiv_id(entry_id: str) -> str:
    """'http://arxiv.org/abs/2303.15105v1' -> '2303.15105'"""
    return re.sub(r"v\d+$", "", entry_id.rsplit("/abs/", 1)[-1])


def _arxiv_search(query: str, max_results: int) -> list[PaperIn]:
    client = arxiv.Client(num_retries=1, delay_seconds=1.0)
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    return [
        PaperIn(
            title=p.title.strip(),
            abstract=p.summary.strip(),
            link=p.entry_id,
            authors=[a.name for a in p.authors],
            year=p.published.year if p.published else None,
            venue=p.journal_ref,
            doi=p.doi,
            arxiv_id=_arxiv_id(p.entry_id),
            source="arxiv",
        )
        for p in client.results(search)
    ]


class ArxivSource:
    name = "arxiv"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def search(self, query: str, max_results: int, filters: SearchFilters) -> list[PaperIn]:
        # arXiv has no usable year/citation filters here; they are applied after merging.
        # arxiv.py has no request timeout, so bound the call from outside.
        # ponytail: a timed-out request keeps its worker thread until it returns.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(_arxiv_search, query, max_results).result(timeout=self.timeout)
        except FutureTimeout as e:
            raise FetchError(f"arXiv did not respond within {self.timeout:.0f}s") from e
        except Exception as e:
            raise FetchError(f"arXiv request failed: {e}") from e
        finally:
            pool.shutdown(wait=False)


S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,citationCount,venue,externalIds,url,authors"
S2_MAX_LIMIT = 100


class SemanticScholarSource:
    name = "semantic_scholar"

    def __init__(self, http: Http | None = None, api_key: str | None = None):
        self.http = http or Http()
        self.api_key = api_key

    def search(self, query: str, max_results: int, filters: SearchFilters) -> list[PaperIn]:
        params: dict[str, str | int] = {
            "query": query,
            "limit": min(max_results, S2_MAX_LIMIT),
            "fields": S2_FIELDS,
        }
        # Pushdown only narrows what S2 returns; filters are re-applied after merging.
        if filters.year_from is not None or filters.year_to is not None:
            params["year"] = f"{filters.year_from or ''}-{filters.year_to or ''}"
        if filters.min_citations is not None:
            params["minCitationCount"] = filters.min_citations
        headers = {"x-api-key": self.api_key} if self.api_key else None
        try:
            data = self.http.get_json(S2_URL, params=params, headers=headers)
        except HttpError as e:
            raise FetchError(f"Semantic Scholar request failed: {e}") from e
        return [p for item in data.get("data") or [] if (p := self._to_paper(item))]

    @staticmethod
    def _to_paper(item: dict) -> PaperIn | None:
        abstract = (item.get("abstract") or "").strip()
        title = (item.get("title") or "").strip()
        if not abstract or not title:
            return None  # nothing to summarize
        ids = item.get("externalIds") or {}
        return PaperIn(
            title=title,
            abstract=abstract,
            link=item.get("url") or f"https://www.semanticscholar.org/paper/{item['paperId']}",
            authors=[a["name"] for a in item.get("authors") or [] if a.get("name")],
            year=item.get("year"),
            venue=item.get("venue") or None,
            citation_count=item.get("citationCount"),
            doi=ids.get("DOI"),
            arxiv_id=ids.get("ArXiv"),
            source="semantic_scholar",
        )
