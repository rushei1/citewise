import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import arxiv

from app.schemas import PaperIn

log = logging.getLogger(__name__)


class FetchError(Exception):
    """arXiv could not be reached or returned garbage."""


def _search(topic: str, max_results: int) -> list[PaperIn]:
    client = arxiv.Client(num_retries=1, delay_seconds=1.0)
    search = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    return [
        PaperIn(title=p.title.strip(), abstract=p.summary.strip(), link=p.entry_id)
        for p in client.results(search)
    ]


def fetch_papers(topic: str, max_results: int, timeout: float = 30.0) -> list[PaperIn]:
    # arxiv.py has no request timeout, so bound the call from outside.
    # ponytail: a timed-out request keeps its worker thread until it returns.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(_search, topic, max_results).result(timeout=timeout)
    except FutureTimeout as e:
        raise FetchError(f"arXiv did not respond within {timeout:.0f}s") from e
    except Exception as e:
        log.warning("arXiv fetch failed: %s", e)
        raise FetchError(f"arXiv request failed: {e}") from e
    finally:
        pool.shutdown(wait=False)
